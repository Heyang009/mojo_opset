from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from mojo_opset.core.operator import MojoOperator
from mojo_opset.core.operators.moe import _count_expert_tokens
from mojo_opset.core.operators.moe import MojoMoECombine
from mojo_opset.core.operators.moe import MojoMoEDispatch
from mojo_opset.core.operators.moe import MojoMoEGating
from mojo_opset.core.operators.moe import MojoQuantExperts


def _validate_moe_token_count(token_count: torch.Tensor, route_count: int) -> torch.Tensor:
    token_count_i64 = token_count.to(dtype=torch.int64, device=token_count.device)
    if token_count_i64.dim() != 1:
        raise ValueError(f"token_count must be 1D, but got shape {tuple(token_count.shape)}")
    if int(token_count_i64.sum().item()) != route_count:
        raise ValueError(
            f"token_count sum must equal total routed token count {route_count}, "
            f"but got {token_count_i64.sum().item()}."
        )
    return token_count_i64


def _expand_grouped_route_param(
    param: Optional[torch.Tensor],
    token_count: torch.Tensor,
    route_shape: tuple[int, int],
) -> Optional[torch.Tensor]:
    if param is None:
        return None

    token_count_i64 = _validate_moe_token_count(token_count, route_shape[0] * route_shape[1])
    param_fp = param.float()

    if param_fp.dim() == 1:
        return param_fp.view(1, 1, -1).expand(*route_shape, -1)
    if param_fp.dim() != 2 or param_fp.size(0) != token_count_i64.numel():
        raise ValueError(
            "Grouped route param must be 2D with the first dimension equal to token_count length, "
            f"but got shape {tuple(param.shape)} and token_count length {token_count_i64.numel()}."
        )

    expanded = param_fp.repeat_interleave(token_count_i64, dim=0)
    return expanded.reshape(*route_shape, param_fp.size(-1))


def _block_dynamic_quant(input_fp: torch.Tensor, quant_block_size: int):
    if input_fp.shape[-1] % quant_block_size != 0:
        raise ValueError(
            f"Last dim {input_fp.shape[-1]} must be divisible by quant_block_size {quant_block_size}."
        )
    input_blocks = input_fp.reshape(*input_fp.shape[:-1], -1, quant_block_size)
    scale = input_blocks.abs().amax(dim=-1).clamp(min=1e-12) / 127
    quantized = torch.clamp(torch.round(input_blocks / scale.unsqueeze(-1)), -128, 127)
    return quantized.reshape_as(input_fp).to(torch.int8), scale


def _sort_moe_routes(
    hidden_states: torch.Tensor,
    top_k_gates: torch.Tensor,
    top_k_indices: torch.Tensor,
):
    if hidden_states.dim() != 2:
        raise ValueError(f"hidden_states must be 2D, but got shape {tuple(hidden_states.shape)}")
    if top_k_gates.shape != top_k_indices.shape:
        raise ValueError(
            f"top_k_gates and top_k_indices must have the same shape, got "
            f"{tuple(top_k_gates.shape)} vs {tuple(top_k_indices.shape)}."
        )
    if top_k_indices.dim() != 2:
        raise ValueError(f"top_k_indices must be 2D, but got shape {tuple(top_k_indices.shape)}")

    token_num, top_k = top_k_indices.shape
    hidden_dim = hidden_states.shape[-1]

    flat_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    flat_gates = top_k_gates.reshape(-1, 1)
    flat_experts = top_k_indices.reshape(-1).to(dtype=torch.int64)
    flat_token_indices = (
        torch.arange(token_num, device=top_k_indices.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )

    _, sort_indices = flat_experts.sort(stable=True)
    sorted_experts = flat_experts.index_select(0, sort_indices)
    sorted_hidden = flat_hidden.index_select(0, sort_indices).reshape(token_num, top_k, hidden_dim)
    sorted_gates = flat_gates.index_select(0, sort_indices).reshape(token_num, top_k, 1)
    sorted_token_indices = flat_token_indices.index_select(0, sort_indices).reshape(token_num, top_k, 1)
    return sorted_hidden, sorted_gates, sorted_token_indices, sorted_experts.reshape(token_num, top_k)


class MojoMixLinear(MojoOperator):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}.")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}.")

        self.in_features = in_features
        self.out_features = out_features

        weight_factory_kwargs = dict(self.tensor_factory_kwargs)
        weight_factory_kwargs["dtype"] = torch.float32
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **weight_factory_kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() < 2:
            raise ValueError(f"input must have rank >= 2, got shape {tuple(input.shape)}.")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"input last dim must be in_features={self.in_features}, got {input.shape[-1]}."
            )
        return torch.matmul(input.float(), self.weight.t().float())

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def _prequant_scale_2d(scale: torch.Tensor, token_num: int) -> torch.Tensor:
    if scale.dim() == 1:
        scale_2d = scale.reshape(-1, 1)
    elif scale.dim() == 2 and scale.size(1) == 1:
        scale_2d = scale
    else:
        raise ValueError(f"scale must have shape [tokens] or [tokens, 1], got {tuple(scale.shape)}.")
    if scale_2d.size(0) != token_num:
        raise ValueError(f"scale first dim must match token_num={token_num}, got {scale_2d.size(0)}.")
    return scale_2d.float()


class MojoQuantMoEPreQuant(MojoOperator):
    def __init__(
        self,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size=None,
        activation: str = "swiglu",
        quant_dtype: torch.dtype = torch.int8,
        up_quant_group_size: int = -1,
        up_weight_dtype: Union[torch.dtype, str] = torch.int8,
        down_quant_group_size: int = -1,
        down_weight_dtype: Union[torch.dtype, str] = torch.int8,
        output_dtype: torch.dtype = torch.bfloat16,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group=None,
        dp_input: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if activation != "swiglu":
            raise NotImplementedError(f"MojoQuantMoEPreQuant: Activation {activation} is not supported.")
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"MojoQuantMoEPreQuant: quant_dtype must be 'int8', got {quant_dtype}.")
        if up_weight_dtype not in ("int4", torch.int8) or down_weight_dtype not in ("int4", torch.int8):
            raise ValueError("MojoQuantMoEPreQuant: weight must be w4 or w8")
        if intermediate_size is None:
            raise ValueError("MojoQuantMoEPreQuant: intermediate_size must be provided.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_dtype = quant_dtype
        self.up_quant_group_size = up_quant_group_size
        self.up_weight_dtype = up_weight_dtype
        self.down_quant_group_size = down_quant_group_size
        self.down_weight_dtype = down_weight_dtype
        self.output_dtype = output_dtype

        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.ep_group = ep_group
        base = num_experts // ep_size
        rem = num_experts % ep_size
        self.num_experts_local = base + 1 if ep_rank < rem else base
        self.ep_start = base * ep_rank + min(ep_rank, rem)
        self.ep_end = self.ep_start + self.num_experts_local
        self.dp_input = dp_input

        self.gating = MojoMoEGating._registry.get("torch")(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            **kwargs,
        )
        self.dispatch = MojoMoEDispatch._registry.get("torch")(num_experts=self.num_experts, **kwargs)
        self.experts = MojoQuantExperts._registry.get("torch")(
            num_experts=self.num_experts_local,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=activation,
            quant_dtype=quant_dtype,
            up_quant_group_size=up_quant_group_size,
            up_weight_dtype=up_weight_dtype,
            down_quant_group_size=down_quant_group_size,
            down_weight_dtype=down_weight_dtype,
            **kwargs,
        )
        self.combine = MojoMoECombine._registry.get("torch")(multiply_by_gates=True, **kwargs)

    def _prequant_experts(
        self,
        sorted_hidden_states: torch.Tensor,
        sorted_scale: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        tokens_per_expert_list = tokens_per_expert.to("cpu").tolist()
        x_int8_list = torch.split(sorted_hidden_states, tokens_per_expert_list, dim=0)
        x_scale_list = torch.split(sorted_scale, tokens_per_expert_list, dim=0)

        activated_outs = []
        for expert_idx, token_count in enumerate(tokens_per_expert_list):
            if token_count == 0:
                activated_outs.append(
                    torch.empty(
                        0,
                        self.intermediate_size,
                        device=sorted_hidden_states.device,
                        dtype=self.output_dtype,
                    )
                )
                continue

            fc1_out = self.experts._quant_linear(
                x_int8_list[expert_idx],
                x_scale_list[expert_idx],
                self.experts.up_proj_weight[expert_idx],
                self.experts.up_proj_weight_scale[expert_idx],
                self.output_dtype,
                self.up_weight_dtype,
                self.up_quant_group_size,
            )
            gate_proj, up_proj = fc1_out.float().chunk(2, dim=-1)
            activated_outs.append((F.silu(gate_proj) * up_proj).to(self.output_dtype))
        activated = torch.cat(activated_outs, dim=0)

        y_int8, y_scale = self.experts.down_proj_quantize(activated, tokens_per_expert)
        y_int8_list = torch.split(y_int8, tokens_per_expert_list, dim=0)
        y_scale_list = torch.split(y_scale, tokens_per_expert_list, dim=0)
        outputs = []
        for expert_idx, token_count in enumerate(tokens_per_expert_list):
            if token_count == 0:
                outputs.append(
                    torch.empty(
                        0,
                        self.hidden_size,
                        device=sorted_hidden_states.device,
                        dtype=self.output_dtype,
                    )
                )
                continue

            fc2_out = self.experts._quant_linear(
                y_int8_list[expert_idx],
                y_scale_list[expert_idx],
                self.experts.down_proj_weight[expert_idx],
                self.experts.down_proj_weight_scale[expert_idx],
                self.output_dtype,
                self.down_weight_dtype,
                self.down_quant_group_size,
            )
            outputs.append(fc2_out)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        quant_hidden_states: torch.Tensor,
        scale: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        if quant_hidden_states.dtype != torch.int8:
            raise TypeError(
                f"quant_hidden_states must be pre-quantized torch.int8, got {quant_hidden_states.dtype}."
            )
        if quant_hidden_states.dim() != 2:
            raise ValueError(f"quant_hidden_states must be 2D, got shape {tuple(quant_hidden_states.shape)}.")
        if quant_hidden_states.size(1) != self.hidden_size:
            raise ValueError(
                f"quant_hidden_states last dim must be hidden_size={self.hidden_size}, "
                f"got {quant_hidden_states.size(1)}."
            )
        if hidden_states.shape != quant_hidden_states.shape:
            raise ValueError(
                "hidden_states must have the same shape as quant_hidden_states, "
                f"got {tuple(hidden_states.shape)} vs {tuple(quant_hidden_states.shape)}."
            )

        scale_2d = _prequant_scale_2d(scale, quant_hidden_states.size(0)).to(device=quant_hidden_states.device)

        if self.dp_input and self.ep_size > 1:
            local_tokens = quant_hidden_states.shape[0]
            full_quant_hidden = torch.empty(
                local_tokens * self.ep_size,
                quant_hidden_states.shape[1],
                dtype=quant_hidden_states.dtype,
                device=quant_hidden_states.device,
            )
            full_scale = torch.empty(
                local_tokens * self.ep_size,
                scale_2d.shape[1],
                dtype=scale_2d.dtype,
                device=scale_2d.device,
            )
            full_gate_hidden = torch.empty(
                local_tokens * self.ep_size,
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            import torch.distributed as dist

            dist.all_gather_into_tensor(full_quant_hidden, quant_hidden_states.contiguous(), group=self.ep_group)
            dist.all_gather_into_tensor(full_scale, scale_2d.contiguous(), group=self.ep_group)
            dist.all_gather_into_tensor(full_gate_hidden, hidden_states.contiguous(), group=self.ep_group)
            quant_hidden_states = full_quant_hidden
            scale_2d = full_scale
            hidden_states = full_gate_hidden

        top_k_indices, top_k_gates = self.gating(hidden_states)
        sorted_hidden_states, tokens_per_expert, sorted_gates, token_indices = self.dispatch(
            quant_hidden_states,
            top_k_gates,
            top_k_indices,
        )
        sorted_scale = scale_2d.index_select(0, token_indices.to(dtype=torch.long))

        if self.ep_size > 1:
            cumsum = tokens_per_expert.cumsum(0)
            tok_start = 0 if self.ep_start == 0 else cumsum[self.ep_start - 1].item()
            tok_end = cumsum[self.ep_end - 1].item()
            sorted_hidden_states = sorted_hidden_states[tok_start:tok_end]
            sorted_scale = sorted_scale[tok_start:tok_end]
            tokens_per_expert = tokens_per_expert[self.ep_start:self.ep_end]
            sorted_gates = sorted_gates[tok_start:tok_end]
            token_indices = token_indices[tok_start:tok_end]

        expert_outputs = self._prequant_experts(sorted_hidden_states, sorted_scale, tokens_per_expert)
        output_buffer = torch.zeros(
            quant_hidden_states.size(0),
            self.hidden_size,
            dtype=self.output_dtype,
            device=quant_hidden_states.device,
        )
        combined = self.combine(output_buffer, expert_outputs, sorted_gates, token_indices)

        if self.ep_size > 1:
            import torch.distributed as dist

            if self.dp_input:
                local_combined = torch.empty(
                    combined.shape[0] // self.ep_size,
                    combined.shape[1],
                    dtype=combined.dtype,
                    device=combined.device,
                )
                dist.reduce_scatter_tensor(
                    local_combined,
                    combined.contiguous(),
                    op=dist.ReduceOp.SUM,
                    group=self.ep_group,
                )
                combined = local_combined
            else:
                dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=self.ep_group)

        return combined

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, top_k={self.top_k}, hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, output_dtype={self.output_dtype}, "
            f"up_weight_dtype={self.up_weight_dtype}, down_weight_dtype={self.down_weight_dtype}"
        )


class MojoMoEInitRoutingDynamicQuant(MojoOperator):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        quant_block_size: int = 8,
        quant_dtype: torch.dtype = torch.int8,
        start_expert_id: int = 0,
        end_expert_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant_block_size = quant_block_size
        self.quant_dtype = quant_dtype
        self.start_expert_id = start_expert_id
        self.end_expert_id = num_experts if end_expert_id is None else end_expert_id

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_gates: torch.Tensor,
        top_k_indices: torch.Tensor,
        smooth_scale: Optional[torch.Tensor] = None,
        quant_mode: int = 0,
    ):
        if quant_mode not in (0, 1):
            raise NotImplementedError(f"Unsupported quant_mode: {quant_mode}, expected 0 or 1.")

        sorted_hidden, sorted_gates, sorted_token_indices, sorted_experts = _sort_moe_routes(
            hidden_states,
            top_k_gates,
            top_k_indices,
        )

        route_hidden = sorted_hidden.float()
        if smooth_scale is not None:
            if smooth_scale.dim() != 2 or smooth_scale.size(0) != self.num_experts:
                raise ValueError(
                    "smooth_scale must be 2D with shape (num_experts, hidden_size), "
                    f"but got shape {tuple(smooth_scale.shape)} and num_experts={self.num_experts}."
                )
            route_scale = smooth_scale.index_select(0, sorted_experts.reshape(-1).to(dtype=torch.long))
            route_scale = route_scale.reshape_as(route_hidden)
            route_hidden = route_hidden * route_scale.float()

        quantized, scale = _block_dynamic_quant(route_hidden, self.quant_block_size)
        token_count = _count_expert_tokens(top_k_indices, self.num_experts)
        return (
            quantized.to(self.quant_dtype),
            sorted_gates.float(),
            sorted_token_indices.to(dtype=torch.int32),
            token_count,
            scale,
        )


class MojoFusedSwiGLUMoEScaleDynamicQuantize(MojoOperator):
    def __init__(
        self,
        quant_dtype: torch.dtype = torch.int8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if quant_dtype != torch.int8:
            raise NotImplementedError(f"Unsupported quant_dtype: {quant_dtype}, expected torch.int8.")
        self.quant_dtype = quant_dtype

    def forward(
        self,
        input: torch.Tensor,
        smooth_scale: Optional[torch.Tensor],
        token_count: torch.Tensor,
        beta: float = 1.0,
        quant_mode: int = 0,
    ):
        if input.dim() != 3:
            raise ValueError(f"input must be 3D, but got shape {tuple(input.shape)}")
        if input.shape[-1] % 2 != 0:
            raise ValueError(f"input last dim must be even for SwiGLU, but got {input.shape[-1]}")
        if beta == 0:
            raise ValueError("beta must be non-zero.")
        if quant_mode not in (0, 1):
            raise NotImplementedError(f"Unsupported quant_mode: {quant_mode}, expected 0 or 1.")

        route_shape = input.shape[:2]
        _validate_moe_token_count(token_count, route_shape[0] * route_shape[1])

        left, right = input.float().chunk(2, dim=-1)
        output = (F.silu(left * beta) / beta) * right

        expanded_scale = _expand_grouped_route_param(smooth_scale, token_count, route_shape)
        if expanded_scale is not None:
            output = output * expanded_scale

        scale = output.abs().amax(dim=-1).clamp(min=1e-12) / 127
        quantized = torch.clamp(torch.round(output / scale.unsqueeze(-1)), -128, 127)
        return quantized.to(self.quant_dtype), scale


__all__ = [
    "MojoMixLinear",
    "MojoQuantMoEPreQuant",
    "MojoMoEInitRoutingDynamicQuant",
    "MojoFusedSwiGLUMoEScaleDynamicQuantize",
]
