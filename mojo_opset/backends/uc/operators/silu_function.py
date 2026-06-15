import torch

from mojo_opset.core import MojoSiluFunction

from ._utils import run_unary_kernel


class UCSiluFunction(MojoSiluFunction):
    """UC backend autograd wrapper for ``MojoSiluFunction``.

    Forward
    -------
    Reuses the existing ``mojo_silu`` UC wheel kernel (single-pass UB vector
    op).  Equivalent to the eager ``input * sigmoid(input)`` reference.

    Backward raises until uc-kernel ships a dedicated SiLU backward API.
    """

    supported_platforms_list = ["npu"]

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return run_unary_kernel("mojo_silu", input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError(
            "UCSiluFunction.backward is not implemented by the current uc-kernel wheel."
        )
