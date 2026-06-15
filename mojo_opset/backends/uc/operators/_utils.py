import torch
from functools import lru_cache


_DTYPE_API_SUFFIX = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


@lru_cache(maxsize=1)
def _uc_kernels():
    import uc_kernel

    return uc_kernel.load()


def _matrix_shape(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.dim() == 0:
        return 1, 1
    if tensor.dim() == 1:
        return 1, tensor.numel()
    return tensor.numel() // tensor.shape[-1], tensor.shape[-1]


def _typed_api(api: str, dtype: torch.dtype) -> str:
    suffix = _DTYPE_API_SUFFIX.get(dtype)
    if suffix is None:
        raise NotImplementedError(f"UC backend {api} does not support dtype {dtype}.")

    kernels = _uc_kernels()
    typed_api = f"{api}_{suffix}"
    if typed_api in kernels.keys():
        return typed_api
    raise NotImplementedError(f"UC backend {api} does not provide a {suffix} kernel artifact.")


def _bind_workspace_once(kernel_workspace_size: int) -> None:
    from uc_kernel.runtime import init_workspace

    init_workspace(required_nbytes=kernel_workspace_size)


_KERNEL_CACHE: dict = {}


def _get_kernel(api: str, dtype: torch.dtype):
    key = (api, dtype)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is not None:
        return kernel
    typed = _typed_api(api, dtype)
    kernel = _uc_kernels()[typed]
    _bind_workspace_once(kernel.workspace_size)
    _KERNEL_CACHE[key] = kernel
    return kernel


def run_unary_kernel(api: str, x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.empty_like(x)
    if not x.is_contiguous():
        raise NotImplementedError(f"UC backend {api} requires contiguous input.")

    kernel_output = torch.empty_like(x)
    rows, cols = _matrix_shape(x)
    kernel = _get_kernel(api, x.dtype)
    kernel(x, kernel_output, rows, cols)
    return kernel_output.reshape(x.shape)


def run_binary_kernel(api: str, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError(f"UC backend {api} expects matching input shapes, got {lhs.shape} and {rhs.shape}.")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"UC backend {api} expects matching input dtypes, got {lhs.dtype} and {rhs.dtype}.")
    if lhs.numel() == 0:
        return torch.empty_like(lhs)
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        raise NotImplementedError(f"UC backend {api} requires contiguous inputs.")

    kernel_output = torch.empty_like(lhs)
    rows, cols = _matrix_shape(lhs)
    kernel = _get_kernel(api, lhs.dtype)
    kernel(lhs, rhs, kernel_output, rows, cols)
    return kernel_output.reshape(lhs.shape)


def run_kernel(api: str, dtype: torch.dtype, *args) -> None:
    kernel = _get_kernel(api, dtype)
    kernel(*args)
