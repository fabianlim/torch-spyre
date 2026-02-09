# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor utility functions for Spyre device."""

import torch


def from_blob(
    data_ptr: int,
    size: list[int] | tuple[int, ...],
    dtype: torch.dtype,
    device_layout: "torch_spyre._C.SpyreTensorLayout",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Create a Spyre tensor from existing device memory without allocating.

    This function wraps existing device memory in a Spyre tensor without
    performing any new device memory allocation. The caller is responsible
    for ensuring the memory remains valid for the lifetime of the tensor.

    Args:
        data_ptr: Pointer to existing device memory (as integer).
        size: Shape of the tensor (e.g., [512, 1024]).
        dtype: PyTorch data type (e.g., torch.float16).
        device_layout: SpyreTensorLayout specifying the device memory layout.
        device: Device specification (default: "spyre:0").

    Returns:
        A Spyre tensor wrapping the existing device memory.

    Warning:
        The caller must ensure that:
        1. The device memory pointed to by data_ptr remains valid
        2. The memory is not freed while the tensor is in use
        3. The device_layout matches the actual memory layout

    Example:
        >>> # Assume we have device memory pointer from some operation
        >>> device_ptr = 0x12345678  # Example pointer
        >>> layout = torch_spyre._C.SpyreTensorLayout(
        ...     [512, 1024], torch.float16
        ... )
        >>> tensor = torch_spyre.from_blob(
        ...     device_ptr, [512, 1024], torch.float16, layout
        ... )
        >>> # Use tensor without allocating new device memory
    """
    import torch_spyre

    if device is None:
        device = torch.device("spyre:0")
    elif isinstance(device, str):
        device = torch.device(device)

    # Ensure runtime is initialized
    torch_spyre.spyre._lazy_init()

    # Call the C++ binding
    return torch_spyre._C.from_blob(
        data_ptr, size, device_layout, dtype, None, device, None, None
    )


def from_blob_with_strides(
    data_ptr: int,
    size: list[int] | tuple[int, ...],
    stride: list[int] | tuple[int, ...],
    dtype: torch.dtype,
    device_layout: "torch_spyre._C.SpyreTensorLayout",
) -> torch.Tensor:
    """Create a Spyre tensor from existing device memory with custom strides.

    Similar to from_blob but allows specifying custom strides for non-contiguous
    tensors.

    Args:
        data_ptr: Pointer to existing device memory (as integer).
        size: Shape of the tensor.
        stride: Stride for each dimension.
        dtype: PyTorch data type.
        device_layout: SpyreTensorLayout specifying the device memory layout.

    Returns:
        A Spyre tensor wrapping the existing device memory with specified strides.

    Example:
        >>> device_ptr = 0x12345678
        >>> layout = torch_spyre._C.SpyreTensorLayout([512, 1024], torch.float16)
        >>> tensor = torch_spyre.from_blob_with_strides(
        ...     device_ptr, [512, 1024], [1024, 1], torch.float16, layout
        ... )
    """
    import torch_spyre

    # Ensure runtime is initialized
    torch_spyre.spyre._lazy_init()

    # Call the C++ binding directly with strides
    return torch_spyre._C.spyre_from_blob(
        data_ptr, size, stride, dtype, device_layout
    )
