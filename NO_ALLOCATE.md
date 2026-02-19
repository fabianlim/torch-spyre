# Creating Spyre Tensors Without Allocating Device Memory

## Goal

Add a `from_blob` API that wraps an existing device memory pointer in a Spyre tensor, without performing a new device allocation. This is analogous to `torch.from_blob()` in upstream PyTorch.

## Changes Made

### 1. `SpyreStorageImpl` - new constructor (`spyre_storage_impl.h/.cpp`)

Added a second constructor that accepts an `at::DataPtr` directly instead of an `at::Allocator*`. This allows creating storage backed by externally-owned memory:

```cpp
SpyreStorageImpl(use_byte_size_t, c10::SymInt size_bytes, at::DataPtr data_ptr);
```

The parent `c10::StorageImpl` is constructed with `allocator=nullptr` and `resizable=false` since we don't own the memory.

### 2. C++ functions (`spyre_mem.h/.cpp`)

Two new functions:

- **`spyre_from_blob(data_ptr, size, stride, dtype, device_layout)`** - Core implementation. Takes a raw `void*`, wraps it in an `at::DataPtr` with a no-op deleter (since the caller owns the memory), creates a `SpyreStorageImpl` from that, and builds a `SpyreTensorImpl` on top.

- **`from_blob(data_ptr, size, device_layout, dtype_opt, ...)`** - Convenience wrapper matching the `empty_with_layout` signature style. Computes contiguous strides automatically and delegates to `spyre_from_blob`.

Key design decisions:
- The `at::DataPtr` uses a **no-op deleter** (`[](void*) {}`) so the memory is never freed by the tensor
- The `ctx` pointer in `DataPtr` is `nullptr` (no `SharedOwnerCtx`) since we don't have a `DeviceMemoryAllocationPtr`
- **Consequence**: Operations that call `storage().data_ptr().get_context()` and cast to `SharedOwnerCtx*` (e.g., `copy_host_to_device`, `copy_device_to_host`, `launchKernel` in module.cpp) will get a null context. This needs to be handled or documented as a limitation.

### 3. Pybind11 bindings (`module.cpp`)

Both functions exposed:
```cpp
m.def("spyre_from_blob", &spyre::spyre_from_blob);
m.def("from_blob", &spyre::from_blob);
```

### 4. Python wrapper (`tensor_utils.py`)

New file with two functions:
- `from_blob(data_ptr, size, dtype, device_layout, device=None)` - Main user-facing API
- `from_blob_with_strides(data_ptr, size, stride, dtype, device_layout)` - Variant with explicit strides

Not yet wired into `__init__.py` exports (was interrupted).

## Remaining Work

### Must Do

1. **Wire up Python exports** - Add `from_blob` / `from_blob_with_strides` to `torch_spyre/__init__.py` or make them accessible via `torch.spyre.from_blob(...)`.

2. **Handle null `SharedOwnerCtx`** - Several code paths cast `storage().data_ptr().get_context()` to `SharedOwnerCtx*`:
   - `spyre_mem.cpp:415` (`copy_host_to_device`)
   - `spyre_mem.cpp:429` (`copy_device_to_host`)
   - `module.cpp:166,173,181` (`launchKernel`)

   These will segfault if called on a from_blob tensor. Options:
   - Add null-checks before dereferencing
   - Document that from_blob tensors are read-only / cannot be used with H2D/D2H copy
   - Store a sentinel context that signals "externally owned"

3. **Build and test** - `python setup.py build_ext --inplace` to compile, then verify.

4. **Add tests** - Test in `tests/` that:
   - A from_blob tensor has the correct shape, dtype, device, and layout
   - The from_blob tensor shares the same `data_ptr` as the source (no new allocation)
   - Deleting the tensor does NOT free the underlying memory

### Nice to Have

- Support passing a Python callback as a custom deleter (for prevent-leak safety)
- `torch.spyre.from_blob()` style API via monkey-patch (like `empty_with_layout`)
- Integration with the inductor compiled path (output tensors reusing existing buffers)

## How It Works (Summary)

```
caller provides: void* data_ptr  (existing device memory)
        |
        v
at::DataPtr(data_ptr, nullptr, no_op_deleter, device)   <-- wraps ptr, no ownership
        |
        v
SpyreStorageImpl(use_byte_size_t, size_bytes, DataPtr)   <-- storage without allocator
        |
        v
SpyreTensorImpl(storage, pu1_dks, dtype)                 <-- tensor with spyre_layout
        |
        v
set sizes/strides + spyre_layout                         <-- metadata
        |
        v
return tensor                                            <-- ready to use
```

Compared to `spyre_empty_with_layout` which does:
```
SpyreAllocator::allocate(size_bytes)  -->  new DeviceMemoryAllocation
        |
        v
SharedOwnerCtx{allocation, device_id}
        |
        v
at::DataPtr(raw_ptr, ctx, ReportAndDelete, device)       <-- owns memory, will free
        |
        v
SpyreStorageImpl(use_byte_size_t, size_bytes, &SpyreAllocator, true)
        ...
```

The key difference is skipping `SpyreAllocator::allocate()` and using a no-op deleter instead of `ReportAndDelete`.
