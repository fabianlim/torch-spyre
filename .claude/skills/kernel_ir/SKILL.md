---
name: kernel-ir
description: >
  Use when generating, reviewing, or reasoning about KTDP MLIR dialect code,
  tiled tensor memory layouts, host-device mappings, SpyreTensorLayout,
  FixedTiledLayout, access tiles, memory views, or data-parallel kernels
  targeting the Spyre accelerator. Also use when working with Arith, Math,
  LinAlg, or SCF MLIR dialect operations in the context of Spyre compilation.
---

# Kernel IR Skill

This skill provides expertise on the intermediate representations used in torch-spyre's compilation pipeline, specifically **Tiled Tensor Layouts** (RFC 0047) and the **KTDP IR** (RFC 0682). Use this skill when generating, reviewing, or reasoning about kernel IR code for the Spyre accelerator.

## When to Use

- Generating or reviewing `ktdp` MLIR dialect code
- Reasoning about tiled tensor memory layouts and host↔device mappings
- Understanding how tensors are distributed, accessed, and computed on Spyre's multi-core architecture
- Designing data-parallel kernels targeting Spyre
- Working with `SpyreTensorLayout`, `FixedTiledLayout`, access tiles, or memory views

---

## Tiled Tensor Layouts (RFC 0047)

### Why Tiling Matters

Spyre is a SIMD dataflow accelerator built around systolic arrays. Its memory subsystem requires bulk loads from **contiguous** addresses for performance. Standard row-major layouts produce non-contiguous access patterns under tiling — so tensors must be **physically tiled in device memory**.

The fundamental unit is a **stick**: 128 bytes of contiguous memory. For float16, one stick = 64 elements.

### Host vs Device Layout

A tensor has two layouts:

| Aspect | Host Layout | Device Layout |
|--------|-------------|---------------|
| Representation | PyTorch `size()` + `strides()` | `device_size` + `dim_map` (row-major) |
| Rank | N | N+k (k = number of tiling splits) |
| Strides | Explicit | Implicit row-major |
| Padding | Typically none | Stick-aligned (128-byte) |

**Concrete example**: A 2D float16 tensor `(1024, 256)`:
- Host: size `(1024, 256)`, strides `(256, 1)` — standard row-major
- Device: size `(4, 1024, 64)`, strides `(65536, 64, 1)` — 3D tiled layout
- Each row of 256 elements → 4 sticks of 64 elements
- Sticks within the same tile are contiguous in device memory

### The Tiling Mapping

The host↔device mapping is encoded as 3 tuples of N+k integers: **(ranges, device_strides, host_strides)**, ordered by decreasing device stride. This defines a loop nest:

```
for i in range(4):       # iterate over stick groups
  for j in range(1024):  # iterate over rows
    for k in range(64):  # iterate within stick
      device[i*65536 + j*64 + k] = host[j*256 + i*64 + k]
```

### SpyreTensorLayout

The runtime representation consists of:
- **`device_size`**: padded, tiled shape (higher rank than PyTorch shape)
- **`dim_map`**: maps each device dimension back to a PyTorch dimension (-1 for synthetic dimensions)
- **`device_dtype`**: on-device data format

Rules:
- Layouts are always **row-major** (strides are implicit, always decreasing)
- The **stick dimension** is always the innermost (last) device dimension, with size = max elements per stick for the dtype (e.g., 64 for float16)
- Repeated entries in `dim_map` encode tiling: coordinates combine right-to-left
- `-1` in `dim_map` = synthetic dimension (e.g., for sparse tensors with one element per stick)
- PyTorch dimensions of size 1 are eliminated before computing the Spyre layout (canonical form)

**Default layout algorithm**: (a) last PyTorch dimension → stick dimension, (b) tile along first dimension, (c) pad stick dimension to multiple of stick size.

Example: `torch.rand(5, 100, 150, dtype=torch.float16).to("spyre")` →
```
device_size=[100, 3, 5, 64], dim_map=[1, 2, 0, 2]
```
Dimension 2 (size 150) is padded to 192, split into 3×64. Dimension 1 (size 100) becomes outermost.

### Padded Tensors

When a dimension doesn't evenly divide into sticks, the last stick is padded. E.g., float16 tensor `(1000, 200)` → device shape `(4, 1000, 64)` where the 4th stick of each row only holds 8 real elements + 112 bytes of padding.

### Sparse Tensors

Reduction operations along the stick dimension produce tensors with one element per stick. A synthetic inner dimension (size = elements per stick) is added to the Spyre layout with `dim_map = -1`.

### Compiler Integration

- `FixedTiledLayout` extends Inductor's `FixedLayout` with a `device_layout` field containing a `DeviceTensorLayout`
- Layout propagation happens during a topological traversal of `SchedulerNodes` via the `_pre_fusion_custom_pass` extension point
- The device layout information feeds into: memory planning (accurate device sizes), code generation (kernel loop nests), and cross-core work division

---

## KTDP IR (RFC 0682)

### Overview

**KTDP** (Kernel Tile Data Parallel) is a mid-level MLIR-based IR that serves as the interface between compiler frontends (TorchInductor, Triton/Helion) and the Spyre backend compiler (DeepTools). It replaces the previous SuperDSC-bundle format.

KTDP is designed around a **multi-core accelerator abstraction**:
- Multiple **cores**, each with a **compute engine** and **on-chip scratchpad** (LX)
- Cores connected via an **on-chip interconnect**
- Shared **off-chip HBM** (high-bandwidth memory)

### Design Philosophy

KTDP separates three concerns into composable abstractions:

1. **Memory interpretation** — `construct_memory_view`: how raw memory is viewed as a tensor
2. **Address computation** — `construct_access_tile`: which coordinates will be accessed
3. **Data movement** — `load` / `store`: actual memory reads and writes
4. **Compute** — delegated to standard MLIR dialects (Arith, Math, LinAlg)
5. **Control flow** — delegated to MLIR's SCF dialect (loops, conditionals)

Work partitioning is decided **before** KTDP — the IR assumes decomposition is already fixed and expresses how each tile executes its portion.

### KTDP Dialect Operations

#### `ktdp.get_compute_tile_id`
Returns the ID of the current compute tile (core). Used to partition work.
```mlir
%id = ktdp.get_compute_tile_id : index
```

#### `ktdp.construct_memory_view`
Creates a logical memref view over pre-allocated memory at a given address. Does **not** allocate memory.
```mlir
%view = ktdp.construct_memory_view %address, sizes: [96, 64], strides: [64, 1] {
    coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
    memory_space = #ktdp.spyre_memory_space<HBM>
} : memref<96x64xf16>
```
- `coordinate_set`: subset of logical coordinates this view covers (for distributed views)
- `memory_space`: physical location — `HBM` (shared) or `LX` (core-local scratchpad)
- Sizes and strides can be static or dynamic (SSA values)

#### `ktdp.construct_distributed_memory_view`
Composes multiple per-partition memory views into a single logical view. Global coordinate domain = union of constituent coordinate sets. No data movement.
```mlir
%dview = ktdp.construct_distributed_memory_view (%A0, %A1 : memref<32x64xf16>, memref<32x64xf16>)
    : memref<64x64xf16>
```

#### `ktdp.construct_access_tile`
Creates a structured set of coordinates (an access tile) from a memory view. Does **not** read or write memory. The tile is anchored at base indices and bounded by an `access_tile_set` (IntegerSet — can represent rectangular, strided, triangular, or polyhedral regions).
```mlir
%tile = ktdp.construct_access_tile %view[%row, %col] {
    access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
} : memref<96x64xf16> -> !ktdp.tile<3x64xindex>
```

#### `ktdp.construct_indirect_access_tile`
For gather/scatter-style indexing. Each base dimension can be **direct** (subscript used as-is) or **indirect** (subscript indexes into an auxiliary memory view that supplies the actual coordinate).

```mlir
// Y[m,k] = X[IDX1[m,k], IDX2[m,k]]
%tile = ktdp.construct_indirect_access_tile
    intermediate_variables(%m, %k)
    %X[%IDX1[%m, %k], %IDX2[%m, %k]] {
        variables_space_set = {(d0, d1) | 0 <= d0 < 2, 0 <= d1 < 64}
    } : memref<64x64xf16> -> !ktdp.tile<2x64xindex>
```

Key concepts:
- **`intermediate_variables`**: loop-like variables local to the operation, defining the iteration domain
- **`common_variables`**: SSA values from outside the operation, usable in subscript expressions
- **`variables_space_set`**: IntegerSet constraining the domain of intermediate variables
- Subscripts are affine/quasi-affine expressions (division, modulo allowed via MLIR)
- Paged attention example: `X[Idx[b][tkv/64], hkv, tkv % 64, dkv]` — dimension 0 is indirect (paged), dimensions 1-3 are direct

#### `ktdp.load`
Reads data from memory using an access tile's coordinates. Produces a tensor.
```mlir
%data = ktdp.load %access_tile : !ktdp.tile<3x64xindex> -> tensor<3x64xf16>
```

#### `ktdp.store`
Writes a data tile to memory at the coordinates specified by an access tile.
```mlir
ktdp.store %data, %access_tile : tensor<3x64xf16>, !ktdp.tile<3x64xindex>
```

### KTDP Types

#### `!ktdp.tile<...xindex>`
The AccessTileType — a ranked tile of index coordinates. Shape matches the coordinate collection. Only `index` element type is allowed. Dimensions can be static or dynamic (`?`).
```mlir
!ktdp.tile<3x64xindex>     // static shape
!ktdp.tile<?x64xindex>     // partially dynamic
!ktdp.tile<?x?xindex>      // fully dynamic
```

### KTDP Attributes

#### `#ktdp.spyre_memory_space<...>`
Identifies the physical memory location:
- `#ktdp.spyre_memory_space<HBM>` — global high-bandwidth memory (shared across cores)
- `#ktdp.spyre_memory_space<LX, core = 7>` — core-local scratchpad (core ID required)

### External MLIR Dialects Used in KTDP

KTDP delegates compute, control flow, and standard type operations to established MLIR dialects. The sections below provide a working reference for operations likely to appear in KTDP programs.

---

### Arith Dialect — Scalar & Index Arithmetic

Used in KTDP for: constants, address arithmetic, index computation, tile ID math, stride calculations.

#### Constants
```mlir
%c0 = arith.constant 0 : index
%c3 = arith.constant 3 : index
%addr = arith.constant 1024 : index
%f = arith.constant 1.0 : f16
```

#### Integer Arithmetic
| Operation | Description | Example |
|-----------|-------------|---------|
| `arith.addi` | N-bit integer addition | `%r = arith.addi %a, %b : index` |
| `arith.subi` | N-bit integer subtraction | `%r = arith.subi %a, %b : index` |
| `arith.muli` | N-bit integer multiplication | `%row = arith.muli %id, %tile_size : index` |
| `arith.divsi` | Signed division (toward zero) | `%r = arith.divsi %a, %b : index` |
| `arith.divui` | Unsigned division (toward zero) | `%pages = arith.divui %Ntkv, %Ptkv : index` |
| `arith.ceildivsi` | Signed ceiling division | `%r = arith.ceildivsi %a, %b : index` |
| `arith.ceildivui` | Unsigned ceiling division | `%r = arith.ceildivui %a, %b : index` |
| `arith.floordivsi` | Signed floor division | `%r = arith.floordivsi %a, %b : index` |
| `arith.remsi` | Signed remainder (sign of dividend) | `%r = arith.remsi %a, %b : index` |
| `arith.remui` | Unsigned remainder | `%r = arith.remui %a, %b : index` |

Optional overflow flags: `arith.addi %a, %b overflow<nsw, nuw> : i32`

#### Floating-Point Arithmetic
| Operation | Description |
|-----------|-------------|
| `arith.addf` | FP addition |
| `arith.subf` | FP subtraction |
| `arith.mulf` | FP multiplication |
| `arith.divf` | FP division |
| `arith.remf` | FP remainder |
| `arith.negf` | FP negation |

All accept optional `fastmath<...>` flags.

#### Comparisons
```mlir
// Integer: eq, ne, slt, sle, sgt, sge, ult, ule, ugt, uge
%cond = arith.cmpi slt, %a, %b : index

// Float: oeq, one, olt, ole, ogt, oge, ueq, une, ult, ule, ugt, uge, ord, uno
%cond = arith.cmpf olt, %x, %y : f16
```

#### Min / Max
| Operation | Description |
|-----------|-------------|
| `arith.maxsi` / `arith.minsi` | Signed integer max/min |
| `arith.maxui` / `arith.minui` | Unsigned integer max/min |
| `arith.maximumf` / `arith.minimumf` | FP max/min (NaN-propagating, -0.0 < +0.0) |
| `arith.maxnumf` / `arith.minnumf` | FP max/min (prefer numeric over NaN) |

#### Casting
| Operation | Description |
|-----------|-------------|
| `arith.extsi` / `arith.extui` | Sign-extend / zero-extend integer |
| `arith.trunci` | Truncate integer to narrower type |
| `arith.extf` / `arith.truncf` | Extend / narrow floating-point |
| `arith.sitofp` / `arith.uitofp` | Signed/unsigned int → float |
| `arith.fptosi` / `arith.fptoui` | Float → signed/unsigned int |
| `arith.bitcast` | Reinterpret bits without conversion |
| `arith.index_cast` | Convert between `index` and integer types |

#### Bitwise
| Operation | Description |
|-----------|-------------|
| `arith.andi` / `arith.ori` / `arith.xori` | AND / OR / XOR |
| `arith.shli` | Left shift |
| `arith.shrsi` / `arith.shrui` | Arithmetic / logical right shift |

#### Select
```mlir
%result = arith.select %cond, %true_val, %false_val : f16
```

---

### Math Dialect — Transcendental & Special Functions

Used in KTDP for: activation functions, numerical operations within compute tiles. All ops accept optional `fastmath<...>` flags.

#### Exponential & Logarithmic
| Operation | Computes |
|-----------|----------|
| `math.exp %x` | e^x |
| `math.exp2 %x` | 2^x |
| `math.expm1 %x` | e^x - 1 (accurate near 0) |
| `math.log %x` | ln(x) |
| `math.log2 %x` | log₂(x) |
| `math.log10 %x` | log₁₀(x) |
| `math.log1p %x` | ln(1 + x) (accurate near 0) |

#### Power & Root
| Operation | Computes |
|-----------|----------|
| `math.powf %base, %exp` | base^exp (float) |
| `math.fpowi %base, %exp` | base^exp (float base, int exp) |
| `math.sqrt %x` | √x |
| `math.rsqrt %x` | 1/√x |
| `math.cbrt %x` | ∛x |

#### Trigonometric
| Operation | Computes |
|-----------|----------|
| `math.sin` / `math.cos` / `math.tan` | sin/cos/tan |
| `math.asin` / `math.acos` / `math.atan` | Inverse trig |
| `math.atan2 %y, %x` | atan2(y, x) |

#### Hyperbolic
| Operation | Computes |
|-----------|----------|
| `math.tanh` | tanh(x) — commonly used for activation functions |
| `math.sinh` / `math.cosh` | Hyperbolic sin/cos |
| `math.asinh` / `math.acosh` / `math.atanh` | Inverse hyperbolic |

#### Rounding
| Operation | Semantics |
|-----------|-----------|
| `math.ceil` | Round up |
| `math.floor` | Round down |
| `math.round` | Round half-away-from-zero |
| `math.roundeven` | Round half-to-even (banker's) |
| `math.trunc` | Truncate toward zero |

#### Special Functions
| Operation | Computes |
|-----------|----------|
| `math.erf %x` | Error function |
| `math.erfc %x` | Complementary error function (1 - erf) |
| `math.fma %a, %b, %c` | Fused multiply-add (a*b + c, single rounding) |
| `math.absf` / `math.absi` | Absolute value (float / int) |
| `math.copysign %mag, %sign` | Magnitude of first, sign of second |
| `math.clampf %val to [%min, %max]` | Clamp to range |

#### Classification (return i1)
`math.isnan`, `math.isinf`, `math.isfinite`, `math.isnormal`

#### Bit Counting (integer)
`math.ctlz` (leading zeros), `math.cttz` (trailing zeros), `math.ctpop` (popcount)

---

### LinAlg Dialect — Tensor Compute Operations

Used in KTDP for: the actual compute on data tiles after `ktdp.load`. LinAlg ops use the **`ins` / `outs` convention**: `ins(...)` are read-only inputs, `outs(...)` are output/accumulator tensors. Operations support both tensor and memref operands, include optional numeric casting/promotion, and can be tiled and fused by the compiler.

#### Core Pattern in KTDP
```mlir
// Load data tiles
%A = ktdp.load %A_tile : !ktdp.tile<3x64xindex> -> tensor<3x64xf16>
%B = ktdp.load %B_tile : !ktdp.tile<3x64xindex> -> tensor<3x64xf16>

// Allocate output
%C = tensor.empty() : tensor<3x64xf16>

// Compute
linalg.add ins(%A, %B : tensor<3x64xf16>, tensor<3x64xf16>)
           outs(%C : tensor<3x64xf16>) -> tensor<3x64xf16>
```

#### Elementwise Unary Operations
| Operation | Computes | Syntax |
|-----------|----------|--------|
| `linalg.abs` | \|x\| | `linalg.abs ins(%in) outs(%out)` |
| `linalg.ceil` | ⌈x⌉ | `linalg.ceil ins(%in) outs(%out)` |
| `linalg.floor` | ⌊x⌋ | `linalg.floor ins(%in) outs(%out)` |
| `linalg.exp` | e^x | `linalg.exp ins(%in) outs(%out)` |
| `linalg.log` | ln(x) | `linalg.log ins(%in) outs(%out)` |
| `linalg.negf` | -x | `linalg.negf ins(%in) outs(%out)` |
| `linalg.reciprocal` | 1/x | `linalg.reciprocal ins(%in) outs(%out)` |
| `linalg.round` | round(x) | `linalg.round ins(%in) outs(%out)` |
| `linalg.sqrt` | √x | `linalg.sqrt ins(%in) outs(%out)` |
| `linalg.rsqrt` | 1/√x | `linalg.rsqrt ins(%in) outs(%out)` |
| `linalg.square` | x² | `linalg.square ins(%in) outs(%out)` |
| `linalg.tanh` | tanh(x) | `linalg.tanh ins(%in) outs(%out)` |
| `linalg.erf` | erf(x) | `linalg.erf ins(%in) outs(%out)` |
| `linalg.copy` | copy (with optional cast) | `linalg.copy ins(%in) outs(%out)` |

#### Elementwise Binary Operations
| Operation | Computes | Syntax |
|-----------|----------|--------|
| `linalg.add` | a + b | `linalg.add ins(%a, %b) outs(%c)` |
| `linalg.sub` | a - b | `linalg.sub ins(%a, %b) outs(%c)` |
| `linalg.mul` | a * b | `linalg.mul ins(%a, %b) outs(%c)` |
| `linalg.div` | a / b | `linalg.div ins(%a, %b) outs(%c)` |
| `linalg.div_unsigned` | a / b (unsigned) | `linalg.div_unsigned ins(%a, %b) outs(%c)` |
| `linalg.powf` | a^b | `linalg.powf ins(%a, %b) outs(%c)` |
| `linalg.max` | max(a, b) | `linalg.max ins(%a, %b) outs(%c)` |
| `linalg.min` | min(a, b) | `linalg.min ins(%a, %b) outs(%c)` |

#### Elementwise Ternary
```mlir
// Conditional select
linalg.select ins(%cond, %true, %false : tensor<...xi1>, tensor<...xf16>, tensor<...xf16>)
              outs(%out : tensor<...xf16>) -> tensor<...xf16>
```

#### Generalized Elementwise
```mlir
// Unary, binary, or ternary via kind attribute
linalg.elementwise kind=#linalg.elementwise_kind<exp>
    ins(%in : tensor<...xf16>) outs(%out : tensor<...xf16>) -> tensor<...xf16>
```

#### Contraction / Linear Algebra Operations
```mlir
// Matrix multiply: C[m,n] += A[m,k] * B[k,n]
linalg.matmul ins(%A, %B : tensor<MxKxf16>, tensor<KxNxf16>)
              outs(%C : tensor<MxNxf16>) -> tensor<MxNxf16>

// Batched matmul: C[b,m,n] += A[b,m,k] * B[b,k,n]
linalg.batch_matmul ins(%A, %B : tensor<BxMxKxf16>, tensor<BxKxNxf16>)
                    outs(%C : tensor<BxMxNxf16>) -> tensor<BxMxNxf16>

// Dot product: c += a[k] * b[k]
linalg.dot ins(%a, %b : tensor<Kxf16>, tensor<Kxf16>)
           outs(%c : tensor<f16>) -> tensor<f16>

// Matrix-vector: y[m] += A[m,k] * x[k]
linalg.matvec ins(%A, %x : tensor<MxKxf16>, tensor<Kxf16>)
              outs(%y : tensor<Mxf16>) -> tensor<Mxf16>

// Vector-matrix: y[n] += x[k] * A[k,n]
linalg.vecmat ins(%x, %A : tensor<Kxf16>, tensor<KxNxf16>)
              outs(%y : tensor<Nxf16>) -> tensor<Nxf16>
```

Custom indexing maps allow transposed or broadcast variants:
```mlir
linalg.batch_matmul
    indexing_maps = [affine_map<(b,m,n,k) -> (b,k,m)>,   // A transposed
                     affine_map<(b,m,n,k) -> (b,k,n)>,
                     affine_map<(b,m,n,k) -> (b,m,n)>]
    ins(%A, %B : ...) outs(%C : ...) -> ...
```

Contraction ops perform **numeric casting on inner multiply operands**, promoting them to the accumulator/output dtype.

#### Data Manipulation
```mlir
// Fill tensor with scalar
linalg.fill ins(%scalar : f16) outs(%tensor : tensor<3x64xf16>) -> tensor<3x64xf16>

// Transpose dimensions
linalg.transpose ins(%in : tensor<3x64xf16>) outs(%out : tensor<64x3xf16>)
    permutation = [1, 0]

// Broadcast along specified dimensions
linalg.broadcast ins(%in : tensor<64xf16>) outs(%out : tensor<3x64xf16>)
    dimensions = [0]
```

#### Reduction & Softmax
```mlir
// Reduce along dimensions
linalg.reduce ins(%in : tensor<3x64xf16>) outs(%out : tensor<3xf16>)
    dimensions = [1]
    (%a: f16, %b: f16) {
        %sum = arith.addf %a, %b : f16
        linalg.yield %sum : f16
    }

// Softmax normalization along a dimension
linalg.softmax dimension(1)
    ins(%in : tensor<3x64xf16>) outs(%out : tensor<3x64xf16>) -> tensor<3x64xf16>
```

#### Generic Operation (fully custom compute)
```mlir
linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
} ins(%in : tensor<3x64xf16>) outs(%out : tensor<3x64xf16>) {
    ^bb0(%a: f16, %b: f16):
        %r = arith.mulf %a, %a : f16   // e.g. square
        linalg.yield %r : f16
} -> tensor<3x64xf16>
```

- `indexing_maps`: affine maps from iteration space to operand indices
- `iterator_types`: `"parallel"` (independent) or `"reduction"` (accumulated)
- Body region receives one scalar per operand, must terminate with `linalg.yield`

#### Convolution Operations (selected)
Named variants encode data layout: `conv_{rank}d_{input_layout}_{kernel_layout}`
- 1D: `conv_1d_ncw_fcw`, `conv_1d_nwc_wcf`
- 2D: `conv_2d_nchw_fchw`, `conv_2d_nhwc_hwcf`, `conv_2d_nhwc_fhwc`
- 3D: `conv_3d_ncdhw_fcdhw`, `conv_3d_ndhwc_dhwcf`
- Depthwise: `depthwise_conv_2d_nhwc_hwc`, `depthwise_conv_2d_nhwc_hwcm`
- Quantized variants: suffix `_q`, add zero-point inputs
- All accept `strides` and `dilations` attributes

#### Pooling Operations (selected)
Named as `pooling_{layout}_{reduction}`: `pooling_nhwc_max`, `pooling_nhwc_sum`, `pooling_nchw_max`, `pooling_ndhwc_min`, etc.

---

### SCF Dialect — Structured Control Flow

Used in KTDP for: iterating over tile rows, conditional execution, loop-carried accumulators.

#### `scf.for` — Counted Loop
```mlir
// Simple loop: iterate i from 0 to tile_size, step 1
scf.for %i = %c0 to %tile_size step %c1 {
    // ... construct access tiles, load, compute, store per row ...
}
```

With loop-carried variables (accumulators):
```mlir
%result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> tensor<64xf16> {
    // %acc is the current accumulator value
    %new_acc = ...  // compute updated value
    scf.yield %new_acc : tensor<64xf16>
}
// %result holds the final accumulated value
```

- Induction variable type: `index` or signless integer
- Range is half-open: `[lb, ub)` with step
- Body must terminate with `scf.yield`
- `iter_args` threads values across iterations; yielded values become next iteration's args

#### `scf.while` — General While Loop
```mlir
%result = scf.while (%arg = %init) : (f16) -> f16 {
    // "before" region: evaluate condition
    %cond = arith.cmpf olt, %arg, %threshold : f16
    scf.condition(%cond) %arg : f16
} do {
^bb0(%val: f16):
    // "after" region: loop body
    %next = arith.addf %val, %step : f16
    scf.yield %next : f16
}
```

- Two regions: "before" (condition check) and "after" (body)
- `scf.condition(%bool)` terminates "before": true → execute body, false → exit
- Value flow: init → before args → condition forwards to after → yield back to before

#### `scf.if` — Conditional
```mlir
%result = scf.if %cond -> tensor<3x64xf16> {
    // then branch
    scf.yield %val_a : tensor<3x64xf16>
} else {
    // else branch
    scf.yield %val_b : tensor<3x64xf16>
}
```

- `else` is optional (required if `scf.if` produces results)
- Both branches must yield matching types

#### `scf.forall` — Multi-Dimensional Parallel Region
```mlir
scf.forall (%i, %j) = (0, 0) to (%dim0, %dim1) step (1, 1)
    shared_outs(%o = %tensor) -> tensor<...> {
    // body executes virtually in parallel per (i, j)
    scf.forall.in_parallel {
        // tensor insertion ops for combining results
    }
}
```

- Target-independent parallel iteration
- Threads are virtual; synchronization at completion
- `shared_outs` tensors accessed via block arguments

#### `scf.parallel` — Parallel Loop with Reductions
```mlir
%result = scf.parallel (%iv) = (%lb) to (%ub) step (%step) init(%init) -> f16 {
    // body
    scf.reduce(%partial) : f16 {
    ^bb0(%a: f16, %b: f16):
        %sum = arith.addf %a, %b : f16
        scf.reduce.return %sum : f16
    }
}
```

- Iteration order is unspecified
- Reductions must be associative + commutative for determinism

#### Terminators
| Operation | Used In | Purpose |
|-----------|---------|---------|
| `scf.yield %vals` | `scf.for`, `scf.while` (after), `scf.if` | Forward values to parent/next iteration |
| `scf.condition(%bool) %vals` | `scf.while` (before) | Continue or exit loop |
| `scf.forall.in_parallel { ... }` | `scf.forall` | Combine parallel results |
| `scf.reduce(%val) { ... }` | `scf.parallel` | Reduction combiner |

---

## Canonical Patterns

### Pattern: Data-Parallel Elementwise Operation

```mlir
func.func @elementwise_add() {
    // 1. Get tile ID and compute work partition
    %id = ktdp.get_compute_tile_id : index
    %start = arith.muli %id, %tile_size : index

    // 2. Construct memory views (no allocation, just interpretation)
    %A = ktdp.construct_memory_view %addr_a, sizes: [...], strides: [...] {
        coordinate_set = ..., memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<...>

    // 3. Construct access tiles (symbolic coordinates, no data movement)
    %A_tile = ktdp.construct_access_tile %A[%start, %c0] {
        access_tile_set = ...
    } : memref<...> -> !ktdp.tile<...xindex>

    // 4. Load data tiles
    %A_data = ktdp.load %A_tile : !ktdp.tile<...xindex> -> tensor<...>

    // 5. Compute using standard MLIR ops
    %C_out = tensor.empty() : tensor<...>
    linalg.add ins(%A_data, %B_data : ...) outs(%C_out : ...) -> tensor<...>

    // 6. Construct output access tile and store
    ktdp.store %C_out, %C_tile : tensor<...>, !ktdp.tile<...xindex>
}
```

### Pattern: Paged Tensor Access (Attention Kernels)

For paged KV-cache tensors where some dimensions are indirect:

```mlir
// Index tensor maps (batch, token_page) → physical page ID
%Idx = ktdp.construct_memory_view %idx_addr, sizes: [%Nb, %Ntkv_per_page], strides: [...] {
    ..., memory_space = #ktdp.spyre_memory_space<HBM>
} : memref<4x32xi32>

// Input tensor is laid out by physical pages
%X = ktdp.construct_memory_view %x_addr, sizes: [%Npages, %Nhkv, %Ptkv, %Ndkv], strides: [...] {
    ..., memory_space = #ktdp.spyre_memory_space<HBM>
} : memref<10000x8x64x128xf16>

// Indirect access: X[Idx[b, tkv/Ptkv], hkv, tkv % Ptkv, dkv]
%X_tile = ktdp.construct_indirect_access_tile
    intermediate_variables(%b, %h, %tkv, %dkv)
    %X[Idx[%b, %tkv / 64], %h, %tkv % 64, %dkv] {
        variables_space_set = ...
    } : memref<10000x8x64x128xf16> -> !ktdp.tile<4x8x2048x128xindex>
```

---

## Key Invariants and Constraints

1. **Stick alignment**: All memory accesses are 128-byte aligned. Innermost device dimension always equals max elements per stick for the dtype.
2. **Work partitioning is pre-decided**: KTDP expresses an already-fixed parallel decomposition. Each compute tile operates on its assigned region independently.
3. **Memory views don't allocate**: `construct_memory_view` only interprets existing memory — allocation is external.
4. **Access tiles don't access memory**: They materialize coordinate sets consumed by subsequent `load`/`store` ops.
5. **Access tile ordering**: An access tile does not dictate the order coordinates are accessed. Use multiple smaller tiles in a loop for explicit ordering.
6. **Layout compatibility**: Operands to an operation must have compatible memory layouts. The compiler propagates and validates layouts topologically.
7. **Tiled dimension coordinates combine right-to-left**: For repeated `dim_map` entries, e.g., `dim_map=[1, 2, 0, 2]` with device coords `(a, b, c, d)` → PyTorch dim 2 coordinate = `b*64 + d`.
8. **Sparse tensors**: Reductions along the stick dimension produce one element per stick. Represented via synthetic dimension (`dim_map = -1`).
9. **Canonical form**: PyTorch dimensions of size 1 are eliminated before computing the Spyre layout.

## Compilation Pipeline Context

```
PyTorch Model
  → torch.compile / Dynamo (FX Graph)
    → Inductor Frontend (decompositions, pre-passes)
      → LoopLevelIR with FixedTiledLayout (layout propagation via stickify)
        → Scheduler passes (core division, scratchpad planning)
          → KTDP IR generation (replaces former SuperDSC codegen)
            → DeepTools backend compiler (KTDP → device binaries)
```

KTDP sits at the boundary between the TorchInductor frontend and the DeepTools backend, replacing the previous SuperDSC-bundle format with an open MLIR-based representation.
