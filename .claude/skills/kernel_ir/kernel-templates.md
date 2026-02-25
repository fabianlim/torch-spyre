# KTDP Kernel Templates

Reusable patterns and design rationale for KTDP MLIR kernels targeting the Spyre accelerator.

---

## Matrix Multiply — 2D Tiled, Data-Parallel

### Problem
`C[M,N] = A[M,K] * B[K,N]` distributed across P compute cores.

### Key Design Principles

1. **Partition the output matrix C in 2D**, not just row-strips. Row-only partitioning forces every core to load the entire B matrix, wasting HBM bandwidth.
2. **Minimize HBM↔scratchpad (LX) transfers.** Each `ktdp.load` / `ktdp.store` is an expensive data movement. Keep the C accumulator in scratchpad across the entire K-loop and store only once at the end.
3. **Tile the K dimension** and iterate with `scf.for` + `iter_args` to carry the accumulator as a loop-carried tensor.

### Tiling Strategy

Arrange P cores as a 2D grid of `(Gm × Gn)` where `Gm * Gn = P`:

| Parameter | Formula |
|---|---|
| Mt (rows per core) | M / Gm |
| Nt (cols per core) | N / Gn |
| Kt (K tile size) | Choose to balance scratchpad usage: A_tile[Mt,Kt] + B_tile[Kt,Nt] + C_acc[Mt,Nt] must fit in LX |
| K steps | K / Kt |

Core `(i, j)` computes `C[i*Mt:(i+1)*Mt, j*Nt:(j+1)*Nt]`.

**Grid shape heuristic**: Choose Gm, Gn so that Mt and Nt are roughly proportional to M and N, keeping tiles balanced. For `M=32, N=64, P=16`: use `Gm=4, Gn=4` → `Mt=8, Nt=16`.

### Data Movement Per Core

| Operation | Count | Shape | Purpose |
|---|---|---|---|
| A loads | K/Kt | Mt × Kt | One A strip per K-step |
| B loads | K/Kt | Kt × Nt | One B strip per K-step |
| C store | 1 | Mt × Nt | Single final write-back |
| **Total** | 2*(K/Kt) loads + 1 store | | |

### Concrete Example: C[32,64] = A[32,32] × B[32,64], 16 cores, f16

- Grid: 4×4, Mt=8, Nt=16, Kt=8, 4 K-steps
- Per core: 4 A loads (8×8) + 4 B loads (8×16) + 1 C store (8×16) = **8 loads + 1 store**
- vs. naive row-strip: each core loads all of B (32×64), **4× more B traffic**

```mlir
func.func @matmul_32x32_times_32x64(
    %addr_A: index,
    %addr_B: index,
    %addr_C: index
) {
    // ---- constants ----
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %c8  = arith.constant 8 : index    // Mt, Kt
    %c16 = arith.constant 16 : index   // Nt
    %c32 = arith.constant 32 : index   // K

    // ---- work partition: 4×4 grid ----
    %id = ktdp.get_compute_tile_id : index              // 0..15
    %core_i = arith.divui %id, %c4 : index              // row in grid (0..3)
    %core_j = arith.remui %id, %c4 : index              // col in grid (0..3)
    %m_start = arith.muli %core_i, %c8 : index          // M offset
    %n_start = arith.muli %core_j, %c16 : index         // N offset

    // ---- HBM memory views ----

    %A = ktdp.construct_memory_view %addr_A,
             sizes: [32, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) :
            (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x32xf16>

    %B = ktdp.construct_memory_view %addr_B,
             sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) :
            (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    %C = ktdp.construct_memory_view %addr_C,
             sizes: [32, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) :
            (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<32x64xf16>

    // ---- initialize accumulator in scratchpad ----
    %zero = arith.constant 0.0 : f16
    %C_init = tensor.empty() : tensor<8x16xf16>
    %C_acc_init = linalg.fill ins(%zero : f16)
                      outs(%C_init : tensor<8x16xf16>) -> tensor<8x16xf16>

    // ---- K-loop: 4 steps of Kt=8, accumulate C in scratchpad ----
    %C_final = scf.for %k_step = %c0 to %c32 step %c8
                   iter_args(%C_acc = %C_acc_init) -> tensor<8x16xf16> {

        // -- load A tile [8,8] from HBM --
        %A_tile = ktdp.construct_access_tile %A[%m_start, %k_step] {
            access_tile_set = affine_set<(d0, d1) :
                (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 7 >= 0)>
        } : memref<32x32xf16> -> !ktdp.tile<8x8xindex>

        %A_data = ktdp.load %A_tile
            : !ktdp.tile<8x8xindex> -> tensor<8x8xf16>

        // -- load B tile [8,16] from HBM --
        %B_tile = ktdp.construct_access_tile %B[%k_step, %n_start] {
            access_tile_set = affine_set<(d0, d1) :
                (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>
        } : memref<32x64xf16> -> !ktdp.tile<8x16xindex>

        %B_data = ktdp.load %B_tile
            : !ktdp.tile<8x16xindex> -> tensor<8x16xf16>

        // -- accumulate: C_acc[8,16] += A[8,8] * B[8,16] --
        %C_acc_next = linalg.matmul
            ins(%A_data, %B_data : tensor<8x8xf16>, tensor<8x16xf16>)
            outs(%C_acc : tensor<8x16xf16>) -> tensor<8x16xf16>

        scf.yield %C_acc_next : tensor<8x16xf16>
    }

    // ---- store C tile [8,16] to HBM (once) ----
    %C_tile = ktdp.construct_access_tile %C[%m_start, %n_start] {
        access_tile_set = affine_set<(d0, d1) :
            (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 + 15 >= 0)>
    } : memref<32x64xf16> -> !ktdp.tile<8x16xindex>

    ktdp.store %C_final, %C_tile
        : tensor<8x16xf16>, !ktdp.tile<8x16xindex>

    return
}
```

### Anti-Patterns to Avoid

| Anti-pattern | Why it's bad | Correct approach |
|---|---|---|
| **Row-only partitioning** | Every core loads the full B matrix; HBM bandwidth wasted | Partition C in 2D so each core loads only its B strip |
| **Storing C every K-step** | Unnecessary HBM writes; C can stay in scratchpad | Use `scf.for` with `iter_args` to carry accumulator; store once |
| **Loading full A or B per core** | Oversized transfers that don't fit scratchpad | Tile K dimension, load smaller A and B strips per step |
| **No K-tiling** | Forces large A and B tiles into scratchpad simultaneously | Tile K to control scratchpad pressure |

### Generalization Checklist

- [ ] Adjust Gm × Gn grid to match core count and M/N aspect ratio
- [ ] Choose Kt so `Mt*Kt + Kt*Nt + Mt*Nt` (in elements × dtype size) fits scratchpad
- [ ] Handle non-divisible dimensions with padding or coordinate_set bounds
- [ ] For very large K, consider double-buffering: overlap next A/B load with current matmul
