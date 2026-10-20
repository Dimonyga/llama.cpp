# ik_llama.cpp comparison for Pascal and AVX2

Compared on 2026-10-20:

- Local llama.cpp: `12129f717`.
- ik_llama.cpp: `35e72560fac6eb4e0bedc99976d3a12040749230`.
- Intended GPU: GTX 1080 Ti, compute capability 6.1.
- Intended CPU: Ryzen 7 3800X, AVX2/FMA, without AVX-512 or VNNI.
- GPU workload: the supplied mixed-quant Qwen model, fully offloaded. Its tensor types include IQ1_S, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, Q2_K, Q4_K, Q5_K and Q8_0. IQ1_M was also requested for other models.

## Changes retained

| Area | Comparison and source | Adaptation |
| --- | --- | --- |
| Pascal MMVQ decode | [llama.cpp #25479](https://github.com/ggml-org/llama.cpp/pull/25479) uses two warps for one output column on DP4A Pascal. This candidate comes from mainline, not ik. | Select two warps on sm_61/sm_62. Preserve the current two-row paths, including IQ1_M and Q2_K. Other architectures and multi-column batches retain their previous tables. Performance of individual IQ types remains unmeasured. |
| Pascal tile FA | [ik #2142](https://github.com/ikawrakow/ik_llama.cpp/pull/2142), `7ae6b337`, keeps K/V tiles as half2 in shared memory and converts them to FP32 in registers. | Apply the storage part to the current sm_61/sm_62 FP32 tile path. Keep Q, scores, softmax and output accumulators in FP32. The current code already keeps raw QK scores in registers, so that part needs no port. |
| Qwen delta-net | [ik #1333](https://github.com/ikawrakow/ik_llama.cpp/pull/1333), restored by #1340, shares Q/K across the block and computes output from the old state before its update. [#1649](https://github.com/ikawrakow/ik_llama.cpp/pull/1649) establishes the required token-loop synchronization. | Adapt these ideas to the current state layout for scalar-gate, head-size-128 GDN on DP4A Pascal. Keep state in registers, existing GQA/sequence strides, rollback snapshots and direct-cache writes. |
| Qwen output gate | The fork's `delta_net::build_gated_output` uses `ggml_fused_mul_unary` for SiLU gating. | Express the same operation using the existing `ggml_swiglu_split` in Qwen3.5 dense/MoE and Qwen3-Next. Preserve the original graph for layouts that do not satisfy the GLU contiguity requirement. CUDA already fuses the original chain; the explicit fused operator primarily helps CPU execution and graph size. |
| CPU quantized FA prefill | The fork's IQK FA uses blocked attention for quantized caches, starting with [#51](https://github.com/ikawrakow/ik_llama.cpp/pull/51). The local tiled path only accepted matching F16/F32 caches. | Reuse the local SIMD GEMM and online-softmax implementation for Q4_0/Q8_0 K and V, including mixed types. Dequantize a KV tile once per query tile instead of once per query row. Add one DK-sized row buffer to each thread's planned scratch allocation. No IQK subsystem or new quant format is introduced. |

For example, the FA configuration with `nbatch_fa=64`, `nbatch_K=128` previously used 33792 bytes for its K/V staging buffer. Half2 staging uses 17408 bytes, excluding the other buffers. This is a storage calculation, not a measured occupancy or throughput claim.

The adapted delta-net formula is:

```text
delta = beta * (v - decay * (S^T * k))
output = scale * (decay * (S^T * q) + delta * dot(k, q))
S_new = decay * S + k * delta^T
```

The fork's sigmoid on beta is not copied: the current Qwen graph already applies it. The fork's decay/state clamps are not copied either, because they change numerical behavior. The current vector-gate/KDA kernel and other head sizes keep their original implementation.

## Other candidates examined

| Candidate | Result |
| --- | --- |
| Quantized KV copy acceleration, [ik #2279](https://github.com/ikawrakow/ik_llama.cpp/pull/2279), `7642ac3e` | The local `cpy.cu` already batches quantized copies into CUDA_CPY_BLOCK_SIZE threads. It does not launch a one-thread block for every quant block as the old fork code did. There is no equivalent missing launch fix to port. |
| Better Q4_0 KV scale, [ik #1547](https://github.com/ikawrakow/ik_llama.cpp/pull/1547), `b9a2ce46`, and [#1556](https://github.com/ikawrakow/ik_llama.cpp/pull/1556), `bd7daa71` | Implemented and tested during comparison, then removed. It minimizes weighted reconstruction error and improves accuracy, but adds work when writing KV. The local x86_64/Rosetta copy microbenchmark changed from 2079 us to 4809 us for 8192 x 512 x 2 F32 -> Q4_0 values. This is not a speed optimization at an unchanged cache size. Final quantizers are unchanged. |
| Hadamard K/V cache transforms | Useful for low-bit cache accuracy, but require matching query/output transforms. They are not a direct throughput improvement for an existing Q4_0 cache. Not included. |
| GQA MMA FA, [ik #370](https://github.com/ikawrakow/ik_llama.cpp/pull/370), `711ba7e8`, and Qwen special handling [#1328](https://github.com/ikawrakow/ik_llama.cpp/pull/1328) | These paths require tensor-core MMA instructions unavailable on GTX 1080 Ti. Forcing their dispatch on sm_61 would not provide a working optimization. |
| Delta-net state-copy fusion, [ik #2251](https://github.com/ikawrakow/ik_llama.cpp/pull/2251), `af4e0cbd` | Local CUDA already recognizes GDN plus snapshot CPY and writes the cache directly. Local GDN also already keeps recurrent state in registers. These parts are retained rather than duplicated. |
| Tiny F32 GEMM for Qwen3.8, [ik #2373](https://github.com/ikawrakow/ik_llama.cpp/pull/2373), `15dddc60` | The supplied alpha/beta matrices are Q8_0, so the fork's F32-only dispatch does not accelerate those projections. They use MMVQ instead. |
| IQ1_M and IQ3_S CUDA vec-dot | The relevant non-repacked implementations are already substantially the same. The fork's faster R4 formats require a different internal layout. The retained Pascal MMVQ table applies to existing GGUF tensor types without requantizing weights. |
| Q2_K CUDA vec-dot | The fork processes one packed word per call; local code processes two and combines scale loads. Copying the older function would undo this batching. The changed Pascal launch table still covers Q2_K. |
| AVX2 R8 GEMM, [ik #2472](https://github.com/ikawrakow/ik_llama.cpp/pull/2472), `659d691b` | Its broadcasts operate on the fork's R8 packing. Local Q4/Q8 already have their own repacked GEMM/GEMV kernels with different layouts. This is not a byte-compatible kernel replacement; no model-weight GEMM change is claimed here. |
| CPU quantized decode KV splitting | Tested as an additional candidate, then removed. Three local Q4/Q8 decode microcases showed approximately neutral results, including a roughly 3.5% difference in the unmasked Q8 case that is insufficient to establish a general improvement. Final CPU decode dispatch is unchanged. |

Both trees still convert quantized KV to F16 before the generic CUDA tile path. This patch reduces shared-memory staging cost inside that kernel; it does not eliminate the whole-cache conversion. The existing vector FA path already consumes Q4_0/Q8_0 directly for supported decode shapes.

## Validation

### Local ARM and Rosetta checks

CPU measurements use an Apple M3 running an x86_64 AVX2/FMA/F16C build through Rosetta, with four worker threads, Accelerate/BLAS/OpenMP disabled. These are local operation microbenchmarks, not Ryzen 3800X or end-to-end model measurements.

The prefill cases use 8 query heads, 2 KV heads, 64 query tokens and 1024 cached tokens:

| KV type | Head size | Before, us | After, us | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 128 | 13547.29 | 7909.59 | 1.71x |
| Q4_0 | 256 | 24712.02 | 14763.84 | 1.67x |
| Q8_0 | 128 | 11849.30 | 8235.57 | 1.44x |
| Q8_0 | 256 | 19420.23 | 15315.63 | 1.27x |

Focused CPU FA checks cover mixed cache types, non-contiguous tensors, different K/V head sizes, partial tiles, GQA, multiple sequences, ALiBi, softcap, sinks and a shared K/V MLA view: 23/23 pass on both native ARM and x86_64/AVX2. The same 23 cases pass under AddressSanitizer. Four split-SwiGLU cases and the quantization-function suite pass on both architectures. Six focused GDN cases pass on the CPU; this does not execute the changed CUDA kernel. Additional GDN cases cover Qwen-shaped 128-wide heads with a value-head repeat of 3, token counts 1/8/65 and rollback snapshots. The native llama library builds with the changed Qwen graphs.

### Native AMD CPU checks

Host 192.168.8.206 was used only for CPU validation. It has a Ryzen 7 2700U (Zen 1, 4 cores / 8 threads), not a 3800X. Both source trees were exported from `12129f717` and received the same additional test cases; only the optimized tree received the implementation changes. GCC 11.5 builds use AVX2/FMA/F16C/BMI2, with native tuning, AVX-512, VNNI, OpenMP, BLAS and CUDA disabled.

The same four prefill shapes above were measured with four worker threads pinned to logical CPUs 0,2,4,6, one thread per physical core. One initial baseline/optimized pair was discarded, followed by seven pairs with alternating execution order. The table reports median time and the ratio of medians. No compilation ran during these measurements. Existing services were left running, so background load and the unchanged schedutil governor still cause variation; these are not isolated-machine or end-to-end inference results.

| KV type | Head size | Before median, us | After median, us | Ratio | Before range, us | After range, us |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 128 | 35831.83 | 9941.28 | 3.60x | 32863.88-36951.80 | 9604.70-11019.42 |
| Q4_0 | 256 | 64643.07 | 19635.98 | 3.29x | 60621.03-68541.63 | 19207.07-21952.78 |
| Q8_0 | 128 | 18778.40 | 9811.57 | 1.91x | 17701.47-19327.57 | 9284.68-10252.55 |
| Q8_0 | 256 | 31807.29 | 17969.08 | 1.77x | 28301.47-32793.71 | 16887.52-18603.98 |

Both native builds pass the 23 focused FA cases, 6 focused CPU GDN cases, 4 split-SwiGLU cases and the full quantization-function suite. This supports retaining the quantized-KV prefill path on native AVX2, rather than relying only on Rosetta measurements. Model-weight Q4/Q8 GEMM/GEMV and CPU single-token attention dispatch are unchanged, and their speed is not represented by this table.

Source archives, patches, builds, the test script and raw logs remain under `/tmp/llama-ik-bench-20261020` on 192.168.8.206. Local copies of the logs and a median calculator are under `/private/tmp/llama-cpu-tools`. No active checkout or service configuration was modified.

### CUDA sm_61 checks

The current CUDA validation scope is sm_61 only. The Clang 21 front end with NVIDIA CUDA 12.8 headers passes device checks for all 12 tile head-size instances and the changed GDN/MMVQ translation units. A temporary declaration for the legacy cudaConfigureCall name is supplied only to accommodate the Darwin front end. This is not an nvcc/PTX build or a GPU correctness/performance run. The occupied GPU host was not accessed. Host 192.168.8.206 is used only for CPU validation.

## Target-machine checks

Use CUDA 12.x for sm_61. When the target machine is available:

```sh
cmake -S . -B build-pascal -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DLLAMA_BUILD_TESTS=ON
cmake --build build-pascal --target llama-server llama-bench test-backend-ops -j 8
build-pascal/bin/test-backend-ops test -b CUDA0 -o GATED_DELTA_NET -p 'head_count=16,head_size=128,n_seq_tokens=(1|8|65),n_seqs='
build-pascal/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p 'hsk=(128|256),hsv=(128|256).*kv=(113|1024),nb=(1|64|65),'
build-pascal/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p 'type_a=(iq1_m|iq3_s|q2_K).*m=4096,n=1,'
```

Compare the same GGUF, KV types, context, prompt, batch and server options against an unmodified build. Check prompt processing and token generation separately, including populated long contexts and Qwen recurrent-state reuse. Model inference speed, perplexity and CUDA race checking remain target-machine validation items.
