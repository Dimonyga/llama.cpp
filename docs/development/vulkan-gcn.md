# Vulkan on C206

## Target and scope

The test host is `192.168.8.206`, an AMD Ryzen 7 2700U with integrated Vega graphics, 30 GiB RAM, AlmaLinux 9.8 and Mesa RADV 25.2.7. RADV exposes `AMD Radeon Vega 3 / 10 Graphics (RADV RAVEN)`, Vulkan 1.4.318, FP16 arithmetic and 64-lane subgroups. Neither KHR cooperative matrices nor NV cooperative matrix 2 is available. The separate Topaz adapter is present on PCI but RADV cannot open its render node; it is not a second usable Vulkan device in this setup.

This audit compares this fork at `8585fb70f` with the neighboring `ik_llama.cpp` checkout at `35e72560fac6eb4e0bedc99976d3a12040749230`. Runtime optimizations are limited to GCN devices; RMS additionally requires subgroup arithmetic and full compute subgroups. Other devices keep their existing dispatch, including the 512-thread shared-memory RMS reduction.

## Source comparison

| Candidate | Decision |
| --- | --- |
| ik fused RMS norm, `b445c83e` / [ik #577](https://github.com/ikawrakow/ik_llama.cpp/pull/577) | Already provided by Vulkan graph fusion here, including additional ADD, RoPE and SET_ROWS combinations. Do not introduce ik's separate graph opcode. |
| ik fused activation and multiply, `8d4f0a61` / [ik #580](https://github.com/ikawrakow/ik_llama.cpp/pull/580) | Existing UNARY + MUL fusion and GLU kernels cover the same work. |
| ik multi-add, `28e81fc7` / [ik #582](https://github.com/ikawrakow/ik_llama.cpp/pull/582) | Existing backend fusion supports multiple sources and also calculates RMS partial sums. Retain it. |
| ik DeepSeek FA and split-K, `2fddc45a`, `19c57dbe` / [ik #584](https://github.com/ikawrakow/ik_llama.cpp/pull/584), [ik #608](https://github.com/ikawrakow/ik_llama.cpp/pull/608) | Based on older upstream kernels. Current kernels support mixed head sizes, quantized KV, sinks and newer device dispatch. Copying the old shaders would remove functionality. |
| ik KQ mask padding, `c574620b` / [ik #2350](https://github.com/ikawrakow/ik_llama.cpp/pull/2350) | An ik-specific fix for its graph layout. This tree builds masks through the current KV-cache graph inputs and has its own padding/shape checks. Do not copy the old global mask constant. |
| ik IQ4_KS / IQ4_KT shaders, `64109a4d` / [ik #2332](https://github.com/ikawrakow/ik_llama.cpp/pull/2332) | Require ik quantization types and per-row scale metadata absent from this tree. Not a compatible speed patch for existing GGUF models. |
| ik optional cooperative-matrix builds, `58f3bda0` / [ik #674](https://github.com/ikawrakow/ik_llama.cpp/pull/674) | Adapted the NO_COOPMAT and NO_COOPMAT2 options to the current shader generator. Defaults preserve automatic detection. Explicit OFF values also reach the child CMake cache when reconfiguring an existing build. |
| Subgroup RMS norm and smaller workgroups, [upstream #29882](https://github.com/ggml-org/llama.cpp/pull/29882), `a5cca5e28504` | Adapted for GCN only. Subgroups reduce each wave, followed by one shared-memory exchange between waves. Short rows use 128/256 threads instead of 512. Existing scalar reduction remains the fallback. |
| Q4_0 KV dequant + transpose, [upstream #30738](https://github.com/ggml-org/llama.cpp/pull/30738), `c2709db06823` | Adapted for GCN only. Extends the existing Q8_0 scratch-layout path to Q4_0 and mixed Q4_0/Q8_0 caches during prefill. Other devices and their Q8_0 dispatch remain unchanged. No new quantization format is added. |
| Tiled transpose-concat for GDN, [upstream #30591](https://github.com/ggml-org/llama.cpp/pull/30591), `52e41a44519e` | Adapted for GCN only. A 32x32 shared-memory tile coalesces reads from the transposed current input and writes it after the convolution history. Unsupported layouts, 64-bit values, quantized types and tile grids exceeding device limits keep the generic concat path. |
| RDNA split-K, cooperative-matrix and workgroup scheduling changes | Not applicable to Vega's scalar GCN path. Many are already in the fork; pending RDNA-only tuning is not enabled on GCN. |

The RMS adaptation retains upstream's fused RoPE staging bound: at most two elements per thread, with a maximum of 1024 columns. SET_ROWS without RoPE has no such staging limit and is tested separately with odd and large row widths. Tests cover broadcast norm weights, F16/F32 destinations and both normal and NeoX RoPE.

Concat tests cover each new storage width, odd tile boundaries, empty history, transposed history, higher-dimensional fallbacks and grids above the device workgroup limit. Quantized-KV tests cover dense cache layout, GQA and mixed Q4_0/Q8_0 pairs. The existing backend operator tests and CPU reference are reused.

## Build

On AlmaLinux, the build dependencies used here are:

```sh
sudo dnf install gcc-c++ cmake glslc vulkan-tools vulkan-loader-devel vulkan-headers spirv-headers-devel
```

Use a separate directory from CPU and CUDA builds:

```sh
cmake -S . -B build-local-vulkan \
    -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF \
    -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
    -DGGML_VULKAN_NO_COOPMAT=ON -DGGML_VULKAN_NO_COOPMAT2=ON \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-local-vulkan --target llama-server llama-bench llama-perplexity test-backend-ops -j2
```

The two NO_COOPMAT options reduce the compiled shader set; they do not accelerate inference on a GPU that already lacks these features. Leave them OFF when building a portable binary for other GPUs. NO_COOPMAT2 also disables the associated decode-vector shader variants. Integer-dot and BF16 compiler checks remain independent.

Select RADV explicitly so software Vulkan devices are not benchmarked by accident:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
export GGML_VK_VISIBLE_DEVICES=0
build-local-vulkan/bin/llama-bench --list-devices
build-local-vulkan/bin/test-backend-ops test -b Vulkan0 -o '.*RMS_NORM.*'
build-local-vulkan/bin/test-backend-ops perf -b Vulkan0 -o '.*RMS_NORM.*'
```

## Validation

Measurements and validation results are recorded below after execution on the target. Source patches, build logs, model files, comparison scripts and raw results are kept in `/tmp/llama-vulkan-20261020` on C206. Existing services are left running; CPU scheduling, shared memory bandwidth and thermals can affect results.
