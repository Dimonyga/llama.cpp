# Local llama-server builds

This branch keeps server code for CPU, CUDA, ASR, and vision-language use in one source tree. CPU and CUDA still require separate build directories because they link different backends.

## Local changes

- `--moe-routing-log PATH` writes per-layer MoE expert-selection counts to JSON every 30 seconds and at shutdown. It is opt-in and adds graph synchronization overhead.
- The OpenAI audio transcription final response removes a leading `language <code> <asr_text>` marker emitted by Qwen3-ASR. The streaming delta response remains upstream behavior.

Multimodal image, video, and audio input are provided by upstream llama.cpp and use the usual `--mmproj` option. There is no separate VL source fork.

## Build

```sh
cmake -S . -B build-local-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build-local-cpu --target llama-server -j4
```

For the GTX 1080 Ti host with CUDA 12.8:

```sh
cmake -S . -B build-local-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=61
cmake --build build-local-cuda --target llama-server -j2
```

Use the CPU executable for CPU, ASR, and CPU VL deployments; use the CUDA executable for GPU deployments. Both executables come from the same commit.

## Update from upstream

After cloning this fork, add the upstream remote once:

```sh
git remote add upstream https://github.com/ggml-org/llama.cpp.git
```

For each update, work on a branch, merge upstream, rebuild both configurations, and verify model requests before updating deployed executables:

```sh
git fetch upstream master
git switch -c update-upstream
git merge upstream/master
```

Keep the two local changes as separate commits so upstream merges and patch review stay straightforward.
