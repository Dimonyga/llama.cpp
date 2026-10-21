# Local llama-server builds

This branch keeps server code for CPU, CUDA, Vulkan, ASR, and vision-language use in one source tree. Use separate build directories for the CPU, CUDA and Vulkan configurations.

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

For the integrated Vega GPU on C206 (`192.168.8.206`):

```sh
cmake -S . -B build-local-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_VULKAN_NO_COOPMAT=ON -DGGML_VULKAN_NO_COOPMAT2=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-local-vulkan --target llama-server llama-bench test-backend-ops -j2
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json build-local-vulkan/bin/llama-server --list-devices
```

Use `--device Vulkan0` when launching this configuration. The NO_COOPMAT options omit unsupported shader families on Vega; keep their default OFF values for builds that must support matrix-core GPUs. See [Vulkan on C206](docs/development/vulkan-gcn.md) for the source comparison, correctness checks and measured performance.

## Gemma 4 12B on C206

Use the QAT Q4_0 text model with the Q8_0 unified multimodal projector. The 12.67 GB Q8_0 text model does not fit in C206's 9.27 GiB Vulkan memory budget with runtime buffers. The Q4_0 model is 7.22 GB and keeps audio and vision projection in Q8_0.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
export GGML_VK_VISIBLE_DEVICES=0

build-local-vulkan/bin/llama-server \
    -m /home/dimonyga/models/gemma-4-12b/gemma-4-12B-it-Q4_0.gguf \
    --mmproj /home/dimonyga/models/gemma-4-12b/mmproj-gemma-4-12B-it-Q8_0.gguf \
    --device Vulkan0 --mmproj-device Vulkan0 -ngl 99 \
    -c 4096 -np 1 -b 256 -ub 128 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --host 0.0.0.0 --port 8080
```

The OpenAI-compatible transcription route accepts the same multipart fields as a Whisper client. The client does not need to construct a Gemma chat message. For Gemma 4, the server selects the official speech-recognition prompt, treats `prompt` as optional recognition context and disables thinking.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions \
    -F file=@voice-message.ogg -F model=gemma-4-12b \
    -F language=ru -F response_format=json -F temperature=0
```

`/v1/audio/translations` uses the Gemma 4 speech-translation prompt. Its `target_language` extension defaults to `English`, matching the OpenAI API; pass `target_language=ru` for Russian.

```sh
curl http://127.0.0.1:8080/v1/audio/translations \
    -F file=@voice-message.ogg -F model=gemma-4-12b \
    -F language=en -F target_language=ru -F response_format=json -F temperature=0
```

`/v1/audio/speech` remains the text-to-speech route. It returns `not_supported_error` for Gemma 4 because this checkpoint generates text only; the route is retained for Qwen3-TTS and PocketTTS models.

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
