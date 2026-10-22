#pragma once

#include "common.cuh"

// Single token flash attention decode for a q4_0 KV cache on dp4a parts (Pascal): one block works on one KV head and
// all query heads of its GQA group, so the KV cache is read and dequantized once per group instead of once per head.

bool ggml_cuda_fattn_gqa_q_supported(int device, const ggml_tensor * dst);

void ggml_cuda_fattn_gqa_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
