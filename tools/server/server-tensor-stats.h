#pragma once

// Live activation statistics for self-made quantization.
//
// Collects, per weight tensor, the sum of squared inputs (and the number of rows) exactly like
// tools/imatrix does, from the traffic the server really processes. The snapshot is a regular GGUF
// imatrix, so it feeds `llama-quantize --imatrix` directly. Only batches of at least `min_rows`
// rows are observed: single-token decode steps would force a device->host copy per matmul.

#include "common.h"
#include "imatrix-loader.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct server_tensor_stats {
    std::mutex mutex;
    std::string path;
    int64_t min_rows = 16;
    std::map<std::string, std::vector<float>>   sums;
    std::map<std::string, std::vector<int64_t>> counts;
    std::vector<char> src1_data;
    std::vector<char> ids_data;
    bool dirty = false;
    std::chrono::steady_clock::time_point last_flush = std::chrono::steady_clock::now();

    // Continue from the previous snapshot so restarts accumulate instead of resetting.
    void configure(const std::string & output_path, int64_t min_rows_) {
        std::lock_guard<std::mutex> lock(mutex);
        path = output_path;
        min_rows = min_rows_ > 0 ? min_rows_ : 1;
        common_imatrix previous;
        if (common_imatrix_load(path, previous) && !previous.is_legacy) {
            for (auto & [name, entry] : previous.entries) {
                sums[name]   = entry.sums;
                counts[name] = entry.counts;
            }
            LOG_INF("[tensor-stats] resumed %zu tensors from %s\n", sums.size(), path.c_str());
        }
    }

    static std::string weight_name(const char * name) {
        // "CUDA0#blk.0.attn_k.weight#0" => "blk.0.attn_k.weight"
        const char * p = std::strchr(name, '#');
        if (p == nullptr) {
            return name;
        }
        ++p;
        const char * q = std::strchr(p, '#');
        return q == nullptr ? std::string(p) : std::string(p, q - p);
    }

    bool observe(ggml_tensor * t, bool ask) {
        if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        const ggml_tensor * src0 = t->src[0];
        const ggml_tensor * src1 = t->src[1];
        if (src0 == nullptr || src1 == nullptr || src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float)) {
            return false;
        }
        const bool    is_id = t->op == GGML_OP_MUL_MAT_ID;
        const int64_t rows  = is_id ? src1->ne[2] : src1->ne[1];
        if (rows < min_rows || std::strncmp(weight_name(src0->name).c_str(), "blk.", 4) != 0) {
            return false;
        }
        if (ask) {
            return true;
        }

        const std::string name = weight_name(src0->name);
        const bool is_host = ggml_backend_buffer_is_host(src1->buffer);
        std::lock_guard<std::mutex> lock(mutex);
        if (!is_host) {
            src1_data.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, src1_data.data(), 0, src1_data.size());
        }
        const char * data = is_host ? (const char *) src1->data : src1_data.data();
        const int64_t ne0 = src1->ne[0];

        if (is_id) {
            const ggml_tensor * ids = t->src[2];
            const int64_t n_as  = src0->ne[2];
            const int64_t n_ids = ids->ne[0];
            if (ids->ne[1] != src1->ne[2] || ggml_nrows(src1) != src1->ne[1] * src1->ne[2]) {
                return true;
            }
            ids_data.resize(ggml_nbytes(ids));
            ggml_backend_tensor_get(ids, ids_data.data(), 0, ids_data.size());
            auto & s = sums[name];
            auto & c = counts[name];
            if (s.empty()) {
                s.assign(ne0 * n_as, 0.0f);
                c.assign(n_as, 0);
            }
            if (s.size() != (size_t) (ne0 * n_as) || c.size() != (size_t) n_as) {
                return true; // snapshot of another model shape: ignore
            }
            for (int64_t idx = 0; idx < n_ids; ++idx) {
                for (int64_t row = 0; row < rows; ++row) {
                    const int32_t ex = *(const int32_t *) (ids_data.data() + row * ids->nb[1] + idx * ids->nb[0]);
                    if (ex < 0 || ex >= n_as) {
                        continue;
                    }
                    const float * x   = (const float *) (data + (idx % src1->ne[1]) * src1->nb[1] + row * src1->nb[2]);
                    float *       acc = s.data() + ex * ne0;
                    c[ex]++;
                    for (int64_t j = 0; j < ne0; ++j) {
                        acc[j] += x[j] * x[j];
                    }
                }
            }
        } else {
            const int64_t n_mat = src0->ne[2] * src0->ne[3];
            auto & s = sums[name];
            auto & c = counts[name];
            if (s.empty()) {
                s.assign(ne0 * n_mat, 0.0f);
                c.assign(1, 0);
            }
            if (s.size() != (size_t) (ne0 * n_mat) || c.size() != 1) {
                return true;
            }
            for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                    float * acc = s.data() + ((i3 % src0->ne[3]) * src0->ne[2] + (i2 % src0->ne[2])) * ne0;
                    for (int64_t row = 0; row < src1->ne[1]; ++row) {
                        const float * x = (const float *) (data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                        for (int64_t j = 0; j < ne0; ++j) {
                            acc[j] += x[j] * x[j];
                        }
                    }
                }
            }
            c[0] += ggml_nrows(src1) / n_mat;
        }
        dirty = true;
        if (std::chrono::steady_clock::now() - last_flush >= std::chrono::seconds(60)) {
            write_snapshot_locked();
            last_flush = std::chrono::steady_clock::now();
        }
        return true;
    }

    void write_snapshot_locked() {
        if (path.empty() || !dirty) {
            return;
        }
        size_t data_size = 0;
        for (const auto & [name, s] : sums) {
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * s.size(), GGML_MEM_ALIGN);
            data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * counts.at(name).size(), GGML_MEM_ALIGN);
        }
        ggml_init_params params = { data_size + ggml_tensor_overhead() * 4, nullptr, false };
        ggml_context *   ctx    = ggml_init(params);
        gguf_context *   gguf   = gguf_init_empty();
        if (ctx == nullptr || gguf == nullptr) {
            return;
        }
        int64_t max_rows = 0;
        for (const auto & [name, c] : counts) {
            for (int64_t v : c) {
                max_rows = std::max(max_rows, v);
            }
        }
        const char * dataset = "server-traffic";
        gguf_set_val_str(gguf, "general.type", "imatrix");
        gguf_set_arr_str(gguf, LLM_KV_IMATRIX_DATASETS, &dataset, 1);
        gguf_set_val_u32(gguf, LLM_KV_IMATRIX_CHUNK_COUNT, (uint32_t) (max_rows / 512));
        gguf_set_val_u32(gguf, LLM_KV_IMATRIX_CHUNK_SIZE, 512);
        for (const auto & [name, s] : sums) {
            const auto & c = counts.at(name);
            const int64_t n_mat = (int64_t) c.size();
            ggml_tensor * in_sum2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) s.size() / n_mat, n_mat);
            ggml_tensor * cnt     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_mat);
            ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
            ggml_format_name(cnt, "%s.counts", name.c_str());
            std::memcpy(in_sum2->data, s.data(), s.size() * sizeof(float));
            for (int64_t j = 0; j < n_mat; ++j) {
                ((float *) cnt->data)[j] = (float) c[j];
            }
            gguf_add_tensor(gguf, in_sum2);
            gguf_add_tensor(gguf, cnt);
        }
        const std::string tmp = path + ".tmp";
        if (gguf_write_to_file(gguf, tmp.c_str(), false) && std::rename(tmp.c_str(), path.c_str()) == 0) {
            dirty = false;
        } else {
            fprintf(stderr, "[tensor-stats] failed to write %s\n", path.c_str());
        }
        gguf_free(gguf);
        ggml_free(ctx);
    }

    ~server_tensor_stats() {
        std::lock_guard<std::mutex> lock(mutex);
        write_snapshot_locked();
    }
};
