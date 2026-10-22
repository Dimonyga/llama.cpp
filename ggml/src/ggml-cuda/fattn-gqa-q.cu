#include "fattn-gqa-q.cuh"
#include "fattn-common.cuh"

#include <algorithm>
#include <cstdlib>

// Layout of a q4_0 block: half d, 16 bytes qs. Byte j holds element j in its low and element j + 16 in its high nibble.
// The rows of the KV cache are only 2 byte aligned (18 byte blocks), so everything is read as 16 bit words.
//
// A block of GQAQ_NT threads walks GQAQ_TILE KV positions per iteration and works on all G query heads that share one KV head:
//   1. KQ: NB = D/32 threads per position, one q4_0 block each, dp4a against Q quantized like q8_1, the NB partial sums are
//      combined with shuffles.
//   2. Online softmax over the tile, one position per thread.
//   3. VKQ: every thread accumulates 4 elements of V for all G heads over a subset of the positions of the tile.
// If the KV range of a head is split over several blocks the results are merged by flash_attn_combine_results.

#define GQAQ_NT   128 // threads per block
#define GQAQ_TILE 128 // KV positions per iteration, one per thread in the softmax step
#define GQAQ_EPT    4 // elements of V per thread
#define GQAQ_BPT    2 // q4_0 bytes of V per thread, one 16 bit word

static __device__ __forceinline__ unsigned short gqaq_ld16(const char * p) {
    return *(const unsigned short *) p;
}

// Nibble i (0..3) of a word with the nibbles masked into the low bits of its bytes, as float and offset by -8:
// (2^23 + n) - (2^23 + 8). Avoids the slow integer to float conversion.
static __device__ __forceinline__ float gqaq_nibble(const int w_masked, const int i) {
    return __int_as_float(__byte_perm(w_masked, 0x4B000000, 0x7440 | i)) - (8388608.0f + 8.0f);
}

template <int D, int G>
__launch_bounds__(GQAQ_NT, G >= 8 ? 4 : 5)
static __global__ void flash_attn_gqa_q4_0(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V, const char * __restrict__ mask,
        float * __restrict__ dst, float2 * __restrict__ dst_meta, const float scale,
        const int64_t nb02, const int64_t nb03, const int64_t nb11, const int64_t nb12, const int64_t nb13,
        const int64_t nb21, const int64_t nb22, const int64_t nb23, const int64_t nb33, const int ne33,
        const int ntiles) {
    constexpr int NT   = GQAQ_NT;
    constexpr int TILE = GQAQ_TILE;
    static_assert(TILE == NT, "one position per thread in the softmax step");
    constexpr int NB  = D/32;        // q4_0 blocks per row, threads per position in the KQ step
    constexpr int PP  = NT/NB;       // positions per pass of the KQ step
    constexpr int DG  = D/GQAQ_EPT;  // threads per position in the VKQ step
    constexpr int PG  = NT/DG;       // position groups in the VKQ step
    constexpr int TPB = 16/GQAQ_BPT; // VKQ threads per q4_0 block
    constexpr int NW  = NT/WARP_SIZE;

    const int pbi  = blockIdx.x;
    const int pb   = gridDim.x;
    const int hk   = blockIdx.y;
    const int seq  = blockIdx.z;
    const int Hq   = gridDim.y*G;
    const int t    = threadIdx.x;
    const int lane = t % WARP_SIZE;
    const int warp = t / WARP_SIZE;

    const int c  = t % NB;  // KQ step: q4_0 block of the row
    const int pp = t / NB;  // KQ step: position within the pass
    const int dg = t % DG;  // VKQ step: which elements of the row
    const int pg = t / DG;  // VKQ step: which positions

    const char * Kh = K + nb13*seq + nb12*hk;
    const char * Vh = V + nb23*seq + nb22*hk;
    const half * maskh = mask ? (const half *) (mask + nb33*(seq % ne33)) : nullptr;

    // Q of the G heads of this group, quantized per 32 elements like q8_1 with the softmax scale folded in:
    __shared__ __align__(16) int sQi[G][NB][8];
    __shared__ float2 sQd[G][NB]; // d, sum
    if (t < G*NB) {
        const int g  = t / NB;
        const int cb = t % NB;
        const float * qf = (const float *) (Q + nb03*seq + nb02*(hk*G + g)) + cb*32;
        float v[32];
        float amax = 0.0f;
        float sum  = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            v[i]  = scale*qf[i];
            amax  = fmaxf(amax, fabsf(v[i]));
            sum  += v[i];
        }
        const float d = amax / 127;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q32 = 0;
            if (d != 0.0f) {
                int8_t * q8 = (int8_t *) &q32;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    q8[l] = roundf(v[4*i + l] / d);
                }
            }
            sQi[g][cb][i] = q32;
        }
        sQd[g][cb] = make_float2(d, sum);
    }
    __syncthreads();

    __shared__ __align__(16) float sP[TILE][8]; // KQ scores, then the softmax weights, of the tile
    __shared__ float s_red[G][NW];

    float acc[G][GQAQ_EPT];
    float m[G];
    float lsum[G]; // softmax denominator of the position this thread owns in the softmax step
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g]    = -FLT_MAX/2.0f;
        lsum[g] = 0.0f;
#pragma unroll
        for (int e = 0; e < GQAQ_EPT; ++e) {
            acc[g][e] = 0.0f;
        }
    }

    for (int tile = pbi; tile < ntiles; tile += pb) {
        const int pos0 = tile*TILE;

        // KQ of the tile for all G heads
#pragma unroll 2
        for (int pass = 0; pass < TILE/PP; ++pass) {
            const int pos_t = pass*PP + pp;
            const char * kb = Kh + int64_t(pos0 + pos_t)*nb11 + c*18;

            const float dk = __half2float(__ushort_as_half(gqaq_ld16(kb)));
            int lo[4];
            int hi[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int w = int(gqaq_ld16(kb + 2 + 4*i)) | (int(gqaq_ld16(kb + 4 + 4*i)) << 16);
                lo[i] = w & 0x0F0F0F0F;
                hi[i] = (w >> 4) & 0x0F0F0F0F;
            }

            float r[G];
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int4 q0 = *(const int4 *) &sQi[g][c][0];
                const int4 q1 = *(const int4 *) &sQi[g][c][4];
                const float2 qds = sQd[g][c];
                int s = 0;
                s = ggml_cuda_dp4a(lo[0], q0.x, s);
                s = ggml_cuda_dp4a(lo[1], q0.y, s);
                s = ggml_cuda_dp4a(lo[2], q0.z, s);
                s = ggml_cuda_dp4a(lo[3], q0.w, s);
                s = ggml_cuda_dp4a(hi[0], q1.x, s);
                s = ggml_cuda_dp4a(hi[1], q1.y, s);
                s = ggml_cuda_dp4a(hi[2], q1.z, s);
                s = ggml_cuda_dp4a(hi[3], q1.w, s);
                r[g] = dk*(qds.x*s - 8.0f*qds.y);
            }
#pragma unroll
            for (int g = 0; g < G; ++g) {
#pragma unroll
                for (int o = NB/2; o > 0; o >>= 1) {
                    r[g] += __shfl_xor_sync(0xFFFFFFFF, r[g], o, WARP_SIZE);
                }
            }
            if (c == 0) {
                const float mv = maskh ? __half2float(maskh[pos0 + pos_t]) : 0.0f;
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    sP[pos_t][g] = r[g] + mv;
                }
            }
        }
        __syncthreads();

        // online softmax, thread t owns position t of the tile
        float sv[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            sv[g] = sP[t][g];
            const float wm = warp_reduce_max(sv[g]);
            if (lane == 0) {
                s_red[g][warp] = wm;
            }
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float mn = m[g];
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                mn = fmaxf(mn, s_red[g][w]);
            }
            const float sc = expf(m[g] - mn);
            m[g] = mn;
            const float pv = expf(sv[g] - mn);
            lsum[g] = lsum[g]*sc + pv;
            sP[t][g] = pv;
#pragma unroll
            for (int e = 0; e < GQAQ_EPT; ++e) {
                acc[g][e] *= sc;
            }
        }
        __syncthreads();

        // VKQ: thread (dg, pg) takes 4 elements of the rows pg, pg + PG, ...
        const int b   = dg / TPB;
        const int qd4 = dg % TPB;
        const char * vptr = Vh + int64_t(pos0 + pg)*nb21 + b*18;
        const int64_t vstep = int64_t(PG)*nb21;
#pragma unroll 8
        for (int j = pg; j < TILE; j += PG, vptr += vstep) {
            const float dv = __half2float(__ushort_as_half(gqaq_ld16(vptr)));
            const int w = int(gqaq_ld16(vptr + 2 + GQAQ_BPT*qd4));

            const int wl = w & 0x0F0F0F0F;
            const int wh = (w >> 4) & 0x0F0F0F0F;
            float f[GQAQ_EPT];
#pragma unroll
            for (int i = 0; i < GQAQ_BPT; ++i) {
                f[i]            = gqaq_nibble(wl, i);
                f[GQAQ_BPT + i] = gqaq_nibble(wh, i);
            }

            const float4 p0 = *(const float4 *) &sP[j][0];
            const float4 p1 = *(const float4 *) &sP[j][4];
            const float pw[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pd = pw[g]*dv;
#pragma unroll
                for (int e = 0; e < GQAQ_EPT; ++e) {
                    acc[g][e] += pd*f[e];
                }
            }
        }
        __syncthreads();
    }

    // denominators of the whole block
    float l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float wl = warp_reduce_sum(lsum[g]);
        if (lane == 0) {
            s_red[g][warp] = wl;
        }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < G; ++g) {
        l[g] = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            l[g] += s_red[g][w];
        }
    }

    // sum the partial results of the PG position groups
    __shared__ float red[G][D];
    {
        const int b   = dg / TPB;
        const int qd4 = dg % TPB;
        for (int k = 0; k < PG; ++k) {
            if (pg == k) {
#pragma unroll
                for (int g = 0; g < G; ++g) {
#pragma unroll
                    for (int e = 0; e < GQAQ_EPT; ++e) {
                        const int idx = b*32 + (e < GQAQ_BPT ? GQAQ_BPT*qd4 + e : 16 + GQAQ_BPT*qd4 + (e - GQAQ_BPT));
                        if (k == 0) {
                            red[g][idx] = acc[g][e];
                        } else {
                            red[g][idx] += acc[g][e];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = t; i < D; i += NT) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int64_t row = int64_t(seq)*Hq + hk*G + g;
            if (pb == 1) {
                dst[row*D + i] = red[g][i] / l[g];
            } else {
                dst[(row*pb + pbi)*D + i] = red[g][i];
            }
        }
    }
    if (pb != 1 && t < G) {
        // thread t publishes the statistics of head hk*G + t
        float mt = m[0];
        float lt = l[0];
#pragma unroll
        for (int g = 1; g < G; ++g) {
            if (t == g) {
                mt = m[g];
                lt = l[g];
            }
        }
        const int64_t row = int64_t(seq)*Hq + hk*G + t;
        dst_meta[row*pb + pbi] = make_float2(mt, lt);
    }
}

static bool gqaq_enabled() {
    // GGML_CUDA_FATTN_GQAQ=0 selects the generic vector kernel
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FATTN_GQAQ");
        return !(s && s[0] == '0');
    }();
    return enabled;
}

bool ggml_cuda_fattn_gqa_q_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!gqaq_enabled()) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_DP4A || cc >= GGML_CUDA_CC_VOLTA) {
        return false;
    }
    if (K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || Q->type != GGML_TYPE_F32) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || sinks != nullptr) {
        return false;
    }

    const int64_t D = Q->ne[0];
    if ((D != 64 && D != 128 && D != 256) || V->ne[0] != D || K->ne[0] != D) {
        return false;
    }
    if (Q->ne[1] != 1 || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int64_t G = Q->ne[2] / K->ne[2];
    if (G != 2 && G != 4 && G != 6 && G != 8) {
        return false;
    }
    if (K->ne[1] % GQAQ_TILE != 0 || K->ne[1] != V->ne[1] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (K->nb[0] != sizeof(block_q4_0) || V->nb[0] != sizeof(block_q4_0) || K->nb[1] % 2 != 0 || V->nb[1] % 2 != 0) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->nb[0] != sizeof(half))) {
        return false;
    }
    return true;
}

template <int D, int G>
static void gqaq_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int Hk     = K->ne[2];
    const int ne3    = Q->ne[3];
    const int ntiles = K->ne[1] / GQAQ_TILE;

    // one wave of blocks, the KV range of a KV head is split between the blocks
    int occ = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, flash_attn_gqa_q4_0<D, G>, GQAQ_NT, 0));
    occ = std::max(occ, 1);
    const int pb = std::max(1, std::min(ntiles, (nsm*occ + Hk*ne3 - 1) / (Hk*ne3)));

    ggml_cuda_pool_alloc<float>  dst_tmp(ctx.pool());
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(ctx.pool());
    if (pb > 1) {
        dst_tmp.alloc(size_t(pb)*ggml_nelements(dst));
        dst_tmp_meta.alloc(size_t(pb)*ggml_nrows(dst));
    }

    const dim3 blocks_num(pb, Hk, ne3);
    const dim3 block_dim(GQAQ_NT, 1, 1);
    flash_attn_gqa_q4_0<D, G><<<blocks_num, block_dim, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, mask ? (const char *) mask->data : nullptr,
        pb > 1 ? dst_tmp.ptr : (float *) dst->data, dst_tmp_meta.ptr, scale,
        Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3],
        V->nb[1], V->nb[2], V->nb[3], mask ? mask->nb[3] : 0, mask ? (int) mask->ne[3] : 1,
        ntiles);
    CUDA_CHECK(cudaGetLastError());

    if (pb > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = pb*sizeof(float2);
        const ggml_cuda_kernel_launch_params launch_params =
            ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<D>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) dst->data, pb);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int D>
static void gqaq_launch_D(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int G = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    switch (G) {
        case 2: gqaq_launch<D, 2>(ctx, dst); break;
        case 4: gqaq_launch<D, 4>(ctx, dst); break;
        case 6: gqaq_launch<D, 6>(ctx, dst); break;
        case 8: gqaq_launch<D, 8>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_fattn_gqa_q(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[0]) {
        case  64: gqaq_launch_D< 64>(ctx, dst); break;
        case 128: gqaq_launch_D<128>(ctx, dst); break;
        case 256: gqaq_launch_D<256>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}
