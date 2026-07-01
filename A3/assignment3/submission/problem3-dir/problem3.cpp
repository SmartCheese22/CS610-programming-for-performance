#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <x86intrin.h>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint32_t NX = 128;
const uint32_t NY = 128;
const uint32_t NZ = 128;
const uint64_t TOTAL_SIZE = (NX * NY * NZ);

const uint32_t N_ITERATIONS = 100;
const uint64_t INITIAL_VAL = 1000000;
static constexpr size_t ALIGN = 32; // for AVX2 aligned loads/stores

void scalar_3d_gradient(const uint64_t *A, uint64_t *B)
{
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i)
    {
        for (int j = 0; j < NY; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                uint64_t base_idx = (i * NY * NZ) + j * NZ + k;
                // A[i+1, j, k]
                int A_right = A[base_idx + stride_i];
                // A[i-1, j, k]
                int A_left = A[base_idx - stride_i];
                B[base_idx] = A_right - A_left;
            }
        }
    }
}

void sse4_3d_gradient(const uint64_t *__restrict__ A, uint64_t *__restrict__ B)
{
    A = static_cast<const uint64_t *>(__builtin_assume_aligned(A, ALIGN));
    B = static_cast<uint64_t *>(__builtin_assume_aligned(B, ALIGN));
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i)
    {
        const uint64_t ip = i * stride_i;
        for (int j = 0; j < NY; ++j)
        {
            const uint64_t base = ip + j * NZ;

            uint32_t k = 0;
            // sse regs can hold 2 uint64_t values
            for (; k + 1 < NZ; k += 2)
            {
                const uint64_t idx = base + k;
                __m128i vecA_right = _mm_load_si128(reinterpret_cast<const __m128i *>(&A[idx + stride_i]));
                __m128i vecA_left = _mm_load_si128(reinterpret_cast<const __m128i *>(&A[idx - stride_i]));
                __m128i vecD = _mm_sub_epi64(vecA_right, vecA_left);
                _mm_store_si128(reinterpret_cast<__m128i *>(&B[idx]), vecD);
            }

            // remaining elements
            for (; k < NZ; ++k)
            {
                const uint64_t idx = base + k;
                B[idx] = A[idx + stride_i] - A[idx - stride_i];
            }
        }
    }
}

void avx2_3d_gradient(const uint64_t *__restrict__ A, uint64_t *__restrict__ B)
{
    A = static_cast<const uint64_t *>(__builtin_assume_aligned(A, ALIGN));
    B = static_cast<uint64_t *>(__builtin_assume_aligned(B, ALIGN));
    const uint64_t stride_i = (NY * NZ);
    for (int i = 1; i < NX - 1; ++i)
    {
        const uint64_t ip = i * stride_i;
        for (int j = 0; j < NY; ++j)
        {
            const uint64_t base = ip + j * NZ;

            uint32_t k = 0;
            // avx2 regs can hold 4 uint64_t values
            for (; k + 3 < NZ; k += 4)
            {
                const uint64_t idx = base + k;
                __m256i vecA_right = _mm256_load_si256(reinterpret_cast<const __m256i *>(&A[idx + stride_i]));
                __m256i vecA_left = _mm256_load_si256(reinterpret_cast<const __m256i *>(&A[idx - stride_i]));
                __m256i vecD = _mm256_sub_epi64(vecA_right, vecA_left);
                _mm256_store_si256(reinterpret_cast<__m256i *>(&B[idx]), vecD);
            }

            // remaining elements
            for (; k < NZ; ++k)
            {
                const uint64_t idx = base + k;
                B[idx] = A[idx + stride_i] - A[idx - stride_i];
            }
        }
    }
}

long compute_checksum(const uint64_t *grid)
{
    uint64_t sum = 0;
    for (int i = 1; i < (NX - 1); i++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int k = 0; k < NZ; k++)
            {
                sum += grid[i * NY * NZ + j * NZ + k];
            }
        }
    }
    return sum;
}

int main()
{
    auto *i_grid = static_cast<uint64_t *>(aligned_alloc(ALIGN, TOTAL_SIZE * sizeof(uint64_t)));
    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int k = 0; k < NZ; k++)
            {
                i_grid[i * NY * NZ + j * NZ + k] = (INITIAL_VAL + i +
                                                    2 * j + 3 * k);
            }
        }
    }

    auto *o_grid1 = new uint64_t[TOTAL_SIZE];
    std::fill_n(o_grid1, TOTAL_SIZE, 0);

    auto start = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
    {
        scalar_3d_gradient(i_grid, o_grid1);
    }
    auto end = HR::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Scalar kernel time (ms): " << duration << "\n";

    // Compare checksum with vector versions
    uint64_t scalar_checksum = compute_checksum(o_grid1);
    cout << "Checksum: " << scalar_checksum << "\n";

    // Assert the checksum for vectors variants
    auto *o_grid2 = static_cast<uint64_t *>(aligned_alloc(ALIGN, TOTAL_SIZE * sizeof(uint64_t)));
    std::fill_n(o_grid2, TOTAL_SIZE, 0);

    auto start_sse4 = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
    {
        sse4_3d_gradient(i_grid, o_grid2);
    }
    auto end_sse4 = HR::now();
    auto duration_sse4 = duration_cast<milliseconds>(end_sse4 - start_sse4).count();
    cout << "SSE4 kernel time (ms): " << duration_sse4 << "\n";

    uint64_t sse4_checksum = compute_checksum(o_grid2);
    cout << "Checksum: " << sse4_checksum << "\n";

    auto *o_grid3 = static_cast<uint64_t *>(aligned_alloc(ALIGN, TOTAL_SIZE * sizeof(uint64_t)));
    std::fill_n(o_grid3, TOTAL_SIZE, 0);
    auto start_avx2 = HR::now();
    for (int iter = 0; iter < N_ITERATIONS; ++iter)
    {
        avx2_3d_gradient(i_grid, o_grid3);
    }
    auto end_avx2 = HR::now();
    auto duration_avx2 = duration_cast<milliseconds>(end_avx2 - start_avx2).count();
    cout << "AVX2 kernel time (ms): " << duration_avx2 << "\n";
    uint64_t avx2_checksum = compute_checksum(o_grid3);
    cout << "Checksum: " << avx2_checksum << "\n";

    delete[] i_grid;
    delete[] o_grid1;
    delete[] o_grid2;
    delete[] o_grid3;

    return EXIT_SUCCESS;
}
