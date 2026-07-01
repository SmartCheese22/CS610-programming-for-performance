#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<double>::epsilon())

using std::cerr;
using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)                                                    \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

const uint64_t N = (64);

__global__ void naive_kernel(const double* in, double* out, uint64_t n) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  uint64_t k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  
  if (i < (n - 1) && j < (n - 1) && k < (n - 1)) {
    out[i * n * n + j * n + k] = 
        0.8 * (in[(i - 1) * n * n + j * n + k] + 
               in[(i + 1) * n * n + j * n + k] +
               in[i * n * n + (j - 1) * n + k] + 
               in[i * n * n + (j + 1) * n + k] +
               in[i * n * n + j * n + (k - 1)] + 
               in[i * n * n + j * n + (k + 1)]);
  }
}

__global__ void shmem_kernel(const double* in, double* out, uint64_t n) {
  extern __shared__ double s_data[];
  
  int tx = threadIdx.x + 1;
  int ty = threadIdx.y + 1;
  int tz = threadIdx.z + 1;
  
  int bx = blockDim.x + 2;
  int by = blockDim.y + 2;
  
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  uint64_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  uint64_t k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  
  // Load main data point
  if (i < (n - 1) && j < (n - 1) && k < (n - 1)) {
    s_data[tx * by * bx + ty * bx + tz] = in[i * n * n + j * n + k];
    
    // Load halo regions
    if (threadIdx.x == 0 && i > 1)
      s_data[(tx - 1) * by * bx + ty * bx + tz] = in[(i - 1) * n * n + j * n + k];
    if (threadIdx.x == blockDim.x - 1 && i < n - 2)
      s_data[(tx + 1) * by * bx + ty * bx + tz] = in[(i + 1) * n * n + j * n + k];
    
    if (threadIdx.y == 0 && j > 1)
      s_data[tx * by * bx + (ty - 1) * bx + tz] = in[i * n * n + (j - 1) * n + k];
    if (threadIdx.y == blockDim.y - 1 && j < n - 2)
      s_data[tx * by * bx + (ty + 1) * bx + tz] = in[i * n * n + (j + 1) * n + k];
    
    if (threadIdx.z == 0 && k > 1)
      s_data[tx * by * bx + ty * bx + (tz - 1)] = in[i * n * n + j * n + (k - 1)];
    if (threadIdx.z == blockDim.z - 1 && k < n - 2)
      s_data[tx * by * bx + ty * bx + (tz + 1)] = in[i * n * n + j * n + (k + 1)];
  }
  
  __syncthreads();
  
  if (i < (n - 1) && j < (n - 1) && k < (n - 1)) {
    out[i * n * n + j * n + k] = 
        0.8 * (s_data[(tx - 1) * by * bx + ty * bx + tz] +
               s_data[(tx + 1) * by * bx + ty * bx + tz] +
               s_data[tx * by * bx + (ty - 1) * bx + tz] +
               s_data[tx * by * bx + (ty + 1) * bx + tz] +
               s_data[tx * by * bx + ty * bx + (tz - 1)] +
               s_data[tx * by * bx + ty * bx + (tz + 1)]);
  }
}

__global__ void opt_kernel() {}

__global__ void pinned_kernel() {}

__host__ void stencil(const double* in, double* out) {
  for (uint64_t i = 1; i < (N - 1); i++) {
    for (uint64_t j = 1; j < (N - 1); j++) {
      for (uint64_t k = 1; k < (N - 1); k++) {
        out[i * N * N + j * N + k] =
            0.8 *
            (in[(i - 1) * N * N + j * N + k] + in[(i + 1) * N * N + j * N + k] +
             in[i * N * N + (j - 1) * N + k] + in[i * N * N + (j + 1) * N + k] +
             in[i * N * N + j * N + (k - 1)] + in[i * N * N + j * N + (k + 1)]);
      }
    }
  }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
                           const uint64_t size) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        double this_diff =
            w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print_mat(const double* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        printf("%lf,", A[i * N * N + j * N + k]);
      }
      printf("      ");
    }
    printf("\n");
  }
}

int main() {
  uint64_t NUM_ELEMS = (N * N * N);
  uint64_t SIZE_BYTES = (N * N * N) * sizeof(double);

  auto* h_in = new double[NUM_ELEMS];
  auto* h_out_cpu = new double[NUM_ELEMS];
  auto* h_out_gpu = new double[NUM_ELEMS];

  srand(42);
  for (uint64_t i = 0; i < NUM_ELEMS; i++) {
    h_in[i] = static_cast<double>(rand());
  }
  std::fill_n(h_out_cpu, NUM_ELEMS, 0.0);
  std::fill_n(h_out_gpu, NUM_ELEMS, 0.0);

  auto cpu_start = HR::now();
  stencil(h_in, h_out_cpu);
  auto cpu_end = HR::now();
  auto duration = duration_cast<milliseconds>(cpu_end - cpu_start).count();
  cout << "Stencil time on CPU: " << duration << " ms\n";

  cudaError_t status;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // Allocate device memory
  double *d_in, *d_out;
  cudaCheckError(cudaMalloc(&d_in, SIZE_BYTES));
  cudaCheckError(cudaMalloc(&d_out, SIZE_BYTES));

  // Copy input data to device
  cudaCheckError(cudaMemcpy(d_in, h_in, SIZE_BYTES, cudaMemcpyHostToDevice));

  // Kernel configuration
  dim3 blockDim(8, 8, 8);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,
               (N + blockDim.z - 1) / blockDim.z);

  // Naive kernel
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
  cudaEventRecord(start);
  naive_kernel<<<gridDim, blockDim>>>(d_in, d_out, N);
  cudaCheckError(cudaGetLastError());
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, start, end);
  cout << "Naive kernel time: " << kernel_time << " ms\n";
  
  // Copy result back and check
  cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
  check_result(h_out_cpu, h_out_gpu, N);

  // Shared memory kernel
  size_t shmem_size = (blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z + 2) * sizeof(double);
  cudaCheckError(cudaMemset(d_out, 0, SIZE_BYTES));
  cudaEventRecord(start);
  shmem_kernel<<<gridDim, blockDim, shmem_size>>>(d_in, d_out, N);
  cudaCheckError(cudaGetLastError());
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Shmem kernel time: " << kernel_time << " ms\n";
  
  // Copy result back and check
  cudaCheckError(cudaMemcpy(h_out_gpu, d_out, SIZE_BYTES, cudaMemcpyDeviceToHost));
  check_result(h_out_cpu, h_out_gpu, N);

  // Free memory
  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out_cpu;
  delete[] h_out_gpu;

  cudaEventDestroy(start);
  cudaEventDestroy(end);

  return EXIT_SUCCESS;
}
