#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <omp.h>

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const uint64_t TIMESTEPS = 100;

const double W_OWN = (1.0 / 7.0);
const double W_NEIGHBORS = (1.0 / 7.0);

const uint64_t NX = 66; // 64 interior points + 2 boundary points
const uint64_t NY = 66;
const uint64_t NZ = 66;
const uint64_t TOTAL_SIZE = NX * NY * NZ;

const static double EPSILON = std::numeric_limits<double>::epsilon();

// base version
void stencil_3d_7pt(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];

        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_simd(const double* curr, double* next) {
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[(i - 1) * NY * NZ + j * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j + 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + (j - 1) * NZ + k];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k + 1)];
        neighbors_sum += curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_simd_indexed(const double* curr, double* next) {
  int i_step = NY * NZ;
  int j_step = NZ;
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int i_prev = (i - 1) * i_step;
      int i_next = (i + 1) * i_step;
      int ii = i * i_step;
      int j_prev = (j - 1) * j_step;
      int j_next = (j + 1) * j_step;
      int jj = j * j_step;
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_next + jj + k];
        neighbors_sum += curr[i_prev + jj + k];
        neighbors_sum += curr[ii + j_next + k];
        neighbors_sum += curr[ii + j_prev + k];
        neighbors_sum += curr[ii + jj + (k + 1)];
        neighbors_sum += curr[ii + jj + (k - 1)];

        next[ii + jj + k] =W_OWN * curr[ii + jj + k] + W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_omp_ijk(const double* curr, double* next) {
  #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k] + curr[(i - 1) * NY * NZ + j * NZ + k] + curr[i * NY * NZ + (j + 1) * NZ + k] + curr[i * NY * NZ + (j - 1) * NZ + k] +
                           curr[i * NY * NZ + j * NZ + (k + 1)] + curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_omp_sections(const double* curr, double* next) {

  for (int i = 1; i < NX - 1; i += 2) {
    #pragma omp parallel sections
    {
      #pragma omp section
      {
        for (int j = 1; j < NY - 1; j ++) {
          #pragma omp simd
          for (int k = 1; k < NZ - 1; ++k) {
            double neighbors_sum = 0.0;
            neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k] + curr[(i - 1) * NY * NZ + j * NZ + k] + curr[i * NY * NZ + (j + 1) * NZ + k] + curr[i * NY * NZ + (j - 1) * NZ + k] +
                              curr[i * NY * NZ + j * NZ + (k + 1)] + curr[i * NY * NZ + j * NZ + (k - 1)];
            next[i * NY * NZ + j * NZ + k] =
                W_OWN * curr[i * NY * NZ + j * NZ + k] +
                W_NEIGHBORS * neighbors_sum;
          }
        }
      }
       
      #pragma omp section
      {
          for (int j = 1; j < NY - 1; j ++) {
          #pragma omp simd
          for (int k = 1; k < NZ - 1; ++k) {
            double neighbors_sum = 0.0;
            neighbors_sum += curr[(i + 2) * NY * NZ + j * NZ + k] + curr[(i) * NY * NZ + j * NZ + k] + curr[(i + 1) * NY * NZ + (j + 1) * NZ + k] + curr[(i + 1) * NY * NZ + (j - 1) * NZ + k] +
                              curr[(i + 1) * NY * NZ + j * NZ + (k + 1)] + curr[(i + 1) * NY * NZ + j * NZ + (k - 1)];
            next[(i + 1) * NY * NZ + j * NZ + k] =
                W_OWN * curr[(i + 1) * NY * NZ + j * NZ + k] +
                W_NEIGHBORS * neighbors_sum;
          }
        }
      }
    } //sections
  }
}

void stencil_3d_7pt_omp_i(const double* curr, double* next) {
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    const int i_prev = (i - 1) * NY * NZ;
    const int i_next = (i + 1) * NY * NZ;
    const int ii = i * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const int j_prev = (j - 1) * NZ;
      const int j_next = (j + 1) * NZ;
      const int jj = j * NZ;
      
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 
            curr[i_next + jj + k] +
            curr[i_prev + jj + k] +
            curr[ii + j_next + k] +
            curr[ii + j_prev + k] +
            curr[ii + jj + k + 1] +
            curr[ii + jj + k - 1];
        
        next[ii + jj + k] =
            W_OWN * curr[ii + jj + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_omp_ij(const double* curr, double* next) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[(i + 1) * NY * NZ + j * NZ + k] + curr[(i - 1) * NY * NZ + j * NZ + k] + curr[i * NY * NZ + (j + 1) * NZ + k] + curr[i * NY * NZ + (j - 1) * NZ + k] +
                           curr[i * NY * NZ + j * NZ + (k + 1)] + curr[i * NY * NZ + j * NZ + (k - 1)];
        next[i * NY * NZ + j * NZ + k] =
            W_OWN * curr[i * NY * NZ + j * NZ + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_omp_ij_indexed(const double* curr, double* next) {
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < NX - 1; ++i) {
    for (int j = 1; j < NY - 1; ++j) {
      int ii = i * NY * NZ;
      int i_prev = (i - 1) * NY * NZ;
      int i_next = (i + 1) * NY * NZ;
      int jn = (j + 1) * NZ;
      int jp = (j - 1) * NZ;
      int jj = j * NZ;
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 0.0;
        neighbors_sum += curr[i_next + jj + k] + curr[i_prev + jj + k] + curr[ii + jn + k] + curr[ii + jp + k] +
                         curr[ii + jj + (k + 1)] + curr[ii + jj + (k - 1)];
        next[ii + jj + k] =
            W_OWN * curr[ii + jj + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

void stencil_3d_7pt_omp_dynamic(const double* curr, double* next) {
  #pragma omp parallel for schedule(dynamic, 4)
  for (int i = 1; i < NX - 1; ++i) {
    const int i_prev = (i - 1) * NY * NZ;
    const int i_next = (i + 1) * NY * NZ;
    const int ii = i * NY * NZ;
    
    for (int j = 1; j < NY - 1; ++j) {
      const int j_prev = (j - 1) * NZ;
      const int j_next = (j + 1) * NZ;
      const int jj = j * NZ;
      
      #pragma omp simd
      for (int k = 1; k < NZ - 1; ++k) {
        double neighbors_sum = 
            curr[i_next + jj + k] +
            curr[i_prev + jj + k] +
            curr[ii + j_next + k] +
            curr[ii + j_prev + k] +
            curr[ii + jj + k + 1] +
            curr[ii + jj + k - 1];
        
        next[ii + jj + k] =
            W_OWN * curr[ii + jj + k] +
            W_NEIGHBORS * neighbors_sum;
      }
    }
  }
}

#ifndef TILE_I
#define TILE_I 2
#endif
#ifndef TILE_J
#define TILE_J 8
#endif

void stencil_3d_7pt_omp_tiled(const double* __restrict__ curr,
                                 double* __restrict__ next) {
  const size_t plane = NY * NZ;
  const size_t row   = NZ;

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ii = 1; ii < (int)NX - 1; ii += TILE_I) {
    for (int jj = 1; jj < (int)NY - 1; jj += TILE_J) {
      const int i_end = std::min((int)NX - 1, ii + TILE_I);
      const int j_end = std::min((int)NY - 1, jj + TILE_J);

      for (int i = ii; i < i_end; ++i) {
        const size_t ip   = (size_t)i * plane;
        const size_t ip_p = ip + plane, ip_m = ip - plane;
        for (int j = jj; j < j_end; ++j) {
          const size_t jp   = (size_t)j * row;
          const size_t jp_p = jp + row,   jp_m = jp - row;

          #pragma omp simd
          for (int k = 1; k < (int)NZ - 1; ++k) {
            const size_t idx = ip + jp + (size_t)k;
            const double c = curr[idx];
            const double s =
              curr[ip_p + jp + (size_t)k] + curr[ip_m + jp + (size_t)k] +
              curr[ip + jp_p + (size_t)k] + curr[ip + jp_m + (size_t)k] +
              curr[idx + 1] + curr[idx - 1];
            next[idx] = W_OWN * c + W_NEIGHBORS * s;
          }
        }
      }
    }
  }
}

void test_kernel(void (*stencil_kernel)(const double*, double*), const char* kernel_name) {
  auto* grid1 = new double[TOTAL_SIZE];
  std::fill_n(grid1, TOTAL_SIZE, 0.0);
  grid1[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  auto* grid2 = new double[TOTAL_SIZE];
  std::fill_n(grid2, TOTAL_SIZE, 0.0);
  grid2[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)] = 100.0;

  double* current_grid = grid1;
  double* next_grid = grid2;

  auto start = HR::now();
  for (int t = 0; t < TIMESTEPS; t++) {
    stencil_kernel(current_grid, next_grid);
    std::swap(current_grid, next_grid);
  }
  auto end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << kernel_name << " time: " << duration << " ms" << endl;

  double final = current_grid[(NX / 2) * NY * NZ + (NY / 2) * NZ + (NZ / 2)];
  cout << "Final value at center: " << final << "\n";
  double total_sum = 0.0;
  for (size_t i = 0; i < TOTAL_SIZE; i++) {
    total_sum += current_grid[i];
  }
  cout << "Total sum : " << total_sum << "\n";

  delete[] grid1;
  delete[] grid2;
}

int main() {
  test_kernel(stencil_3d_7pt, "Base kernel");
  test_kernel(stencil_3d_7pt_simd, "stencil_3d_7pt_simd");
  test_kernel(stencil_3d_7pt_simd_indexed, "stencil_3d_7pt_simd_indexed");
  test_kernel(stencil_3d_7pt_omp_ijk, "stencil_3d_7pt_omp_ijk");
  test_kernel(stencil_3d_7pt_omp_sections, "stencil_3d_7pt_omp_sections");
  test_kernel(stencil_3d_7pt_omp_i, "stencil_3d_7pt_omp_i");
  test_kernel(stencil_3d_7pt_omp_ij, "stencil_3d_7pt_omp_ij");
  test_kernel(stencil_3d_7pt_omp_ij_indexed, "stencil_3d_7pt_omp_ij_indexed");
  test_kernel(stencil_3d_7pt_omp_dynamic, "stencil_3d_7pt_omp_dynamic");
  test_kernel(stencil_3d_7pt_omp_tiled, "stencil_3d_7pt_omp_tiled");
  return EXIT_SUCCESS;
}

