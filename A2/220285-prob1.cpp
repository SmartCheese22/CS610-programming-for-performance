#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <papi.h>
#include <chrono>
#include <algorithm>
#include <cstring>

using std::cerr;
using std::cout;
using std::endl;
using std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t;

#define INP_H (1 << 6)  // 64
#define INP_W (1 << 6)  // 64
#define INP_D (1 << 6)  // 64
#define FIL_H (3)
#define FIL_W (3)
#define FIL_D (3)

/** Cross-correlation without padding */
double cc_3d_no_padding(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth) {
  auto start = std::chrono::high_resolution_clock::now();
  
  for (uint64_t i = 0; i < outputHeight; i++) {
    for (uint64_t j = 0; j < outputWidth; j++) {
      for (uint64_t k = 0; k < outputDepth; k++) {
        uint64_t sum = 0;
        for (uint64_t ki = 0; ki < FIL_H; ki++) {
          for (uint64_t kj = 0; kj < FIL_W; kj++) {
            for (uint64_t kk = 0; kk < FIL_D; kk++) {
              sum += input[(i + ki) * INP_W * INP_D + (j + kj) * INP_D + (k + kk)] * kernel[ki][kj][kk];
            }
          }
        }
        result[i * outputWidth * outputDepth + j * outputDepth + k] = sum; // Fixed: use = instead of +=
      }
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  return duration.count() / 1000.0;
}

double cc_3d_blocked(const uint64_t* input,
                   const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                   const uint64_t outputHeight, const uint64_t outputWidth, 
                   const uint64_t outputDepth, uint64_t BLKI, uint64_t BLKJ, uint64_t BLKK) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t ii = 0; ii < outputHeight; ii += BLKI) {
        for (uint64_t jj = 0; jj < outputWidth; jj += BLKJ) {
            for (uint64_t kk = 0; kk < outputDepth; kk += BLKK) {
                uint64_t i_end = std::min(ii + BLKI, outputHeight);
                uint64_t j_end = std::min(jj + BLKJ, outputWidth);
                uint64_t k_end = std::min(kk + BLKK, outputDepth);
                
                for(uint64_t i = ii; i < i_end; i++) {
                  for(uint64_t j = jj; j < j_end; j++) {
                    for(uint64_t k = kk; k < k_end; k++) {
                      uint64_t sum = 0;
                      for(uint64_t u = 0; u < FIL_H; u++) {
                        for(uint64_t v = 0; v < FIL_W; v++) {
                          for(uint64_t w = 0; w < FIL_D; w++) {
                            sum += input[(i + u) * INP_W * INP_D + (j + v) * INP_D + (k + w)] * kernel[u][v][w];
                          }
                        }
                      }
                      result[i * outputWidth * outputDepth + j * outputDepth + k] = sum; // Fixed: use = instead of +=
                    }
                  }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;
}

int measureCacheMisses(const uint64_t* input,
                      const uint64_t (*kernel)[FIL_W][FIL_D], uint64_t* result,
                      const uint64_t outputHeight, const uint64_t outputWidth,
                      const uint64_t outputDepth, bool use_blocked, 
                      uint64_t block_i = 0, uint64_t block_j = 0, uint64_t block_k = 0) {

  cout << "Measuring Cache Misses for " << (use_blocked ? "Blocked" : "Naive") << " Convolution..." << endl;
  
  int retval = PAPI_library_init(PAPI_VER_CURRENT);
  if (retval != PAPI_VER_CURRENT && retval > 0) {
    cerr << "PAPI library version mismatch: " << retval << " != " << PAPI_VER_CURRENT << "\n";
    return -1;
  } else if (retval < 0) {
    cerr << "PAPI library initialization error: " << retval << "\n";
    return -1;
  }

  int eventSet = PAPI_NULL;
  retval = PAPI_create_eventset(&eventSet);
  if(retval != PAPI_OK){
    cerr << "Error at PAPI_create_eventset()" << endl;
    return -1;
  }

  // Add events
  if (PAPI_add_event(eventSet, PAPI_L1_DCM) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_L1_DCM!\n";
    return -1;
  }
  if (PAPI_add_event(eventSet, PAPI_L2_DCM) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_L2_DCM!\n";
    return -1;
  }
  if (PAPI_add_event(eventSet, PAPI_L3_DCM) != PAPI_OK) {
    cout << "Error in PAPI_add_event PAPI_L3_DCM!\n";
    return -1;
  }

  retval = PAPI_start(eventSet);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_start()" << endl;
    return -1;
  }
  
  // Create a temporary result array for measurement
  uint64_t* temp_result = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
  
  if (use_blocked) {
      cc_3d_blocked(input, kernel, temp_result, outputHeight, outputWidth, outputDepth, block_i, block_j, block_k);
  } else {
      cc_3d_no_padding(input, kernel, temp_result, outputHeight, outputWidth, outputDepth);
  }
  
  long long int values[3];  // Fixed: allocate space for 3 events
  retval = PAPI_stop(eventSet, values);
  if (PAPI_OK != retval) {
    cerr << "Error at PAPI_stop()" << endl;
    delete[] temp_result;
    return -1;
  }
  
  PAPI_cleanup_eventset(eventSet);
  PAPI_destroy_eventset(&eventSet);
  PAPI_shutdown();

  std::cout << (use_blocked ? "Blocked" : "Naive") << " Convolution Cache Misses:" << std::endl;
  std::cout << "L1 Data Cache Misses: " << values[0] << std::endl;
  std::cout << "L2 Data Cache Misses: " << values[1] << std::endl;
  std::cout << "L3 Data Cache Misses: " << values[2] << std::endl;
  std::cout << std::endl;
  
  delete[] temp_result;
  return 0;
}

void autoTune(const uint64_t* input,  // Made const for safety
              const uint64_t (*filter)[FIL_W][FIL_D], 
              const uint64_t outputHeight, const uint64_t outputWidth,
              const uint64_t outputDepth, uint64_t& best_BLKI, uint64_t& best_BLKJ, uint64_t& best_BLKK) {
  std::cout << "Auto-tuning block sizes..." << std::endl;
        
  double best_time = 1e9;
  uint64_t block_sizes[] = {4, 8, 16, 32, 64};  // Reduced range for faster tuning
  
  // Create temporary result array for tuning
  uint64_t* temp_result = new uint64_t[outputHeight * outputWidth * outputDepth];
  
  for(auto BLKI: block_sizes) {
      for(auto BLKJ: block_sizes) {
          for(auto BLKK: block_sizes) {
            if(BLKI > outputHeight || BLKJ > outputWidth || BLKK > outputDepth) continue;
              
              double total_time = 0.0;
              // Run 5 times for averaging
              for(int run = 0; run < 5; run++) {
                  total_time += cc_3d_blocked(input, filter, temp_result, outputHeight, outputWidth, outputDepth, BLKI, BLKJ, BLKK);
              }
              double avg_time = total_time / 5.0;
              
              std::cout << "Block size (" << BLKI << "x" << BLKJ << "x" << BLKK
                              << "): " << avg_time << " ms" << std::endl;
              
              if(avg_time < best_time) {
                best_time = avg_time;
                best_BLKI = BLKI;
                best_BLKJ = BLKJ;
                best_BLKK = BLKK;
              }
          }
      }
  }
  
  std::cout << "\nBest block size: (" << best_BLKI << "x" << best_BLKJ << "x" << best_BLKK
                  << ") with time: " << best_time << " ms" << std::endl;
  
  delete[] temp_result;
}

void printOutput(const uint64_t* result, uint64_t outputHeight, uint64_t outputWidth, uint64_t outputDepth) {
    cout << "Output (first 3x3x3 elements):\n";  // Print only a subset for readability
    uint64_t max_print = std::min((uint64_t)3, outputHeight);
    for (uint64_t i = 0; i < max_print; i++) {
        for (uint64_t j = 0; j < std::min((uint64_t)3, outputWidth); j++) {
            for (uint64_t k = 0; k < std::min((uint64_t)3, outputDepth); k++) {
                cout << result[i * outputWidth * outputDepth + j * outputDepth + k] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
}

bool checkOutput(const uint64_t* result1, const uint64_t* result2, uint64_t outputHeight, uint64_t outputWidth, uint64_t outputDepth) {
    for (uint64_t i = 0; i < outputHeight; i++) {
        for (uint64_t j = 0; j < outputWidth; j++) {
            for (uint64_t k = 0; k < outputDepth; k++) {
                uint64_t idx = i * outputWidth * outputDepth + j * outputDepth + k;
                if(result1[idx] != result2[idx]) {
                    std::cout << "Mismatch at (" << i << "," << j << "," << k << "): " 
                              << result1[idx] << " != " << result2[idx] << std::endl;
                    return false;
                }
            }
        }
    }
    std::cout << "✓ Output verification passed - both implementations produce identical results!" << std::endl;
    return true;
}

int main() {
  std::cout << "3D Convolution Implementation (N=" << INP_H << ", K=" << FIL_H << ")" << std::endl;
  std::cout << "Input size: " << INP_H << "x" << INP_W << "x" << INP_D << std::endl;
  std::cout << "Filter size: " << FIL_H << "x" << FIL_W << "x" << FIL_D << std::endl;
  
  uint64_t outputHeight = INP_H - FIL_H + 1;  // 62
  uint64_t outputWidth = INP_W - FIL_W + 1;   // 62
  uint64_t outputDepth = INP_D - FIL_D + 1;   // 62
  
  std::cout << "Output size: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl << std::endl;

  // Initialize input array
  uint64_t* input = new uint64_t[INP_H * INP_W * INP_D];
  std::fill_n(input, INP_H * INP_W * INP_D, 1);

  // Initialize filter
  uint64_t filter[FIL_H][FIL_W][FIL_D] = {{{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}},
                                          {{2, 1, 3}, {2, 1, 3}, {2, 1, 3}}};

  // Allocate result arrays
  auto* result1 = new uint64_t[outputHeight * outputWidth * outputDepth]{0};
  auto* result2 = new uint64_t[outputHeight * outputWidth * outputDepth]{0};

  // Run naive implementation
  std::cout << "=== Naive Implementation ===" << std::endl;
  double naive_time = cc_3d_no_padding(input, filter, result1, outputHeight, outputWidth, outputDepth);
  std::cout << "Naive convolution time: " << naive_time << " ms" << std::endl;

  // Auto-tune to find best block sizes
  std::cout << "\n=== Auto-tuning Blocked Implementation ===" << std::endl;
  uint64_t best_BLKI = 0, best_BLKJ = 0, best_BLKK = 0;
  autoTune(input, filter, outputHeight, outputWidth, outputDepth, best_BLKI, best_BLKJ, best_BLKK);

  // Run blocked implementation with best parameters
  std::cout << "\n=== Blocked Implementation ===" << std::endl;
  double blocked_time = cc_3d_blocked(input, filter, result2, outputHeight, outputWidth, outputDepth, best_BLKI, best_BLKJ, best_BLKK);
  std::cout << "Blocked convolution time: " << blocked_time << " ms" << std::endl;

  // Verify correctness
  std::cout << "\n=== Correctness Check ===" << std::endl;
  checkOutput(result1, result2, outputHeight, outputWidth, outputDepth);

  // Print sample output
  std::cout << "\n=== Sample Output ===" << std::endl;
  printOutput(result1, outputHeight, outputWidth, outputDepth);

  // Detailed benchmarking
  std::cout << "\n=== Detailed Benchmarking ===" << std::endl;
  double naive_total = 0.0;
  std::cout << "Running naive convolution 5 times:" << std::endl;
  for(int i = 0; i < 5; i++) {
      double time = cc_3d_no_padding(input, filter, result1, outputHeight, outputWidth, outputDepth);
      std::cout << "Run " << (i+1) << ": " << time << " ms" << std::endl;
      naive_total += time;
  }
  double naive_avg = naive_total / 5.0;
  std::cout << "Average naive time: " << naive_avg << " ms" << std::endl;

  double blocked_total = 0.0;
  std::cout << "\nRunning blocked convolution 5 times with best block size:" << std::endl;
  for(int i = 0; i < 5; i++) {
      double time = cc_3d_blocked(input, filter, result2, outputHeight, outputWidth, outputDepth, best_BLKI, best_BLKJ, best_BLKK);
      std::cout << "Run " << (i+1) << ": " << time << " ms" << std::endl;
      blocked_total += time;
  }
  double blocked_avg = blocked_total / 5.0;
  std::cout << "Average blocked time: " << blocked_avg << " ms" << std::endl;
  
  std::cout << "\nPerformance Summary:" << std::endl;
  std::cout << "Speedup: " << (naive_avg / blocked_avg) << "x" << std::endl;
  std::cout << "Best block size: " << best_BLKI << "x" << best_BLKJ << "x" << best_BLKK << std::endl;

  // PAPI measurements
  std::cout << "\n=== Cache Miss Analysis ===" << std::endl;
  measureCacheMisses(input, filter, result1, outputHeight, outputWidth, outputDepth, false); // Naive
  measureCacheMisses(input, filter, result2, outputHeight, outputWidth, outputDepth, true, best_BLKI, best_BLKJ, best_BLKK); // Blocked

  // Cleanup
  delete[] input;
  delete[] result1;
  delete[] result2;

  return EXIT_SUCCESS;
}