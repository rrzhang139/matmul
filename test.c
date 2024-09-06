#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <arm_neon.h>
#include <pthread.h>

#define N 1024
#define BLOCK_SIZE 16
#define NUM_THREADS 12

float A[N][N];
float B[N][N];
float C[N][N] = {0};
float C_solution[N][N] = {0};
float B_transposed[N][N];

struct ThreadArg {
 int start;
 int end;
} ThreadArg;

// Function to generate the known solution
void generate_solution() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_solution[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C_solution[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


// Function to compare the computed result with the known solution
int verify_result() {
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float diff = fabsf(C[i][j] - C_solution[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    
    // Allow for some small floating-point error
    const float epsilon = 1e-4;
    if (max_diff < epsilon) {
        printf("Verification PASSED. Max difference: %e\n", max_diff);
        return 1;
    } else {
        printf("Verification FAILED. Max difference: %e\n", max_diff);
        return 0;
    }
}


uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1e9 + (uint64_t)start.tv_nsec;
}

void transpose(float src[N][N], float dst[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dst[j][i] = src[i][j];
        }
    }
}

void mat_mul() {
   for(int bx = 0; bx < N; bx+=BLOCK_SIZE) {
    for(int by = 0; by < N; by+=BLOCK_SIZE) {
      
      float bc[BLOCK_SIZE][BLOCK_SIZE] = {0};
      for (int bk = 0; bk < N; bk+=BLOCK_SIZE) {
        for(int x = 0; x < BLOCK_SIZE; x++) {
          for(int y = 0; y < BLOCK_SIZE; y++) {
            float accum = 0;
            for(int k = 0; k < BLOCK_SIZE; k++) {
              accum += A[bx+x][bk+k] * B[bk+k][by+y]; //B_transposed[by+y][bk+k];
              // B_transposed[by+y][bk+k] is faster than B[bk+k][by+y]
              // because inner loop is over k which are columns of B_transpose. Increases by 0.5 GFLops/s
            }
            bc[x][y] += accum; //Storing in register faster. 0.3 more GFlops/s
            // C[bx + x][by + y] += accum;
          }
        }
      }
      // Copy block cache to C
      for (int i = 0; i < BLOCK_SIZE; i++) {
       for (int j = 0; j < BLOCK_SIZE; j++) {
           C[bx + i][by + j] = bc[i][j];
       }
      }
    }
  }
}

void mat_mul_neon() {
    for (int bx = 0; bx < N; bx += BLOCK_SIZE) {
        for (int by = 0; by < N; by += BLOCK_SIZE) {
            float bc[BLOCK_SIZE][BLOCK_SIZE] = {0};
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    for (int y = 0; y < BLOCK_SIZE; y += 4) {
                        float32x4_t sum = vdupq_n_f32(0.0f);
                        for (int k = 0; k < BLOCK_SIZE; k++) {
                            float32x4_t a = vld1q_f32(&A[bx + x][bk + k]);
                            float32x4_t b = vld1q_f32(&B_transposed[by + y][bk + k]);
                            sum = vmlaq_f32(sum, a, b);
                        }
                        vst1q_f32(&bc[x][y], sum);
                    }
                }
            }
            for (int i = 0; i < BLOCK_SIZE; i++) {
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    C[bx + i][by + j] = bc[i][j];
                }
            }
        }
    }
}

void *mat_mul_thread(void *args) {
 struct ThreadArg *arg = (struct ThreadArg *) args;
 printf("Thread started: %d %d\n", arg->start, arg->end);
 for (int bx = arg->start; bx < arg->end; bx += BLOCK_SIZE) {
    for (int by = 0; by < N; by += BLOCK_SIZE) {
        float bc[BLOCK_SIZE][BLOCK_SIZE] = {0};
        for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
            for (int x = 0; x < BLOCK_SIZE; x++) {
                for (int y = 0; y < BLOCK_SIZE; y++) {
                 float accum = 0;
                 for(int k = 0; k < BLOCK_SIZE; k++) {
                   accum += A[bx+x][bk+k] * B[bk+k][by+y];
                 }
                 bc[x][y] += accum;
                }
            }
        }
        for (int i = 0; i < BLOCK_SIZE; i++) {
          for (int j = 0; j < BLOCK_SIZE; j++) {
              C[bx + i][by + j] = bc[i][j];
          }
        }
    }
  }
  return NULL;
}

int main() {
  assert(N%BLOCK_SIZE == 0);
  printf("Block size: %d\n", BLOCK_SIZE);

  srand(0);
  // Initialize a matrix
  for(int i = 0; i < N; i++) {
   for(int j = 0; j < N; j++) {
    A[i][j] = (float)(rand() % 100);
    B[i][j] = (float)(rand() % 100);
   }
  }

  // Generate the known solution
  generate_solution();

  // Transpose B for better cache locality? 
  transpose(B, B_transposed);
   
  // Main loop
  uint64_t start = nanos();
  // mat_mul();
  // mat_mul_neon();
  pthread_t threads[NUM_THREADS];
  struct ThreadArg arg[NUM_THREADS];

  int blocks_per_thread = (N / BLOCK_SIZE) / NUM_THREADS;
  for(int i = 0; i < NUM_THREADS; i++) {
   arg[i].start = i * blocks_per_thread * BLOCK_SIZE;
   arg[i].end = (i+1) * blocks_per_thread * BLOCK_SIZE;
   // Ensure the last thread covers any remaining blocks
   if (i == NUM_THREADS - 1) {
       arg[i].end = N;
   }
   pthread_create(&threads[i], NULL, mat_mul_thread, &arg[i]);
  }
  for(int i=0; i<NUM_THREADS; i++) 
    pthread_join(threads[i], NULL); // Wait for all threads to finish

  uint64_t end = nanos();

  assert(verify_result());

  printf("Time: %f\n", (end - start) / 1e9);
  float flops = N*N*2.0*N;
  double gflops = flops / 1e9; 
  double seconds = (end - start) / 1e9;
  printf("%f GFLOPs/S\n", gflops/seconds);

  return 0;
}
