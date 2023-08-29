#include <bits/chrono.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

#define ALIGN 64
#define VECTOR_LENGTH_256 8
#define VECTOR_LENGTH_512 16

double dotP256(const float* A, const float* B, int N) {

  std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
  // do vector dot product

  __m256 sumr = _mm256_set1_ps(0.0);
  __m256 sumi = _mm256_set1_ps(0.0);
  const __m256 conj = _mm256_set_ps(-1, 1, -1, 1, -1, 1, -1, 1);

  __m256 *a = (__m256 *)A;
  __m256 *b = (__m256 *)B;

  const int n = (N / VECTOR_LENGTH_256);

  for (int i = 0; i < n; ++i) {
    sumr = _mm256_fmadd_ps(a[i], b[i], sumr);

    __m256 bConj = _mm256_mul_ps(b[i], conj);
    __m256 bPerm = _mm256_permute_ps(bConj, 0b10110001);

    sumi = _mm256_fmadd_ps(a[i], bPerm, sumi);
  }

  float* sr = (float *)&sumr;
  float* si = (float *)&sumi;

  float sumR2 = sr[0] + sr[1] +sr[2] +sr[3] +sr[4] +sr[5] +sr[6] +sr[7];
  float sumI2 = si[0] + si[1] +si[2] +si[3] +si[4] +si[5] +si[6] +si[7];


  std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
  double dt = std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime).count();

  std::cout << "Vectorized method results" << std::endl;
  std::cout << "Result sum: " << sumR2 << " + i" << sumI2 << std::endl;
  std::cout << "Time taken: " << dt << std::endl;

  return dt;
}

double dotP512(const float* A, const float* B, int N) {

  std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
  // do vector dot product

  __m512 sumr = _mm512_set1_ps(0.0);
  __m512 sumi = _mm512_set1_ps(0.0);
  const __m512 conj = _mm512_set_ps(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1);

  __m512 *a = (__m512 *)A;
  __m512 *b = (__m512 *)B;

  const int n = (N / VECTOR_LENGTH_512);

  for (int i = 0; i < n; i++) {
   sumr = _mm512_fmadd_ps(a[i], b[i], sumr);

  __m512 bConj = _mm512_mul_ps(b[i], conj);
  __m512 bPerm = _mm512_permute_ps(bConj, 0b10110001);

   sumi = _mm512_fmadd_ps(a[i], bPerm, sumi);
  }

  float* sr = (float *)&sumr;
  float* si = (float *)&sumi;

  float sumR2 = sr[0] + sr[1] +sr[2] +sr[3] +sr[4] +sr[5] +sr[6] +sr[7] + sr[8] + sr[9] +sr[10] +sr[11] +sr[12] +sr[13] +sr[14] +sr[15];
  float sumI2 = si[0] + si[1] +si[2] +si[3] +si[4] +si[5] +si[6] +si[7] + si[8] + si[9] +si[10] +si[11] +si[12] +si[13] +si[14] +si[15];


  std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
  double dt = std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime).count();

  std::cout << "Vectorized method results" << std::endl;
  std::cout << "Result sum: " << sumR2 << " + i" << sumI2 << std::endl;
  std::cout << "Time taken: " << dt << std::endl;

  return dt;
}

int main(void) {
  const int N = 1 << 26;
  float *A = (float *)aligned_alloc(ALIGN, N * sizeof(float));
  float *B = (float *)aligned_alloc(ALIGN, N * sizeof(float));

  std::cout << "Initializing array" << std::endl;
  srand(0);
  for (int i = 0; i < N; ++i) {
    float ra = (2.0f * ((float)rand()) / RAND_MAX) - 1.0f;
    float rb = (2.0f * ((float)rand()) / RAND_MAX) - 1.0f;
    A[i] = ra;
    B[i] = rb;
  }
  std::cout << "Initailized array" << std::endl;

  std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
  float sumR = 0;
  float sumI = 0;

  // A = a + ib; B = c + id
  // AB* = (ac + ibc - iad + bd)
  // AB*r = ac + bd
  // AB*i = bc - ad
  for (int i = 0; i < N; i += 2) {
    float Ar = A[i];
    float Ai = A[i + 1];
    float Br = B[i];
    float Bi = -B[i + 1];

    float Cr = Ar * Br - Ai * Bi;
    float Ci = Ai * Br + Ar * Bi;

    sumR += Cr;
    sumI += Ci;
  }

  std::chrono::steady_clock::time_point stopTime = std::chrono::steady_clock::now();
  double dt = std::chrono::duration_cast<std::chrono::duration<double>>(stopTime - startTime).count();

  std::cout << "Naive method results" << std::endl;
  std::cout << "Result sum: " << sumR << " + i" << sumI << std::endl;
  std::cout << "Time taken: " << dt << std::endl;

  // double timeVec = dotP256(A, B, N);
  double timeVec = dotP512(A, B, N);

  std::cout << (dt - timeVec)/timeVec * 100 << "% faster" << std::endl;
}
