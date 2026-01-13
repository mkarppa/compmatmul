#ifndef COMPMATMUL_MATMUL_FWHT_HPP
#define COMPMATMUL_MATMUL_FWHT_HPP

#ifdef COMPMATMUL_USE_FXT

#include "rng.hpp"
#include "transform.hpp"
#include "matmul_sketch.hpp"
#include "array.hpp"
#include <omp.h>
#include <vector>

namespace compmatmul {
  template<typename Hash>
  using matmul_fwht_sketch = matmul_sketch<Hash>;
  using index_t = array::index_t;

  /**
   * Computes the compressed product sketch of A^T B using FWHT instead of FFT
   * A^T is assumed to be n_inner * n_a (row-major)
   * B is assumed to be n_inner * n_b (row-major)
   * The result is d polybnomials of length b, that is, p is an array of 
   * length d*b
   */
  template<typename Hash>
  matmul_fwht_sketch<Hash> matmul_compressed_product_fwht(index_t n_a, 
      index_t n_inner, index_t n_b, const double* A, const double* B, int d, 
      int b, std::optional<uint64_t> seed = std::nullopt) {
    // draw hash functions
    if (!seed) {
      seed = std::optional<uint64_t>(rng()());
    }
    matmul_fwht_sketch<Hash> sketch(d, b, *seed);
    double* p = sketch.p.get();
    array::zero(d*b, p);
    int num_threads;
    #pragma omp parallel shared(num_threads) 
    {
      #pragma omp single
      {
        num_threads = omp_get_num_threads();
      }
    }

    // one transformer per thread
    // although this is not really necessary, is it?
    std::vector<fwht_transformer> fwhts(num_threads, fwht_transformer(b));

    std::vector<omp_lock_t> p_locks(d);
    for (int t = 0; t < d; ++t)
      omp_init_lock(&p_locks[t]);

    #pragma omp parallel
    {
      int thread_num = omp_get_thread_num();
      fwht_transformer& fwht = fwhts[thread_num];

      // Fastest approach is to compute all pas and store them,
	    // then compute each pb one by one and do all remaining operations

      double* pas = array::allocate<double>(b * d);
      double* pb = array::allocate<double>(b);

      #pragma omp for schedule(guided)
      for(int k = 0; k < n_inner; k++) {
        for (int t = 0; t < d; t++) {
          const Hash& h1t = sketch.h1[t];
          const Hash& s1t = sketch.s1[t];
          double* pa = pas + t*b;
          array::zero(b, pa);
          // A is in column-major order, so A[k*n_a+i] = A[i][k]
          for (uint32_t i = 0; i < static_cast<uint32_t>(n_a); ++i) {
            pa[h1t(i)] += s1t.sign(i) * A[k*n_a + i];
          }
          fwht.fwd(pa);
        }

        for (int t = 0; t < d; t++) {
          const Hash& h2t = sketch.h2[t];
          const Hash& s2t = sketch.s2[t];
          array::zero(b, pb);
          // B is in row-major order, so B[k*n_b+i] = A[k][i]
          for (uint32_t i = 0; i < static_cast<uint32_t>(n_b); ++i) {
            pb[h2t(i)] += s2t.sign(i) * B[k*n_b + i];
          }
          fwht.fwd(pb);

          double* pa = pas + t*b;
          array::mul(b, pb, pa);

          omp_set_lock(&p_locks[t]);
          array::add(b, p + t*b, pb);
          omp_unset_lock(&p_locks[t]);
        }
      }

      array::deallocate(pas);
      array::deallocate(pb);
    }

    #pragma omp parallel
    {
      fwht_transformer& fwht = fwhts[omp_get_thread_num()];
      #pragma omp for
      for (int t = 0; t < d; ++t) {
        fwht.inv(p + t*b);
      }
    }

    for (int t = 0; t < d; ++t)
      omp_destroy_lock(&p_locks[t]);

    return sketch;
  }



  /**
   * Decompress element C[i,j] from the sketch
   * There should be d elements of scratch space
   */
  template<typename Hash>
  void matmul_decompress_fwht(int cols, double *C, uint32_t i, uint32_t j, 
    const matmul_fwht_sketch<Hash>& sketch, double *scratch) {
    const double* p = sketch.p.get();
    int d = sketch.d;
    int b = sketch.b;
    for (int t = 0; t < d; ++t) {
      uint32_t h = sketch.h1[t](i) ^ sketch.h2[t](j);
      int s = sketch.s1[t].sign(i) * sketch.s2[t].sign(j);
      scratch[t] = s * p[t*b + h];
    }
    C[i*cols + j] = array::median(d, scratch);
  }



  /**
   * Computes unbiased estimate of C=A^T B.
   * That is, AT is expected to have shape n_inner * n_a,
   * B the shape n_inner * n_b, and C the shape n_a * n_b
   * All assumed to be row-major
   */
  template<typename Hash>
  void matmul_fwht(int n_a, int n_inner, int n_b, double* C, const double* AT, 
    const double* B, int d, int b, 
    std::optional<uint64_t> seed = std::nullopt) {
      assert(reinterpret_cast<uintptr_t>(AT) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(C) % 64 == 0);
    auto sketch = matmul_compressed_product_fwht<Hash>(n_a, n_inner, n_b, AT, 
      B, d, b, seed);
    
    #pragma omp parallel 
    {
      double* scratch = array::allocate<double>(d);

      #pragma omp for schedule(guided)
      for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_b; ++j) {
          matmul_decompress_fwht(n_b, C, i, j, sketch, scratch);
        }
      }

      array::deallocate(scratch);
    }
  }
}

#endif // COMPMATMUL_USE_FXT

#endif // COMPMATMUL_MATMUL_FWHT_HPP