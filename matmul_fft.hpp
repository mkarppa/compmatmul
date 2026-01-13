#ifndef COMPMATMUL_MATMUL_FFT_HPP
#define COMPMATMUL_MATMUL_FFT_HPP

#ifdef COMPMATMUL_USE_FFTW

#include "rng.hpp"
#include "transform.hpp"
#include "matmul_sketch.hpp"
#include "array.hpp"
#include <omp.h>
#include <optional>
#include <vector>

namespace compmatmul {
  template<typename Hash>
  using matmul_fft_sketch = matmul_sketch<Hash>;
  using index_t = array::index_t;

  /**
   * Computes the compressed product sketch of A^T B
   * A^T is assumed to be n_inner * n_a (row-major)
   * B is assumed to be n_inner * n_b (row-major)
   * The result is d polybnomials of length b, that is, p is an array of length d*b
   */
  template<typename Hash>
  matmul_fft_sketch<Hash> matmul_compressed_product_fft(int n_a, int n_inner, 
    int n_b, const double* A, const double* B, int d, int b,
    std::optional<uint64_t> seed = std::nullopt) {
    // draw hash functions
    if (!seed) {
      seed = std::optional<uint64_t>(rng()());
    }
    matmul_fft_sketch<Hash> sketch(d, b, *seed);
    // size of the real-transformed polynomial
    const int fft_poly_size = b/2 + 1;
    // intermediate array for transformed polynomials
    std::complex<double>* p_fft = array::zeros<std::complex<double>>(fft_poly_size*d);

    int num_threads;
    #pragma omp parallel shared(num_threads) 
    {
      #pragma omp single
      {
        num_threads = omp_get_num_threads();
      }
    }
    // one fftw transformer per thread
    std::vector<std::unique_ptr<fft_transformer>> ffts(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      ffts[i] = std::make_unique<fft_transformer>(b);
    }

    std::vector<omp_lock_t> p_locks(d);
    for (int t = 0; t < d; ++t)
      omp_init_lock(&p_locks[t]);

    #pragma omp parallel
    {
      std::complex<double>* fft_pa = array::allocate<std::complex<double>>(fft_poly_size);
      std::complex<double>* fft_pb = array::allocate<std::complex<double>>(fft_poly_size);

      int thread_num = omp_get_thread_num();
      fft_transformer& fft = *ffts[thread_num];

      // untransformed polynomials
      double* pas = array::allocate<double>(b * d);
      double* pbs = array::allocate<double>(b * d);


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
        }

        for (int t = 0; t < d; t++) {
          const Hash& h2t = sketch.h2[t];
          const Hash& s2t = sketch.s2[t];
          double* pb = pbs + t*b;
          array::zero(b, pb);
          // B is in row-major order, so B[k*n_b+i] = A[k][i]
          for (uint32_t i = 0; i < static_cast<uint32_t>(n_b); ++i) {
            pb[h2t(i)] += s2t.sign(i) * B[k*n_b + i];
          }
        }


        for (int t = 0; t < d; ++t) {
          double* pa = pas + t*b;
          fft.fwd(fft_pa, pa);
          double* pb = pbs + t*b;
          fft.fwd(fft_pb, pb);
          array::mul(fft_poly_size, fft_pa, fft_pb);

          // acquire lock for p_fft[t]
          omp_set_lock(&p_locks[t]);
          array::add(fft_poly_size, p_fft + t*fft_poly_size, fft_pa);
          omp_unset_lock(&p_locks[t]);
        }
      }
      array::deallocate(pas);
      array::deallocate(pbs);
      array::deallocate(fft_pa);
      array::deallocate(fft_pb);
    }
    
    double* p = sketch.p.get();
    #pragma omp parallel 
    { 
      fft_transformer& fft = *ffts[omp_get_thread_num()];
      #pragma omp for
      for (int t = 0; t < d; ++t) {
        fft.inv(p + t*b, p_fft + t*fft_poly_size);
      }
    }
    

    for (int t = 0; t < d; ++t)
      omp_destroy_lock(&p_locks[t]);
    array::deallocate(p_fft);
    return sketch;
  }

  /**
   * Decompress element C[i,j] from the sketch
   * There should be d elements of scratch space
   */
  template<typename Hash>
  void matmul_decompress_fft(int cols, double *C, uint32_t i, uint32_t j, 
    const matmul_fft_sketch<Hash>& sketch, double *scratch) {
    const double* p = sketch.p.get();
    int d = sketch.d;
    int b = sketch.b;
    for (int t = 0; t < d; ++t) {
      uint32_t h = (sketch.h1[t](i) + sketch.h2[t](j)) & sketch.mod_mask;
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
  void matmul_fft(int n_a, int n_inner, int n_b, double* C, const double* AT, 
    const double* B, int d, int b, 
    std::optional<uint64_t> seed = std::nullopt) {
      assert(reinterpret_cast<uintptr_t>(AT) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(C) % 64 == 0);
    auto sketch = matmul_compressed_product_fft<Hash>(n_a, n_inner, n_b, AT, 
      B, d, b, seed);
    
    #pragma omp parallel 
    {
      double* scratch = array::allocate<double>(d);

      #pragma omp for schedule(guided)
      for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_b; ++j) {
          matmul_decompress_fft(n_b, C, i, j, sketch, scratch);
        }
      }

      array::deallocate(scratch);
    }
  }

}

#endif // COMPMATMUL_USE_FFTW

#endif // COMPMATMUL_MATMUL_FFT_HPP
