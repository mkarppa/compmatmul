#ifndef COMPMATMUL_ARRAY_HPP
#define COMPMATMUL_ARRAY_HPP

// This file acts as a middleware between the algorithm
// and MKL/AOCL/OpenBLAS
// Copyright (c) 2024 Joel Andersson
// Copyright (c) 2024,2025 Matti Karppa

#ifdef COMPMATMUL_USE_OPENBLAS
#include <cblas.h>
#endif // COMPMATMUL_USE_OPENBLAS
#ifdef COMPMATMUL_USE_MKL
// MKL allows manual definition of MKL_Complex8/16 if done before including header
#include <complex>
#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>
#include <mkl.h>
#endif // COMPMATMUL_USE_MKL
#include <cstdlib>
#include <new>
#include <complex>
#include <algorithm>
#include <bit>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cassert>

namespace compmatmul {
  namespace array {
#ifdef COMPMATMUL_USE_OPENBLAS
    using index_t = blasint;
#elif defined(COMPMATMUL_USE_MKL)
    using index_t = MKL_INT;
#endif // COMPMATMUL_USE_OPENBLAS
    static_assert(std::is_same<index_t,int>());
    
    /**
     * Convert a row major n*m matrix to a column major matrix in-place.
     */
    template<typename T>
    void c_to_fortran(index_t n, index_t m, T* A);

    template<>
    inline void c_to_fortran(index_t n, index_t m, double* A) {
#ifdef COMPMATMUL_USE_OPENBLAS
      cblas_dimatcopy(CblasRowMajor, CblasTrans, n, m, 1.0, A, n, m);
#elif defined(COMPMATMUL_USE_MKL)
      mkl_dimatcopy(CblasRowMajor, CblasTrans, n, m, 1.0, A, n, m);
#else
      static_assert(false, "No BLAS defined!");
#endif
    }

    /**
     * Convert a column major n*m matrix to a row major matrix in-place.
     */
    template<typename T>	// 
    void fortran_to_c(index_t n, index_t m, T* A);

    template<>
    inline void fortran_to_c(index_t n, index_t m, double* A) {
#ifdef COMPMATMUL_USE_OPENBLAS
      cblas_dimatcopy(CblasColMajor, CblasTrans, n, m, 1.0, A, n, m);
#elif defined(COMPMATMUL_USE_MKL)
      mkl_dimatcopy(CblasColMajor, CblasTrans, n, m, 1.0, A, n, m);
#else
      static_assert(false, "No BLAS defined!");
#endif
    }

    /**
     * Compute C = A^TB using BLAS GEMM. Both A and B are assumed to be 
     * stored in row-major order. C will be stored in row-major order.
     * Arguments:
     * @param N_a outer dimension of A (no cols)
     * @param N_inner inner dimension (no rows of A / no rows of B)
     * @param N_a outer dimension of B (no cols)
     */
    template<typename T>
    void matmul_gemm(index_t N_a, index_t N_inner, index_t N_b, T* C, const T* AT, const T* B);

    template<>
    inline void matmul_gemm<double>(index_t N_a, index_t N_inner, index_t N_b, double *C, const double *AT, const double *B) {
      assert(reinterpret_cast<uintptr_t>(C) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(B) % 64 == 0);
      assert(reinterpret_cast<uintptr_t>(AT) % 64 == 0);
#if defined(COMPMATMUL_USE_OPENBLAS) || defined(COMPMATMUL_USE_MKL)
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		  N_a, N_b, N_inner, 1.0, AT, N_a, B, N_b, 0.0, C, N_b);
#else
      static_assert(false, "No BLAS defined!");
#endif
    }

#ifndef NDEBUG
    // for checking that no memory leaks occur
    static size_t ALLOCATED_MEMORY = 0;
    static std::unordered_map<void*,size_t> ALLOCATIONS;
    static std::mutex ALLOCATION_MUTEX;
    namespace {
      class memory_killer {
      public:
        ~memory_killer() {
          if (ALLOCATED_MEMORY != 0) {
            std::cerr << "FATAL ERROR: UNALLOCATED MEMORY (" 
                      << ALLOCATED_MEMORY << " bytes) ON EXIT" << std::endl;
          }
        }
      };
      static memory_killer mk;
    }
#endif // NDEBUG

    /**
     * Allocate an 64-byte aligned array of specified size 
     * @tparam T datatype in array
     * @param size size of array (number of elements, not bytes)
     * @return T* pointer to 1d array
     */
    template<typename T>
    T* allocate(size_t size){
      size_t bytes = sizeof(T)*size;
      if (bytes % 64 != 0)
	  bytes += 64 - bytes % 64;
#ifdef COMPMATMUL_USE_OPENBLAS
      T* array = static_cast<T*>(aligned_alloc(64, bytes));
#elif defined(COMPMATMUL_USE_MKL)
      T* array = static_cast<T*>(mkl_malloc(bytes,64));
#else
      static_assert(false, "No BLAS defined!");
#endif
      if (array == nullptr) {
	      throw std::bad_alloc();
      }
#ifndef NDEBUG
      {
        std::lock_guard guard(ALLOCATION_MUTEX);
        ALLOCATED_MEMORY += bytes;
        if (ALLOCATIONS.count(array)) {
          std::cerr << "FATAL ERROR: ARRAY ALREADY IN ALLOCATIONS" << std::endl;
          fprintf(stderr, "ptr=%p\n", static_cast<void*>(array));
          exit(1);
        }
        ALLOCATIONS.insert({array,bytes});
      }
#endif
      return array;
    }

    /**
     * Deallocate the array allocated by allocate above.
     */
    template<typename T>
    void deallocate(T* array) {
#ifndef NDEBUG
      {
        std::lock_guard guard(ALLOCATION_MUTEX);
        auto it = ALLOCATIONS.find(array);
        if (it == ALLOCATIONS.end()) {
          std::cerr << "FATAL ERROR: TRIED TO DEALLOCATE NON-ALLOCATED MEMORY" << std::endl;
          fprintf(stderr, "ptr=%p\n", static_cast<void*>(array));
          exit(1);
        }
        size_t bytes = it->second;
        ALLOCATED_MEMORY -= bytes;
        ALLOCATIONS.erase(it);
      }
#endif

#ifdef COMPMATMUL_USE_OPENBLAS
      free(array);
#elif defined(COMPMATMUL_USE_MKL)
      mkl_free(array);
#else
      static_assert(false, "No BLAS defined!");
#endif
    }


    
    // TODO: homogenize naming and rename
    // TODO: remove scalar_mul, replaced with mul with a scalar
    template<typename T>
    void scalar_mul(index_t, T, T* __restrict);
    
    template<>
    inline void scalar_mul<double>(index_t n, double alpha, double *a) {
      cblas_dscal(n, alpha, a, 1);
    }

    /**
     * In-place scalar multiplication, computes 
     * x[i] = alpha * x[i] for i = 0..n-1
     */
    template<typename T>
    void mul(index_t n, T alpha, T* __restrict x);


    
    template<>
    inline void mul(index_t n, double alpha, double* __restrict x) {
      cblas_dscal(n, alpha, x, 1);
    }

    

    template<typename T>
    void coeff_mul(index_t n, T* __restrict, T* __restrict);

    template<>
    inline void coeff_mul<double>(index_t n, double* __restrict a, double* __restrict b) {
      // TODO: fix to be correct with both MKL and OpenBLAS
#ifdef COMPMATMUL_USE_MKL
        vdMul(n, a, b, a); // Elementwise multiplication
#elif defined(COMPMATMUL_USE_OPENBLAS)
#pragma omp simd
        for (index_t i = 0; i < n; i++) {
	        a[i] = b[i] * a[i];
        }
#else
        static_assert(false,"No BLAS defined!")
#endif
    }

    template<>
    inline void coeff_mul<std::complex<double>>(index_t n, std::complex<double>* __restrict a,
						std::complex<double>* __restrict b) {
      // TODO: fix to be correct with both MKL and OpenBLAS
#ifdef COMPMATMUL_USE_MKL
        vzMul(n, a, b, a);
#elif defined(COMPMATMUL_USE_OPENBLAS)
#pragma omp simd
        for (index_t i = 0; i < n; i++) {
	        a[i] = b[i] * a[i];
        }
#else 
        static_assert(false, "No BLAS defined!");
#endif
    }


    /**
     * Elementwise multiplication
     * Computes A = A * B (elementwise)
     */
    template<typename T>
    void mul(index_t n, T* __restrict A, T* __restrict B);

    template<>
    inline void mul<double>(index_t n, double* __restrict A, double* __restrict B) {
      // TODO: fix to be correct with both MKL and OpenBLAS
#if defined(COMPMATMUL_USE_MKL)
        vdMul(n, A, B, A); // Elementwise multiplication
#elif defined(COMPMATMUL_USE_OPENBLAS)
#pragma omp simd
        for (index_t i = 0; i < n; i++) {
           A[i] = A[i] * B[i];
        }
#else  
        static_assert(false, "No BLAS defined!");
#endif
    }

    template<>
    inline void mul<std::complex<double>>(index_t n, std::complex<double>* __restrict A,
						std::complex<double>* __restrict B) {
      // TODO: fix to be correct with both MKL and OpenBLAS
#if defined(PAGH_USE_MKL)
        vzMul(n, a, b, a);
#else
#pragma omp simd
      for (index_t i = 0; i < n; i++) {
	      A[i] = A[i] * B[i];
      }
#endif
    }



    template<typename T>
    void add(index_t, T* __restrict, T* __restrict);

    template<>
    inline void add<std::complex<double>>(index_t n, std::complex<double>* __restrict a,
                                   std::complex<double>* __restrict b) {
      // TODO: fix to be correct with both MKL and OpenBLAS (also use geadd?)
#ifdef COMPMATMUL_USE_MKL
        vzAdd(n, a, b, a);
#elif defined(COMPMATMUL_USE_OPENBLAS)
        #pragma omp simd
        for(index_t i = 0; i < n; i++){
            a[i] = a[i] + b[i];
        }
#else
        static_assert(false, "No BLAS defined!");
#endif
    }

    template<>
    void add<double>(index_t n, double* __restrict a, double* __restrict b) {
      // TODO fix to be correct in all cases and libraries
#ifdef COMPMATMUL_USE_MKL
        vdAdd(n, a, b, a);
#elif defined(COMPMATMUL_USE_OPENBLAS)
        cblas_daxpy(n, 1.0, b, 1, a, 1);
        // #pragma omp simd
        // for(index_t i = 0; i < n; i++){
        //     a[i] = a[i] + b[i];
        // }
#endif
    }
    
    /**
     * Returns the {size / 2} biggest element of xs, note this differs from the
     * mathematical median when size is even. Modifies xs!
     * @tparam T double/float
     * @param size
     * @param xs WILL BE MODIFIED!
     * @return xs_sorted[size / 2]
     */
    template<typename T>
    T median(index_t size, T *xs) {
      index_t mid = size / 2;
      std::nth_element(xs, xs + mid, xs + size);
      return xs[mid];
    }



    /**
     * Computes discrete base-2 logarithm of an unsigned value, that is, 
     * returns floor(log2(x))
     * Undefined for non-integer types and negative integers
     */
    template<typename T>
    T ilog2(T x) {
      return std::bit_width(x) - 1;
    }


    /**
     * Returns true iff x is a power of 2
     */
    template<typename T>
    bool is_pow2(T x) {
      static_assert(std::is_integral<T>(), "Only defined for integral types!");
      return x > 0 && (x & (x-1)) == 0;
    }


    /**
     * Returns true iff the two arrays are exactly equal
     */
    template<typename T>
    bool equal(index_t n, const T* A, const T* B) {
      for (index_t i = 0; i < n; ++i) {
	      if (A[i] != B[i])
	        return false;
      }
      return true;
    }


    /**
     * Returns true iff the two arrays are almost equal (in terms of absolute error)
     */
    template<typename T>
    bool almost_equal(index_t n, const T* A, const T* B, T abs_err = 1e-6) {
      for (index_t i = 0; i < n; ++i) {
	      if (std::abs(A[i] - B[i]) > abs_err)
	        return false;
      }
      return true;
    }
    
    
    
    /**
     * Copy data
     * B <- A
     */
    template<typename T>
    void mov(index_t n, T* __restrict B, const T* __restrict A) {
      #pragma omp simd
      for (index_t i = 0; i < n; ++i)
	      B[i] = A[i];
    }


    /**
     * Set all array elements to zero
     */
    template<typename T>
    void zero(size_t n, T* A);

    template<>
    void zero(size_t n, double* A) {
      memset(A, 0, sizeof(double) * n);
    }

    /**
     * GCC doesn't like memsetting std::complex
     */
    template<>
    void zero(size_t n, std::complex<double>* A) {
      // TODO: make more efficient
      std::fill(A,A+n,0);
    }


    
    /**
     * Return an array of specified size full of zeros.
     */
    template<typename T>
    T* zeros(size_t n) {
      // TODO: make more efficient
      T* x = allocate<T>(n);
      zero(n,x);
      return x;
    }
  }
}

#endif // COMPMATMUL_ARRAY_HPP
