#ifndef COMPMATMUL_TRANSFORM_HPP
#define COMPMATMUL_TRANSFORM_HPP

// This file contains the fft and fwht transformation convenience classes
// Copyright (c) 2024 Joel Andersson
// Copyright (c) 2024 Matti Karppa

#include "array.hpp"
#include <complex>

#ifdef COMPMATMUL_USE_FFTW
#include <fftw3.h>
#endif // COMPMATMUL_USE_FFTW

#ifdef COMPMATMUL_USE_FXT
#include <walsh/walshwak.h>
#endif // COMPMATMUL_USE_FXT


namespace compmatmul {
#ifdef COMPMATMUL_USE_FFTW
  /**
   * This class provides convenient forward and inverse real DFT,
   * using the FFTW interface
   */
  class fft_transformer {
  public:
    using plan_t = std::remove_pointer<fftw_plan>::type;
    using complex_t = std::complex<double>;
    using real_t = double;
    using index_t = array::index_t;
    
    /**
     * Constructs the transformer for transforming real sequences of n elements
     * Will allocate all data structures
     * The length of the sequence n must be a power of 2
     */
    explicit fft_transformer(index_t n) :
      n_real(array::is_pow2(n) ? n :
	     throw std::invalid_argument("n must be a power of 2")),
      n_complex(n/2+1),
      real(static_cast<real_t*>(fftw_malloc(sizeof(real_t)*n_real)), fftw_free),
      complex(static_cast<complex_t*>(fftw_malloc(sizeof(complex_t)*n_complex)), fftw_free),
      fwd_plan(fftw_plan_dft_r2c_1d(n, real.get(),
				    reinterpret_cast<fftw_complex*>(complex.get()),
				    FFTW_MEASURE), fftw_destroy_plan),
      inv_plan(fftw_plan_dft_c2r_1d(n, reinterpret_cast<fftw_complex*>(complex.get()),
				    real.get(), FFTW_MEASURE), fftw_destroy_plan)
    {
      
    }


    /**
     * Performs the forward transformation. Copies data first into a buffer.
     */
    void fwd(complex_t* out, const real_t* in) {
      memcpy(real.get(), in, sizeof(real_t)*n_real);
      fftw_execute(fwd_plan.get());
      memcpy(out, complex.get(), sizeof(complex_t)*n_complex);
    }


    
    /**
     * Performs the inverse transformation. Copies data first into a buffer.
     */
    void inv(real_t* out, const complex_t* in) {
      memcpy(complex.get(), in, sizeof(complex_t)*n_complex);
      fftw_execute(inv_plan.get());
      array::mul(n_real, 1.0/n_real, real.get());
      memcpy(out, real.get(), sizeof(real_t)*n_real);
    }


    
  private:
    index_t n_real;    // length of the real sequence
    index_t n_complex; // length of the transformed sequence (n/2+1)
    std::unique_ptr<real_t,void(*)(void*)> real;
    std::unique_ptr<complex_t,void(*)(void*)> complex;
    std::unique_ptr<plan_t,void(*)(plan_t*)> fwd_plan;   // real-to-complex dft plan
    std::unique_ptr<plan_t,void(*)(plan_t*)> inv_plan;   // complex-to-real idft plan
  };
#endif // COMPMATMUL_USE_FFTW


  
#ifdef COMPMATMUL_USE_FXT
  /**
   * This class presents abstract the implementation of the Walsh-Hadamard 
   * transform. The transform is computed in-place.
   */
  class fwht_transformer {
  public:
    using real_t = double;    
   
    explicit fwht_transformer(size_t n) :
      n(array::is_pow2(n) ? n : 
        throw std::invalid_argument("n must be a power of 2")), 
        ldn(array::ilog2(n)) {
    }

    

    /**
     * Perform the transformation in-place
     * The transform is its own inverse apart from normalization
     */
    void fwd(real_t* x) {
      walsh_wak(x, ldn);
    }

    /**
     * Perform the transformation out-of-place
     * The transform is its own inverse apart from normalization
     */
    void fwd(real_t* out, const real_t* in) {
      memcpy(out, in, sizeof(real_t)*n);
      fwd(out);
    }

    /**
     * Perform the transformation in-place
     * The transform is its own inverse apart from normalization
     */
    void inv(real_t* x) {
      walsh_wak(x, ldn);
      array::mul(n, 1.0 / n, x);
    }

    /**
     * Perform the transformation out-of-place
     * The transform is its own inverse apart from normalization
     */
    void inv(real_t* out, const real_t* in) {
      memcpy(out, in, sizeof(real_t)*n);
      inv(out);
    }

  private:
    size_t n;
    unsigned long ldn; // log2 of n
  };
#endif // COMPMATMUL_USE_FXT
}

#endif // COMPMATMUL_TRANSFORM_HPP
