#ifndef COMPMATMUL_MATMUL_SKETCH_HPP
#define COMPMATMUL_MATMUL_SKETCH_HPP

#include "array.hpp"
#include <optional>

namespace compmatmul {
  template<typename Hash>
  class matmul_sketch {
  public:
    matmul_sketch(int d, int b, 
      uint64_t seed) : // seed must be given because otherwise we get into trouble
      d(d), b(b), 
      mod_mask(~(~UINT32_C(0) << array::ilog2(static_cast<uint32_t>(b)))),
      h1(Hash::draw(d, b, seed)), 
      h2(Hash::draw(d, b, seed + 1)),
      s1(Hash::draw(d, b, seed + 2)), 
      s2(Hash::draw(d, b, seed + 3)),
      p(array::zeros<double>(d * b),array::deallocate) {
      }

      matmul_sketch(const matmul_sketch& that) :
        d(that.d), b(that.b), mod_mask(that.mod_mask),
        h1(that.h1),
        h2(that.h2),
        s1(that.s1),
        s2(that.s2),
        p(array::allocate<double>(d*b),array::deallocate) {
        memcpy(p.get(),that.p.get(),sizeof(double)*d*b);
      }

      matmul_sketch(matmul_sketch&& that) :
        d(that.d), b(that.d), mod_mask(that.mod_mask),
        h1(std::move(that.h1)),
        h2(std::move(that.h2)),
        s1(std::move(that.s1)),
        s2(std::move(that.s2)),
        p(std::move(that.p)) {
      }

      matmul_sketch& operator=(const matmul_sketch& that) {
        if (this != &that) {
          matmul_sketch copy(that);
          std::swap(d, copy.d);
          std::swap(b, copy.b);
          std::swap(mod_mask, copy.mod_mask);
          std::swap(h1, copy.h1);
          std::swap(h2, copy.h2);
          std::swap(s1, copy.s1);
          std::swap(s2, copy.s2);
          std::swap(p, copy.p);
        }
        return *this;
      }

      matmul_sketch& operator=(matmul_sketch&& that) {
        if (this != &that) {
          std::swap(d, that.d);
          std::swap(b, that.b);
          std::swap(mod_mask, that.mod_mask);
          std::swap(h1, that.h1);
          std::swap(h2, that.h2);
          std::swap(s1, that.s1);
          std::swap(s2, that.s2);
          std::swap(p, that.p);
        }
        return *this;
      }

      ~matmul_sketch() {
      }

      int d;
      int b;
      uint32_t mod_mask; // mask such that x & mod_mask == x % b
      std::vector<Hash> h1;
      std::vector<Hash> h2;
      std::vector<Hash> s1;
      std::vector<Hash> s2;
      std::unique_ptr<double,void(*)(double*)> p;
  };
}

#endif // COMPMATMUL_MATMUL_SKETCH_HPP
