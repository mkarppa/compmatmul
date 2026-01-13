#ifndef COMPMATMUL_RNG_HPP
#define COMPMATMUL_RNG_HPP

// Random number generator class
// Uses Vigna's xoshiro256**
//
// Copyright (c) 2024 Matti Karppa
//

#include <cinttypes>
#include <cstdio>
#include <optional>
#include <chrono>


static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}



namespace compmatmul {
  // Derivative of work of Sebastiano Vigna (2015)
  // https://prng.di.unimi.it/splitmix64.c
  // implements splitmix64 to seed the rng state s
  inline void splitmix64(uint64_t x, uint64_t* s) {
    for (int i = 0; i < 4; ++i) {
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	s[i] = z ^ (z >> 31);
    }
  }



  /**
   * Returns current system time in milliseconds
   */
  inline uint64_t current_system_time_millis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch())
      .count();
  }



  /**
   * A pseudorandom number generator generating 64-bit unsigned integers using
   * the xoshiro256** algorithm
   */
  class rng {
  public:
    explicit rng(std::optional<uint64_t> seed = std::nullopt) {
      uint64_t x = seed.value_or(current_system_time_millis());
      splitmix64(x, s);
    }
    
    uint64_t operator()() {
      // Derivative of work of David Blackman and Sebastiano Vigna (2018)
      // Uses xoshiro256** for creating 64-bit integers
      // https://prng.di.unimi.it/xoshiro256starstar.c
      const uint64_t result = rotl(s[1] * 5, 7) * 9;
      const uint64_t t = s[1] << 17;
      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];
      s[2] ^= t;
      s[3] = rotl(s[3], 45);
      return result;
    }


    
  private:
    uint64_t s[4]; // state
  };



  /**
   * A pseudorandom number generator generating doubles using the xoshiro256+ 
   * algorithm
   */
  class drng {
  public:
    explicit drng(std::optional<uint64_t> seed = std::nullopt) {
      uint64_t x = seed.value_or(current_system_time_millis());
      splitmix64(x, s);
    }


    double operator()() {
      // Derivative of work of David Blackman and Sebastiano Vigna (2018)
      // Uses xoshiro256+ for creating 64-bit integers
      // https://prng.di.unimi.it/xoshiro256plus.c
      uint64_t r = s[0] + s[3];
      uint64_t t = s[1] << 17;
      s[2] ^= s[0];
      s[3] ^= s[1];
      s[1] ^= s[2];
      s[0] ^= s[3];
      s[2] ^= t;
      s[3] = rotl(s[3], 45);
      return (r >> 11) * 0x1.0p-53;
    }
    
  private:
    uint64_t s[4]; // state
  };
}


#endif // COMPMATMUL_RNG_HPP
