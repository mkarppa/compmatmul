#ifndef COMPMATMUL_HASHING_HPP
#define COMPMATMUL_HASHING_HPP

//This file, along with hashing.cpp, implements three hashing methods
//two versions of multiply hash from the paper "High Speed Hashing for Integers and Strings"
//by Mikkel Thorup, as well as simple tabulation hashing, taken from the paper
//"The Power of Simple Tabulation Hashing" by Mikkel Thorup and Mihai Patrascu
// TODO: add more detailed references

// Copyright (c) 2024 Joel Andersson
// Copyright (c) 2024, 2025 Matti Karppa

#include "array.hpp"
#include "rng.hpp"
#include <random>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <climits>
#include <cstring>

namespace compmatmul {
  /**
   * Uses multiply-shift hashing to hash 32-bit keys into 32-bit hash values
   * cf. §2.3 in (Thorup, 2015).
   */
  class multiply_shift_hash {
  public:
    using key_t = uint32_t;
    using hash_t = uint32_t;
    using sign_t = int;

    // the default constructor will leave the function uninitialized
    multiply_shift_hash() = default;

    explicit multiply_shift_hash(size_t buckets,
				 uint64_t a) :
      a(a),
      right_shift_amount(array::is_pow2(buckets) ?
			 64 - array::ilog2(buckets) :
			 throw std::invalid_argument("The number of buckets "
						     "must be a power of 2")) {
    }

    

    /**
     * Draw one multiply_shift_hash function
     */
    static multiply_shift_hash draw_one(size_t buckets,
					std::optional<uint64_t> seed = std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      rng rng(seed);
      uint64_t a;
      do {
	a = rng();
      }
      while ((a&1) == 0);
      return multiply_shift_hash(buckets,a);
    }

    

    /**
     * Draw n multiply_shift_hash functions
     */
    static std::vector<multiply_shift_hash> draw(size_t n,
						 size_t buckets,
						 std::optional<uint64_t> seed =
						 std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      std::vector<multiply_shift_hash> h(n);
      rng rng(seed);
      uint64_t a;
      for (auto it = h.begin(); it != h.end(); ++it) {
	do {
	  a = rng();
	}
	while ((a&1) == 0);
	*it = multiply_shift_hash(buckets,a);
      }
      return h;
    }

    

    hash_t operator()(key_t key) const {
      return (a*key) >> right_shift_amount;
    }

    sign_t sign(key_t key) const {
      return ((a*key) >> 63) ? 1 : -1;
    };
    
  private:
    uint64_t a;             // random parameter, must be odd
    int right_shift_amount; // 64 - log2(buckets)
  };



  /**
   * Uses multiply-add-shift hashing to hash 32-bit keys into 32-bit hash values
   * cf. §3.3 in (Thorup, 2015).
   * Strongly universal!
   */
  class multiply_add_shift_hash {
  public:
    using key_t = uint32_t;
    using hash_t = uint32_t;
    using sign_t = int;

    multiply_add_shift_hash() = default;
    
    explicit multiply_add_shift_hash(size_t buckets,
				     uint64_t a,
				     uint64_t b) :
      a(a), b(b),
      right_shift_amount(array::is_pow2(buckets) ?
			 64 - array::ilog2(buckets) :
			 throw std::invalid_argument("The number of buckets "
						     "must be a power of 2")) {
    }


    
    static multiply_add_shift_hash draw_one(size_t buckets,
					    std::optional<uint64_t> seed =
					    std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      rng rng(seed);
      uint64_t a = rng();
      uint64_t b = rng();
      return multiply_add_shift_hash(buckets, a, b);
    }



    static std::vector<multiply_add_shift_hash> draw(size_t n,
						     size_t buckets,
						     std::optional<uint64_t> seed =
						     std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      std::vector<multiply_add_shift_hash> h(n);
      rng rng(seed);
      for (auto it = h.begin(); it != h.end(); ++it) {
	uint64_t a = rng();
	uint64_t b = rng();
	*it = multiply_add_shift_hash(buckets, a, b);
      }
      return h;
    }


    hash_t operator()(key_t key) const {
      return (a*key + b) >> right_shift_amount;
    }

    sign_t sign(key_t key) const {
      return ((a*key + b) >> 63) ? 1 : -1;
    }

    uint64_t get_a() const {
      return a;
    }

    uint64_t get_b() const {
      return b;
    }
    
  private:
    uint64_t a;             // random parameter, for multiplication
    uint64_t b;             // random parameter, for addition
    int right_shift_amount; // 64 - log2(buckets)
  };



  /**
   * This class works as an adaptor to convert a bucket hash (mapping to [m])
   * into a sign hash (mapping to +/-1)
   * @tparam Hash The hash class
   */
  template<typename Hash>
  class sign_hash {
  public:
    using key_t = Hash::key_t;
    using hash_t = int;

    // uninitialized default constructor
    sign_hash() = default;


    static sign_hash draw_one(std::optional<uint64_t> seed = std::nullopt) {
      return sign_hash(Hash::draw_one(2,seed));
    }

    static std::vector<sign_hash> draw(size_t n, std::optional<uint64_t> seed =
				       std::nullopt) {
      std::vector<Hash> h = Hash::draw(n,2,seed);
      std::vector<sign_hash> s(n);
      auto it = s.begin();
      auto jt = h.begin();
      while (it != s.end())
	*it++ = sign_hash(*jt++);
      return s;
    }

    hash_t operator()(key_t key) const {
      return hash(key) > 0 ? 1 : -1;
    }

  private:
    // initialize with given hash function
    explicit sign_hash(const Hash& h) :
      hash(h) {
    }

    Hash hash;
  };



  template<int CharBits, typename Key, typename Hash>
  class tabulation_hash {
  public:
    static_assert(CharBits % CHAR_BIT == 0, "CharBits must be a multiple of byte size");

    using key_t = Key;
    using hash_t = Hash;
    using sign_t = int;

    /**
     * Default constructor, leaves the function uninitialized.
     */
    tabulation_hash() = default;

    /**
     * Explicit constructor: copy table from S
     */
    tabulation_hash(size_t buckets,
		    const Hash* S) :
      right_shift_amount(array::is_pow2(buckets) ?
			 sizeof(Hash)*CHAR_BIT - array::ilog2(buckets) :
			 throw std::invalid_argument("The number of buckets "
						     "must be a power of 2")) {
      memcpy(&T[0][0], S, sizeof(T));
    }


   
    static tabulation_hash draw_one(size_t buckets,
				    std::optional<uint64_t> seed =
				    std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      rng rng(seed);
      tabulation_hash h;
      h.right_shift_amount = sizeof(Hash)*CHAR_BIT - array::ilog2(buckets);
      Hash* it = &h.T[0][0];
      while (it != &h.T[0][0] + sizeof(T)/sizeof(Hash))
	*it++ = rng();
      return h;
    }



    static std::vector<tabulation_hash> draw(size_t n,
					     size_t buckets,
					     std::optional<uint64_t> seed =
					     std::nullopt) {
      if (!array::is_pow2(buckets)) {
	throw std::invalid_argument("The number of buckets must be a power of 2");
      }
      rng rng(seed);
      std::vector<tabulation_hash> h(n);
      int right_shift_amount = sizeof(Hash)*CHAR_BIT - array::ilog2(buckets);
      for (auto it = h.begin(); it != h.end(); ++it) {
	it->right_shift_amount = right_shift_amount;
	Hash* jt = &it->T[0][0];
	while (jt != &it->T[0][0] + sizeof(T)/sizeof(Hash)) {
	  *jt++ = rng();
	}
      }
      return h;
    }


    hash_t operator()(Key key) const {
      Hash h = 0;
      for (size_t i = 0; i < sizeof(Key)/(CharBits/CHAR_BIT); ++i) {
	Key c = (key >> (i*CharBits)) & char_mask;
	h ^= T[i][c];
      }
      return h >> right_shift_amount;
    }

    sign_t sign(Key key) const {
      Hash h = 0;
      for (size_t i = 0; i < sizeof(Key)/(CharBits/CHAR_BIT); ++i) {
	Key c = (key >> (i*CharBits)) & char_mask;
	h ^= T[i][c];
      }
      return (h >> (sizeof(Hash)*CHAR_BIT-1)) ? 1 : -1;
    }

  private:
    static const Hash char_mask = ~(~Hash(0) << CharBits);
    Hash T[sizeof(Key)/(CharBits/CHAR_BIT)][(1 << CharBits)];
    int right_shift_amount;
  };
}

#endif // COMPMATMUL_HASHING_HPP
