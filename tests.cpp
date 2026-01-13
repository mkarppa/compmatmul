#ifdef COMPMATMUL_USE_FXT
#include "matmul_fwht.hpp"
#endif // COMPMATMUL_USE_FXT
#ifdef COMPMATMUL_USE_FFTW
#include "matmul_fft.hpp"
#endif // COMPMATMUL_USE_FFTW
#include "transform.hpp"
#include "hashing.hpp"
#include "array.hpp"
#include "rng.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <random>

using namespace compmatmul;
using namespace compmatmul::array;

static double AT_4x4[4*4] = {
  1, 5,  9, 13,
  2, 6, 10, 14,
  3, 7, 11, 15,
  4, 8, 12, 16
};

static double AT_2x4[2*4] = {
  1, 3, 5, 7,
  2, 4, 6, 8
};

static double AT_4x2[4*2] = {
  1, 5,
  2, 6,
  3, 7,
  4, 8
};


static double B_4x4[4*4] = {
  17, 18, 19, 20,
  21, 22, 23, 24,
  25, 26, 27, 28,
  29, 30, 31, 32
};

static double B_2x4[2*4] = {
  9,  10, 11, 12,
  13, 14, 15, 16,
};

static double B_4x2[4*2] = {
  9,  10,
  11, 12,
  13, 14,
  15, 16,  
};

static double B_4x8[4*8] = {
  1,   2,  3,  4,  5,  6,  7,  8,
  9,  10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24,
  25, 26, 27, 28, 29, 30, 31, 32
};

static double C_4x4x4[4*4] = {
   250,  260,  270,  280,
   618,  644,  670,  696,
   986, 1028, 1070, 1112,
  1354, 1412, 1470, 1528
};

static double C_4x2x4[4*4] = {
   35,  38,  41,  44,
   79,  86,  93, 100,
  123, 134, 145, 156,
  167, 182, 197, 212
};

static double C_2x4x2[2*2] = {
  130, 140,
  322, 348
};

static double C_2x4x8[2*8] = {
  170, 180, 190, 200, 210, 220, 230, 240,
  378, 404, 430, 456, 482, 508, 534, 560
};

TEST_CASE( "test_rng", "[rng][unit]" ) {
  rng rng(1234);
  REQUIRE( rng() == 0x0bab45d9a0e3ae53 );
  REQUIRE( rng() == 0xd7c640660c19433e );
  REQUIRE( rng() == 0xb0dedaa0d09a6691 );
  REQUIRE( rng() == 0xdec9f41b58ec86eb );
}



TEST_CASE( "test_ilog2", "[log][unit]" ) {
  REQUIRE( ilog2(1u) == 0 );
  REQUIRE( ilog2(2u) == 1 );
  REQUIRE( ilog2(3u) == 1 );
  REQUIRE( ilog2(4u) == 2 );
  REQUIRE( ilog2(5u) == 2 );
  REQUIRE( ilog2(6u) == 2 );
  REQUIRE( ilog2(7u) == 2 );
  REQUIRE( ilog2(8u) == 3 );
  REQUIRE( ilog2(9u) == 3 );
  REQUIRE( ilog2(10u) == 3 );
  REQUIRE( ilog2(11u) == 3 );
  REQUIRE( ilog2(12u) == 3 );
  REQUIRE( ilog2(13u) == 3 );
  REQUIRE( ilog2(14u) == 3 );
  REQUIRE( ilog2(15u) == 3 );
  REQUIRE( ilog2(16u) == 4 );

  REQUIRE(array::ilog2(~UINT64_C(0)) == 63);
  for (int i = 63; i >= 0; --i) {
    REQUIRE(array::ilog2(UINT64_C(1) << i) == static_cast<uint64_t>(i));
  }
}



TEST_CASE( "perf_rng", "[rng][benchmark]" ) {
  rng rng(1000);
  BENCHMARK("rng 1000000") {
    for (int i = 1; i < 1000000; ++i) {
      rng();
    }
    return rng();
  };

  std::mt19937_64 mtrng(1000);
  BENCHMARK("mt19937_64 1000000") {
    for (int i = 1; i < 1000000; ++i) {
      mtrng();
    }
    return mtrng();
  };  
}



TEST_CASE( "test_multiply_shift", "[hash][unit]" ) {
  uint64_t seed = 123456;
  uint32_t m = 64;
  uint64_t a = 2749059519329956733;

  multiply_shift_hash msh1(m, a);
  multiply_shift_hash msh2 = multiply_shift_hash::draw_one(m, seed);
  multiply_shift_hash msh3 = multiply_shift_hash::draw(1, m, seed)[0];
  multiply_shift_hash::hash_t counts[64] = { 0 };
  unsigned int n = 128;
  for (multiply_shift_hash::key_t i = 0; i <= n; ++i) {
    multiply_shift_hash::hash_t h1 = msh1(i);
    multiply_shift_hash::hash_t h2 = msh2(i);
    multiply_shift_hash::hash_t h3 = msh3(i);
    multiply_shift_hash::hash_t h = ((a*i) >> 58);
    REQUIRE(h == h1);
    REQUIRE(h == h2);
    REQUIRE(h == h3);
    REQUIRE(h < 64);
    ++counts[h];
  }
  for (int i = 0; i < 64; ++i) {
    REQUIRE(counts[i] > 0);
  }

  auto sign = sign_hash<multiply_shift_hash>::draw_one(seed);
  for (sign_hash<multiply_shift_hash>::key_t i = 0; i <= n; ++i) {
    sign_hash<multiply_shift_hash>::hash_t s1 = sign(i);
    multiply_shift_hash::sign_t s2 = msh1.sign(i);
    if ((a*i >> 63) == 1) {
      REQUIRE(s1 == 1);
      REQUIRE(s2 == 1);
    }
    else {
      REQUIRE(s1 == -1);
      REQUIRE(s2 == -1);
    }
  }

  uint64_t a1 = UINT64_C(3688180003657674949);
  uint64_t a2 = UINT64_C(11729574476469721337);
  seed = 19850;

  m = 32;
  n = 1024;
  auto mshs = multiply_shift_hash::draw(2,m,seed);
  for (multiply_shift_hash::key_t i = 0; i <= n; ++i) {
    multiply_shift_hash::hash_t h1 = ((a1 * i) >> 59);
    REQUIRE(mshs[0](i) == h1);
    multiply_shift_hash::hash_t h2 = ((a2 * i) >> 59);
    REQUIRE(mshs[1](i) == h2);
  }

  auto signs1 = sign_hash<multiply_shift_hash>::draw(2,seed);
  auto signs2 = multiply_shift_hash::draw(2,2,seed);
  for (sign_hash<multiply_shift_hash>::key_t i = 0; i <= n; ++i) {
    sign_hash<multiply_shift_hash>::hash_t h1 = signs1[0](i);
    sign_hash<multiply_shift_hash>::hash_t h2 = signs1[1](i);
    multiply_shift_hash::sign_t s1 = signs2[0].sign(i);
    multiply_shift_hash::sign_t s2 = signs2[1].sign(i);
    if ((a1*i >> 63) == 1) {
      REQUIRE(h1 == 1);
      REQUIRE(s1 == 1);
    }
    else {
      REQUIRE(h1 == -1);
      REQUIRE(s1 == -1);
    }
    if ((a2*i >> 63) == 1) {
      REQUIRE(h2 == 1);
      REQUIRE(s2 == 1);
    }
    else {
      REQUIRE(h2 == -1);
      REQUIRE(s2 == -1);
    }
  }
}




TEST_CASE( "test_multiply_add_shift", "[hash][unit]" ) {
  uint64_t seed = 123456;
  uint32_t m = 128;
  const uint64_t args[] = { 2366053268901514180, 2749059519329956733 };
  const uint64_t a = args[0];
  const uint64_t b = args[1];

  multiply_add_shift_hash mash1(m, a, b);
  multiply_add_shift_hash mash2 = multiply_add_shift_hash::draw_one(m, seed);
  multiply_add_shift_hash mash3 = multiply_add_shift_hash::draw(1, m, seed)[0];
  multiply_add_shift_hash::hash_t counts[128] = { 0 };
  unsigned int n = 512;
  for (multiply_shift_hash::key_t i = 0; i <= n; ++i) {
    multiply_add_shift_hash::hash_t h = ((a*i+b) >> 57);
    multiply_add_shift_hash::hash_t h1 = mash1(i);
    multiply_add_shift_hash::hash_t h2 = mash2(i);
    multiply_add_shift_hash::hash_t h3 = mash3(i);
    REQUIRE(h1 == h);
    REQUIRE(h2 == h);
    REQUIRE(h3 == h);
    REQUIRE(h < 128);
    ++counts[h];
  }
  for (int i = 0; i < 64; ++i) {
    REQUIRE(counts[i] > 0);
  }

  auto sign = sign_hash<multiply_add_shift_hash>::draw_one(seed);
  for (sign_hash<multiply_shift_hash>::key_t i = 0; i <= n; ++i) {
    sign_hash<multiply_add_shift_hash>::hash_t s1 = sign(i);
    multiply_add_shift_hash::sign_t s2 = mash1.sign(i);
    if (((a*i+b) >> 63) == 1) {
      REQUIRE(s1 == 1);
      REQUIRE(s2 == 1);
    }
    else {
      REQUIRE(s1 == -1);
      REQUIRE(s2 == -1);
    }
  }

  seed = 22201;
  n = 2048;
  m = 128;
  uint64_t a1 = UINT64_C(2016826348251915964);
  uint64_t b1 = UINT64_C(17696970887482165528);
  uint64_t a2 = UINT64_C(6666994550483350943);
  uint64_t b2 = UINT64_C(1097363203091539945);
  auto mashes = multiply_add_shift_hash::draw(2,m,seed);
  for (multiply_add_shift_hash::key_t i = 0; i <= n; ++i) {
    multiply_add_shift_hash::hash_t h1 = ((a1*i + b1) >> 57);
    multiply_add_shift_hash::hash_t h2 = ((a2*i + b2) >> 57);
    REQUIRE(mashes[0](i) == h1);
    REQUIRE(mashes[1](i) == h2);
  }

  auto signs = sign_hash<multiply_add_shift_hash>::draw(2,seed);
  for (sign_hash<multiply_shift_hash>::key_t i = 0; i <= n; ++i) {
    sign_hash<multiply_add_shift_hash>::hash_t h1 = signs[0](i);
    sign_hash<multiply_add_shift_hash>::hash_t h2 = signs[1](i);
    multiply_add_shift_hash::sign_t s1 = mashes[0].sign(i);
    multiply_add_shift_hash::sign_t s2 = mashes[1].sign(i);
    if (((a1*i+b1) >> 63) == 1) {
      REQUIRE(h1 == 1);
      REQUIRE(s1 == 1);
    }
    else {
      REQUIRE(h1 == -1);
      REQUIRE(s1 == -1);
    }
    if (((a2*i+b2) >> 63) == 1) {
      REQUIRE(h2 == 1);
      REQUIRE(s2 == 1);
    }
    else {
      REQUIRE(h2 == -1);
      REQUIRE(s2 == -1);
    }
  }

}



TEST_CASE( "test_tabulation_hash", "[hash][unit]" ) {
  uint64_t seed = 23711;
  uint32_t m = 256;
  rng prng(seed);
  uint32_t T1[4][256];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 256; ++j)
      T1[i][j] = prng();
  tabulation_hash<8,uint32_t,uint32_t> th8_1(m,&T1[0][0]);
  tabulation_hash<8,uint32_t,uint32_t> th8_2 =
    tabulation_hash<8,uint32_t,uint32_t>::draw_one(m,seed);
  tabulation_hash<8,uint32_t,uint32_t> th8_3 =
    tabulation_hash<8,uint32_t,uint32_t>::draw(1, m,seed)[0];

  unsigned int n = 256;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    uint32_t h = (T1[0][(x & 0xff)] ^ T1[1][((x >> 8) & 0xff)] ^ T1[2][((x >> 16) & 0xff)] ^ T1[3][((x >> 24) & 0xff)]) >> 24;
    uint32_t h1 = th8_1(x);
    uint32_t h2 = th8_2(x);
    uint32_t h3 = th8_3(x);
    REQUIRE(h == h1);
    REQUIRE(h == h2);
    REQUIRE(h == h3);
  }  

  auto th8_s = sign_hash<tabulation_hash<8,uint32_t,uint32_t>>::draw_one(seed);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    int h1 = (T1[0][(x & 0xff)] ^ T1[1][((x >> 8) & 0xff)] ^ T1[2][((x >> 16) & 0xff)] ^ T1[3][((x >> 24) & 0xff)]) >> 31;
    h1 = 2*h1 - 1;
    int h2 = th8_s(x);
    REQUIRE(h1 == h2);
    tabulation_hash<8,uint32_t,uint32_t>::sign_t s3 = th8_1.sign(x);
    REQUIRE(h1 == s3);
  }

  ++seed;

  prng = rng(seed);
  uint32_t T2[2][65536];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 65536; ++j)
      T2[i][j] = prng();

  tabulation_hash<16,uint32_t,uint32_t> th16_1(m,&T2[0][0]);
  tabulation_hash<16,uint32_t,uint32_t> th16_2 =
    tabulation_hash<16,uint32_t,uint32_t>::draw_one(m,seed);
  tabulation_hash<16,uint32_t,uint32_t> th16_3 =
    tabulation_hash<16,uint32_t,uint32_t>::draw(1,m,seed)[0];

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    uint32_t h = (T2[0][(x & 0xffff)] ^ T2[1][((x >> 16) & 0xffff)]) >> 24;
    uint32_t h1 = th16_1(x);
    uint32_t h2 = th16_2(x);
    uint32_t h3 = th16_3(x);
    REQUIRE(h == h1);
    REQUIRE(h == h2);
    REQUIRE(h == h3);
  }

  auto th16_s = sign_hash<tabulation_hash<16,uint32_t,uint32_t>>::draw_one(seed);
  for (unsigned int i = 0; i < n; ++i) {
    uint32_t x = prng();
    int h1 = (T2[0][(x & 0xffff)] ^ T2[1][((x >> 16) & 0xffff)]) >> 31;
    h1 = 2*h1 - 1;
    int h2 = th16_s(x);
    REQUIRE(h1 == h2);
    tabulation_hash<16,uint32_t,uint32_t>::sign_t s3 = th16_1.sign(x);
    REQUIRE(h1 == s3);
  }

  seed = 16321;
  prng = rng(seed);
  m = 128;
  n = 1024;
  uint32_t T1a[4][256];
  uint32_t T1b[4][256];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 256; ++j)
      T1a[i][j] = prng();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 256; ++j)
      T1b[i][j] = prng();
  auto t8s = tabulation_hash<8,uint32_t,uint32_t>::draw(2, m,seed);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    uint32_t h1 = (T1a[0][(x & 0xff)] ^ T1a[1][((x >> 8) & 0xff)] ^
		   T1a[2][((x >> 16) & 0xff)] ^ T1a[3][((x >> 24) & 0xff)])
      >> 25;
    uint32_t h2 = (T1b[0][(x & 0xff)] ^ T1b[1][((x >> 8) & 0xff)] ^
		   T1b[2][((x >> 16) & 0xff)] ^ T1b[3][((x >> 24) & 0xff)])
      >> 25;
    REQUIRE(t8s[0](x) == h1);
    REQUIRE(t8s[1](x) == h2);
  }

  auto th8_ss = sign_hash<tabulation_hash<8,uint32_t,uint32_t>>::draw(2,seed);
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    int h1 = (T1a[0][(x & 0xff)] ^ T1a[1][((x >> 8) & 0xff)] ^ T1a[2][((x >> 16) & 0xff)] ^ T1a[3][((x >> 24) & 0xff)]) >> 31;
    h1 = 2*h1 - 1;
    int h2 = (T1b[0][(x & 0xff)] ^ T1b[1][((x >> 8) & 0xff)] ^ T1b[2][((x >> 16) & 0xff)] ^ T1b[3][((x >> 24) & 0xff)]) >> 31;
    h2 = 2*h2 - 1;
    REQUIRE(th8_ss[0](x) == h1);
    REQUIRE(th8_ss[1](x) == h2);
    tabulation_hash<8,uint32_t,uint32_t>::sign_t s3 = t8s[0].sign(x);
    tabulation_hash<8,uint32_t,uint32_t>::sign_t s4 = t8s[1].sign(x);
    REQUIRE(h1 == s3);
    REQUIRE(h2 == s4);
  }
  

  ++seed;
  prng = rng(seed);
  uint32_t T2a[2][65536];
  uint32_t T2b[2][65536];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 65536; ++j)
      T2a[i][j] = prng();
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 65536; ++j)
      T2b[i][j] = prng();
  auto t16s = tabulation_hash<16,uint32_t,uint32_t>::draw(2, m, seed);

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t x = prng();
    uint32_t h1 = (T2a[0][(x & 0xffff)] ^ T2a[1][((x >> 16) & 0xffff)]) >> 25;
    uint32_t h2 = (T2b[0][(x & 0xffff)] ^ T2b[1][((x >> 16) & 0xffff)]) >> 25;
    REQUIRE(t16s[0](x) == h1);
    REQUIRE(t16s[1](x) == h2);
  }

  auto th16_ss = sign_hash<tabulation_hash<16,uint32_t,uint32_t>>::draw(2,seed);
  for (unsigned int i = 0; i < n; ++i) {
    uint32_t x = prng();
    int h1 = (T2a[0][(x & 0xffff)] ^ T2a[1][((x >> 16) & 0xffff)]) >> 31;
    h1 = 2*h1 - 1;
    int h2 = (T2b[0][(x & 0xffff)] ^ T2b[1][((x >> 16) & 0xffff)]) >> 31;
    h2 = 2*h2 - 1;
    REQUIRE(th16_ss[0](x) == h1);
    REQUIRE(th16_ss[1](x) == h2);
    tabulation_hash<16,uint32_t,uint32_t>::sign_t s3 = t16s[0].sign(x);
    tabulation_hash<16,uint32_t,uint32_t>::sign_t s4 = t16s[1].sign(x);
    REQUIRE(h1 == s3);
    REQUIRE(h2 == s4);
  }

}



TEST_CASE( "perf_hash", "[hash][benchmark]" ) {
  auto msh = multiply_shift_hash::draw_one(128,999);
  BENCHMARK("multiply_shift 10000000") {
    volatile multiply_shift_hash::hash_t h;
    for (multiply_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = msh(i);
    }
    return h;
  };

  BENCHMARK("multiply_shift_sign 10000000") {
    volatile multiply_shift_hash::sign_t h;
    for (multiply_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = msh.sign(i);
    }
    return h;
  };
  
  auto mshs = sign_hash<multiply_shift_hash>::draw_one(998);
  BENCHMARK("sign_hash_multiply_shift 10000000") {
    volatile sign_hash<multiply_shift_hash>::hash_t h;
    for (multiply_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = mshs(i);
    }
    return h;
  };
  
  auto mash = multiply_add_shift_hash::draw_one(128,999);
  BENCHMARK("multiply_add_shift 10000000") {
    volatile multiply_add_shift_hash::hash_t h;
    for (multiply_add_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = mash(i);
    }
    return h;
  };

  BENCHMARK("multiply_add_shift_sign 10000000") {
    volatile multiply_add_shift_hash::sign_t h;
    for (multiply_add_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = mash.sign(i);
    }
    return h;
  };

  auto mashs = sign_hash<multiply_add_shift_hash>::draw_one(998);
  BENCHMARK("sign_hash_multiply_add_shift 10000000") {
    volatile sign_hash<multiply_add_shift_hash>::hash_t h;
    for (multiply_add_shift_hash::key_t i = 1; i <= 10000000; ++i) {
      h = mashs(i);
    }
    return h;
  };

  rng prng(999);

  auto th8 = tabulation_hash<8,uint32_t,uint32_t>::draw_one(128,999);
  BENCHMARK("tabulation_hash8 10000000") {
    volatile tabulation_hash<8,uint32_t,uint32_t>::hash_t h;
    for (tabulation_hash<8,uint32_t,uint32_t>::key_t i = 1; i <= 10000000; ++i) {
      h = th8(i);
    }
    return h;
  };

  BENCHMARK("tabulation_hash8_sign 10000000") {
    volatile tabulation_hash<8,uint32_t,uint32_t>::sign_t h;
    for (tabulation_hash<8,uint32_t,uint32_t>::key_t i = 1; i <= 10000000; ++i) {
      h = th8.sign(i);
    }
    return h;
  };

  auto th8s = sign_hash<tabulation_hash<8,uint32_t,uint32_t>>::draw_one(998);
  BENCHMARK("sign_hash_tabulation_hash8 10000000") {
    volatile sign_hash<tabulation_hash<8,uint32_t,uint32_t>>::hash_t h;
    for (sign_hash<tabulation_hash<8,uint32_t,uint32_t>>::key_t i = 1; i <= 10000000; ++i) {
      h = th8s(i);
    }
    return h;
  };

  auto th16 = tabulation_hash<16,uint32_t,uint32_t>::draw_one(128,999);
  BENCHMARK("tabulation_hash16 10000000") {
    volatile tabulation_hash<16,uint32_t,uint32_t>::hash_t h;
    for (tabulation_hash<16,uint32_t,uint32_t>::key_t i = 1; i <= 10000000; ++i) {
      h = th16(i);
    }
    return h;
  };

  BENCHMARK("tabulation_hash16_sign 10000000") {
    volatile tabulation_hash<16,uint32_t,uint32_t>::sign_t h;
    for (tabulation_hash<16,uint32_t,uint32_t>::key_t i = 1; i <= 10000000; ++i) {
      h = th16.sign(i);
    }
    return h;
  };

  auto th16s = sign_hash<tabulation_hash<16,uint32_t,uint32_t>>::draw_one(998);
  BENCHMARK("sign_hash_tabulation_hash16 10000000") {
    volatile sign_hash<tabulation_hash<16,uint32_t,uint32_t>>::hash_t h;
    for (sign_hash<tabulation_hash<16,uint32_t,uint32_t>>::key_t i = 1; i <= 10000000; ++i) {
      h = th16s(i);
    }
    return h;
  };
}



#ifdef COMPMATMUL_USE_FFTW
TEST_CASE( "test_fft", "[fft][unit]" ) {
  using namespace std::complex_literals;
  const int n = 32;
  const int m = 17;
  REQUIRE(n/2+1 == m);
  const double x[n] = {
    0.91527326, 0.07064576, 0.53040597, 0.70546434, 0.36414312, 0.14137265,
    0.50579731, 0.55227775, 0.37729946, 0.95291381, 0.09620142, 0.72017424,
    0.75809317, 0.90013873, 0.46557539, 0.61504573, 0.58568483, 0.25007278,
    0.79962251, 0.22426499, 0.07811993, 0.42820229, 0.72256178, 0.99788856,
    0.1011867 , 0.93158399, 0.6872351 , 0.95747717, 0.14232377, 0.04623507,
    0.1833904 , 0.45529912
  };
  const std::complex<double> X_correct[m] = {
    16.2619711 +0.i        , -1.03093929-0.05840623i,
    -1.03078571+1.11588142i,  0.10465414-2.23170224i,
    1.13343466-0.7142413i  ,  0.89874511-0.48200699i,
    0.60648699-0.68331135i ,  1.02030708+2.99294096i,
    -0.66866564+1.50672682i, -0.09505786-0.3173026i ,
    1.48484497+1.82538006i ,  1.3403471 -0.56288428i,
    0.14009386-0.24196106i , -0.0927295 -0.86637594i,
    3.02934147+1.79195727i ,  0.49138066+0.28645587i,
    -1.63614286+0.i
  };
  std::complex<double> X[m];
  double xi[n];

  fft_transformer fft(n);
  fft.fwd(X, x);
  fft.inv(xi, X);
  
  for (int i = 0; i < m; ++i) {
    REQUIRE_THAT(X[i].real(), Catch::Matchers::WithinAbs(X_correct[i].real(), 1e-6));
    REQUIRE_THAT(X[i].imag(), Catch::Matchers::WithinAbs(X_correct[i].imag(), 1e-6));
  }
  
  for (int i = 0; i < n; ++i) {
    REQUIRE_THAT(xi[i], Catch::Matchers::WithinAbs(x[i], 1e-6));
  }
}
#endif // COMPMATMUL_USE_FFTW



#ifdef COMPMATMUL_USE_FXT
TEST_CASE( "test_fwht", "[fwht][unit]" ) {
  const unsigned int n = 32;
  const double x[n] = {
    0.77956498, 0.24374313, 0.66121831, 0.89375154, 0.95073372, 0.11784131,
    0.50483831, 0.44331508, 0.41599355, 0.5194395, 0.26875662, 0.87481817,
    0.9575786 , 0.33804587, 0.49530665, 0.73303024, 0.42873154, 0.87784849,
    0.19554847, 0.18266582, 0.78896532, 0.05496766, 0.40847289, 0.30497385,
    0.67434276, 0.70394152, 0.45398693, 0.02708591, 0.01455404, 0.69647227,
    0.69386635, 0.45950925
  };
  const double X_correct[n] = {
    1.61639087e+01,  1.22100944e+00,  9.61619873e-01,  1.69532008e+00,
    2.38965842e-01, -2.11131125e+00,  1.20992754e+00, -9.90377533e-01,
    -4.89547794e-01,  1.97692390e+00,  3.33603927e-01,  1.72112646e+00,
    1.13896243e+00, -1.61840476e+00, -9.18335302e-01, -1.20064715e+00,
    2.23204248e+00,  5.19002383e-01, -2.06580837e+00,  4.10387227e+00,
    -5.77377406e-03, -1.25357604e+00, -3.06495295e+00,  2.75067875e-01,
    4.73622156e-01,  1.07388134e+00, -8.74375765e-01, -1.36142179e+00,
    8.74043671e-01,  2.61878400e+00,  1.65027151e-01,  1.90390037e+00
  };
  double X1[n];
  double X2[n];
  double xi1[n];
  double xi2[n];

  fwht_transformer fwht(n);
  fwht.fwd(X1, x);
  fwht.inv(xi1, X1);
  
  
  unsigned long ldn = array::ilog2(n);

  REQUIRE(ldn == 5);
  
  memcpy(X2, x, sizeof(double)*n);
  walsh_wak(X2, ldn);
  memcpy(xi2, X2, sizeof(double)*n);
  walsh_wak(xi2, ldn);
  for (unsigned int i = 0; i < n; ++i)
    xi2[i] /= n;

  for (unsigned int i = 0; i < n; ++i) {
    REQUIRE_THAT(X1[i], Catch::Matchers::WithinAbs(X_correct[i], 1e-6));
    REQUIRE_THAT(X2[i], Catch::Matchers::WithinAbs(X_correct[i], 1e-6));
  }

  for (unsigned int i = 0; i < n; ++i) {
    REQUIRE_THAT(xi1[i], Catch::Matchers::WithinAbs(x[i], 1e-6));
    REQUIRE_THAT(xi2[i], Catch::Matchers::WithinAbs(x[i], 1e-6));
  }
}
#endif // COMPMATMUL_USE_FXT



#ifdef COMPMATMUL_USE_FFTW
TEST_CASE( "perf_fft", "[fft][benchmark]" ) {
  const int n = 524288;
  std::vector<double> x(n);
  std::vector<double> xi(n);
  std::vector<std::complex<double>> X(n/2+1);
  drng rng(123);
  for (int i = 0; i < n; ++i) {
    x[i] = rng();
  }

  fft_transformer fft(n);
  BENCHMARK("fft_fwd 524288") {
    fft.fwd(&X[0], &x[0]);
  };
  BENCHMARK("fft_inv 524288") {
    fft.inv(&xi[0], &X[0]);
  };
}
#endif // COMPMATMUL_USE_FFTW



#ifdef COMPMATMUL_USE_FXT
TEST_CASE( "perf_fwht", "[fwht][benchmark]" ) {
  const int n = 524288;
  std::vector<double> x(n);
  std::vector<double> xi(n);
  std::vector<double> X(n);
  drng rng(123);
  for (int i = 0; i < n; ++i) {
    x[i] = rng();
  }

  fwht_transformer fwht(n);
  BENCHMARK("fwht_fwd 524288") {
    fwht.fwd(&X[0], &x[0]);
  };
  BENCHMARK("fwht_inv 524288") {
    fwht.inv(&xi[0], &X[0]);
  };
}
#endif // COMPMATMUL_USE_FXT



TEST_CASE( "test_gemm", "[matmul][unit]" ) {
  using namespace compmatmul::array;
  double* AT = allocate<double>(4*4);
  double* B = allocate<double>(4*8);
  double* C = allocate<double>(4*4);

  mov(4*4, AT, AT_4x4);
  mov(4*4, B, B_4x4);
  matmul_gemm(4, 4, 4, C, AT, B);
  REQUIRE(equal(4*4, C, C_4x4x4));

  mov(4*2, AT, AT_2x4);
  mov(4*2, B, B_2x4);
  matmul_gemm(4, 2, 4, C, AT, B);
  REQUIRE(equal(4*4, C, C_4x2x4));

  mov(2*4, AT, AT_4x2);
  mov(2*4, B, B_4x2);
  matmul_gemm(2, 4, 2, C, AT, B);
  REQUIRE(equal(2*2, C, C_2x4x2));

  mov(4*8, B, B_4x8);
  matmul_gemm(2, 4, 8, C, AT, B);
  REQUIRE(equal(2*8, C, C_2x4x8));  
  
  deallocate(AT);
  deallocate(B);
  deallocate(C);
}



TEST_CASE( "perf_matmul", "[matmul][benchmark]" ) {
  int n = 4096;
#if defined(COMPMATMUL_USE_FFTW) || defined(COMPMATMUL_USE_FXT)
  int d = 12;
  int b = 4096;
#endif // defined(COMPMATMUL_USE_FFTW) || defined(COMPMATMUL_USE_FXT)
  double* AT = allocate<double>(n*n);
  double* B = allocate<double>(n*n);
  double* C = allocate<double>(n*n);

  drng rng(123);

  for (double* x : {AT, B, C}) {
    double* it = x;
    while (it < x + n*n)
      *it++ = rng();
  }

  BENCHMARK("gemm 4096") {
    matmul_gemm(n, n, n, C, AT, B);
  };

#ifdef COMPMATMUL_USE_FFTW
  BENCHMARK("new fft 4096 b=4096 d=12") {
    matmul_fft<multiply_add_shift_hash>(n, n, n, C, AT, B, d, b);
  };
#endif // COMPMATMUL_USE_FFTW

#ifdef COMPMATMUL_USE_FXT
  BENCHMARK("new fwht 4096 b=4096 d=12") {
    matmul_fwht<multiply_add_shift_hash>(n, n, n, C, AT, B, d, b);
  };
#endif // COMPMATMUL_USE_FXT

  deallocate(AT);
  deallocate(B);
  deallocate(C);  
}



#ifdef COMPMATMUL_USE_FFTW
TEST_CASE( "test_matmul_fft", "[matmul][unit]" ) {
  multiply_add_shift_hash h = multiply_add_shift_hash::draw_one(32,8876);
  uint64_t a = UINT64_C(15506110424104283675);
  uint64_t b = UINT64_C(3138147583905456810);
  REQUIRE(h.get_a() == a);
  REQUIRE(h.get_b() == b);

  std::vector<uint64_t> ab(8);
  ab[0] = a;
  ab[1] = b;
 
  rng rng(32149);
  for (int i = 0; i < 100; ++i) {
    uint32_t x = rng();
    uint32_t hash = ((a*x+b) >> 59);
    int sign = (hash >> 4) ? 1 : -1;
    REQUIRE(h(x) == hash);
    REQUIRE(h.sign(x) == sign);
  }

  matmul_fft_sketch<multiply_add_shift_hash> sketch(2,2,3483);

  REQUIRE(sketch.h1.size() == 2);
  REQUIRE(sketch.h1[0].get_a() == UINT64_C(1312110179027324775));
  REQUIRE(sketch.h1[0].get_b() == UINT64_C(15465668779316680995));
  REQUIRE(sketch.h1[1].get_a() == UINT64_C(4555306672462241870));
  REQUIRE(sketch.h1[1].get_b() == UINT64_C(12288159137770073612));

  REQUIRE(sketch.h2.size() == 2);
  REQUIRE(sketch.h2[0].get_a() == UINT64_C(4329212686603688483));
  REQUIRE(sketch.h2[0].get_b() == UINT64_C(10307990599335447505));
  REQUIRE(sketch.h2[1].get_a() == UINT64_C(12597734872139453857));
  REQUIRE(sketch.h2[1].get_b() == UINT64_C(15119207240207437602));

  REQUIRE(sketch.s1.size() == 2);
  REQUIRE(sketch.s1[0].get_a() == UINT64_C(11899074044546628492));
  REQUIRE(sketch.s1[0].get_b() == UINT64_C(298345645907162146));
  REQUIRE(sketch.s1[1].get_a() == UINT64_C(8628211646415106536));
  REQUIRE(sketch.s1[1].get_b() == UINT64_C(12647867801867038929));

  REQUIRE(sketch.s2.size() == 2);
  REQUIRE(sketch.s2[0].get_a() == UINT64_C(7008367753992221008));
  REQUIRE(sketch.s2[0].get_b() == UINT64_C(16948937881964684143));
  REQUIRE(sketch.s2[1].get_a() == UINT64_C(10319872211339334476));
  REQUIRE(sketch.s2[1].get_b() == UINT64_C(13096518497071785733));

  for (auto [d,b] : std::vector<std::pair<int,int>> { {1,2}, {1,8}, {1, 32}, {3,2}, {3,8}, {5,32} } ) {
    const uint64_t seed = 25926;
    std::vector<unsigned long> h1_ab(2*d);
    std::vector<unsigned long> h2_ab(2*d);
    std::vector<unsigned long> s1_ab(2*d);
    std::vector<unsigned long> s2_ab(2*d);
    sketch = matmul_fft_sketch<multiply_add_shift_hash>(d, b, seed);
    for (size_t i = 0; i < static_cast<size_t>(d); ++i) {
      h1_ab[2*i] = sketch.h1[i].get_a();
      h1_ab[2*i+1] = sketch.h1[i].get_b();
      h2_ab[2*i] = sketch.h2[i].get_a();
      h2_ab[2*i+1] = sketch.h2[i].get_b();
      s1_ab[2*i] = sketch.s1[i].get_a();
      s1_ab[2*i+1] = sketch.s1[i].get_b();
      s2_ab[2*i] = sketch.s2[i].get_a();
      s2_ab[2*i+1] = sketch.s2[i].get_b();
    }

    std::vector<unsigned long*> old_h1(d);
    std::vector<unsigned long*> old_h2(d);
    std::vector<unsigned long*> old_s1(d);
    std::vector<unsigned long*> old_s2(d);
    for (int i = 0; i < d; ++i) {
      old_h1[i] = &h1_ab[2*i];
      old_h2[i] = &h2_ab[2*i];
      old_s1[i] = &s1_ab[2*i];
      old_s2[i] = &s2_ab[2*i];
    }
    
    double* AT = array::allocate<double>(4*4);
    double* B = array::allocate<double>(4*8);
    double* C = array::allocate<double>(4*4);

    double* xs = array::allocate<double>(d);

    memcpy(AT,AT_4x4,4*4*sizeof(double));
    memcpy(B,B_4x4,4*4*sizeof(double));

    sketch = matmul_compressed_product_fft<multiply_add_shift_hash>(4, 4, 4, AT, B, d, b, seed);

    for (int i = 0; i < d; ++i) {
      REQUIRE(sketch.h1[i].get_a() == h1_ab[2*i]);
      REQUIRE(sketch.h1[i].get_b() == h1_ab[2*i+1]);
      REQUIRE(sketch.h2[i].get_a() == h2_ab[2*i]);
      REQUIRE(sketch.h2[i].get_b() == h2_ab[2*i+1]);
      REQUIRE(sketch.s1[i].get_a() == s1_ab[2*i]);
      REQUIRE(sketch.s1[i].get_b() == s1_ab[2*i+1]);
      REQUIRE(sketch.s2[i].get_a() == s2_ab[2*i]);
      REQUIRE(sketch.s2[i].get_b() == s2_ab[2*i+1]);
    }

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        matmul_decompress_fft(4, C, i, j, sketch, xs);
      }
    }
    int n = 4;

    if (d > 1 && b >= 32) {
      REQUIRE( array::almost_equal(n*n, C, C_4x4x4) );
    }

    if (d > 1 && b >= 32) {
      int n_a = 4;
      int n_inner = 4;
      int n_b = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*4, AT, AT_4x4);
      mov(4*4, B, B_4x4);
      matmul_fft<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(almost_equal(n_a*n_b, C, C_4x4x4));

      n_inner = 2;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(2*4, AT, AT_2x4);
      mov(2*4, B, B_2x4);
      matmul_fft<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(almost_equal(n_a*n_b, C, C_4x2x4));

      n_a = n_b = 2;
      n_inner = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*2, AT, AT_4x2);
      mov(4*2, B, B_4x2);
      matmul_fft<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(almost_equal(n_a*n_b, C, C_2x4x2));

      n_a = 2; 
      n_b = 8;
      n_inner = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*2, AT, AT_4x2);
      mov(4*8, B, B_4x8);
      matmul_fft<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(almost_equal(n_a*n_b, C, C_2x4x8));
    }
    array::deallocate(AT);
    array::deallocate(B);
    array::deallocate(C);
    array::deallocate(xs);
  }

  uint64_t seed = 15954;
  int n = 16;
  double* AT1 = array::allocate<double>(n*n);
  double* AT2 = array::allocate<double>(n*n);
  double* AT3 = array::allocate<double>(n*n);
  double* AT4 = array::allocate<double>(n*n);
  double* AT5 = array::allocate<double>(n*n);
  double* B1 = array::allocate<double>(n*n);
  double* B2 = array::allocate<double>(n*n);
  double* B3 = array::allocate<double>(n*n);
  double* B4 = array::allocate<double>(n*n);
  double* B5 = array::allocate<double>(n*n);
  double* C1 = array::allocate<double>(n*n);
  double* C2 = array::allocate<double>(n*n);
  double* C3 = array::allocate<double>(n*n);
  double* C4 = array::allocate<double>(n*n);
  double* C5 = array::allocate<double>(n*n);

  drng drng(seed);
  for (int i = 0; i < n*n; ++i) {
    AT1[i] = AT2[i] = AT3[i] = AT4[i] = AT5[i] = drng();
    B1[i] = B2[i] = B3[i] = B4[i] = B5[i] = drng();
  }

  REQUIRE( equal(n*n, AT1, AT2) );
  REQUIRE( equal(n*n, AT2, AT3) );
  REQUIRE( equal(n*n, AT3, AT4) );
  REQUIRE( equal(n*n, AT4, AT5) );
  REQUIRE( equal(n*n, B1, B2) );
  REQUIRE( equal(n*n, B2, B3) );
  REQUIRE( equal(n*n, B3, B4) );
  REQUIRE( equal(n*n, B4, B5) );

  matmul_gemm(n, n, n, C1, AT1, B1);
  matmul_fft<multiply_add_shift_hash>(n, n, n, C2, AT2, B2, 5, 1024, seed+1);
  matmul_fft<multiply_shift_hash>(n, n, n, C3, AT3, B3, 5, 1024, seed+2);
  matmul_fft<tabulation_hash<8, uint32_t, uint32_t>>(n, n, n, C4, AT4, B4, 11, 2048, seed+3);
  matmul_fft<tabulation_hash<16, uint32_t, uint32_t>>(n, n, n, C5, AT5, B5, 11, 1024, seed+4);

  REQUIRE( almost_equal(n*n, C1, C2, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C3, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C4, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C5, 1e-12) );

  array::deallocate(AT1);
  array::deallocate(AT2);
  array::deallocate(AT3);
  array::deallocate(AT4);
  array::deallocate(AT5);
  array::deallocate(B1);
  array::deallocate(B2);
  array::deallocate(B3);
  array::deallocate(B4);
  array::deallocate(B5);
  array::deallocate(C1);
  array::deallocate(C2);
  array::deallocate(C3);
  array::deallocate(C4);
  array::deallocate(C5);
}
#endif // COMPMATMUL_USE_FFTW



#ifdef COMPMATMUL_USE_FFTW
TEST_CASE( "perf_matmul_fft", "[matmul][benchmark]" ) {
  int n = 512;
  double* AT = array::allocate<double>(n*n);
  double* B = array::allocate<double>(n*n);
  double* C = array::allocate<double>(n*n);

  uint64_t seed = 29123;
  drng rng(seed);
  for (int i = 0; i < n*n; ++i) {
    AT[i] = rng();
    B[i] = rng();
  }

  int b = 1024;
  int d = 13;
  

  BENCHMARK("matmul_fft_multiply_shift 512x512x512 b=1024 d=13") {
    matmul_fft<multiply_shift_hash>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fft_multiply_add_shift 512x512x512 b=1024 d=13") {
    matmul_fft<multiply_add_shift_hash>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fft_tabulation8 512x512x512 b=1024 d=13") {
    matmul_fft<tabulation_hash<8,uint32_t,uint32_t>>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fft_tabulation16 512x512x512 b=1024 d=13") {
    matmul_fft<tabulation_hash<16,uint32_t,uint32_t>>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  array::deallocate(AT);
  array::deallocate(B);
  array::deallocate(C);
}
#endif // COMPMATMUL_USE_FFTW



#ifdef COMPMATMUL_USE_FXT
TEST_CASE( "test_matmul_fwht", "[matmul][unit]" ) {
  for (auto [d,b] : std::vector<std::pair<int,int>> { {1,2}, {1,8}, {1, 32}, {3,2}, {3,8}, {7,32} } ) {
    const uint64_t seed = 25926;
    std::vector<unsigned long> h1_ab(2*d);
    std::vector<unsigned long> h2_ab(2*d);
    std::vector<unsigned long> s1_ab(2*d);
    std::vector<unsigned long> s2_ab(2*d);
    matmul_fwht_sketch<multiply_add_shift_hash> sketch(d, b, seed);
    for (size_t i = 0; i < static_cast<size_t>(d); ++i) {
      h1_ab[2*i] = sketch.h1[i].get_a();
      h1_ab[2*i+1] = sketch.h1[i].get_b();
      h2_ab[2*i] = sketch.h2[i].get_a();
      h2_ab[2*i+1] = sketch.h2[i].get_b();
      s1_ab[2*i] = sketch.s1[i].get_a();
      s1_ab[2*i+1] = sketch.s1[i].get_b();
      s2_ab[2*i] = sketch.s2[i].get_a();
      s2_ab[2*i+1] = sketch.s2[i].get_b();
    }

    std::vector<unsigned long*> old_h1(d);
    std::vector<unsigned long*> old_h2(d);
    std::vector<unsigned long*> old_s1(d);
    std::vector<unsigned long*> old_s2(d);
    for (int i = 0; i < d; ++i) {
      old_h1[i] = &h1_ab[2*i];
      old_h2[i] = &h2_ab[2*i];
      old_s1[i] = &s1_ab[2*i];
      old_s2[i] = &s2_ab[2*i];
    }
    
    double* AT = array::allocate<double>(4*4);
    double* B = array::allocate<double>(4*8);
    double* C = array::allocate<double>(4*4);

    double* xs = array::allocate<double>(d);
    array::mov(4*4, AT, AT_4x4);
    array::mov(4*4, B, B_4x4);
    sketch = matmul_compressed_product_fwht<multiply_add_shift_hash>(4, 4, 4, AT, B, d, b, seed);
    for (int i = 0; i < d; ++i) {
      REQUIRE(sketch.h1[i].get_a() == h1_ab[2*i]);
      REQUIRE(sketch.h1[i].get_b() == h1_ab[2*i+1]);
      REQUIRE(sketch.h2[i].get_a() == h2_ab[2*i]);
      REQUIRE(sketch.h2[i].get_b() == h2_ab[2*i+1]);
      REQUIRE(sketch.s1[i].get_a() == s1_ab[2*i]);
      REQUIRE(sketch.s1[i].get_b() == s1_ab[2*i+1]);
      REQUIRE(sketch.s2[i].get_a() == s2_ab[2*i]);
      REQUIRE(sketch.s2[i].get_b() == s2_ab[2*i+1]);
    }

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        matmul_decompress_fwht(4, C, i, j, sketch, xs);
      }
    }

    int n = 4;

    if (d > 1 && b >= 32) {
      REQUIRE( array::equal(n*n, C, C_4x4x4) );
    }

    if (d > 1 && b >= 32) {
      int n_a = 4;
      int n_inner = 4;
      int n_b = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*4,AT,AT_4x4);
      mov(4*4,B,B_4x4);
      matmul_fwht<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(equal(n_a*n_b, C, C_4x4x4));

      n_inner = 2;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(2*4,AT,AT_2x4);
      mov(2*4,B,B_2x4);
      matmul_fwht<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(equal(n_a*n_b, C, C_4x2x4));

      n_a = n_b = 2;
      n_inner = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*2,AT,AT_4x2);
      mov(4*2,B,B_4x2);
      matmul_fwht<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(equal(n_a*n_b, C, C_2x4x2));

      n_a = 2; 
      n_b = 8;
      n_inner = 4;
      memset(C, 0, n_a*n_b*sizeof(double));
      mov(4*2,AT,AT_4x2);
      mov(4*8,B,B_4x8);
      matmul_fwht<multiply_add_shift_hash>(n_a, n_inner, n_b, C, AT, B, d, b, seed);
      REQUIRE(equal(n_a*n_b, C, C_2x4x8));
    }
    array::deallocate(AT);
    array::deallocate(B);
    array::deallocate(C);
    array::deallocate(xs);    
  }

  uint64_t seed = 15954;
  int n = 16;
  double* AT1 = array::allocate<double>(n*n);
  double* AT2 = array::allocate<double>(n*n);
  double* AT3 = array::allocate<double>(n*n);
  double* AT4 = array::allocate<double>(n*n);
  double* AT5 = array::allocate<double>(n*n);
  double* B1 = array::allocate<double>(n*n);
  double* B2 = array::allocate<double>(n*n);
  double* B3 = array::allocate<double>(n*n);
  double* B4 = array::allocate<double>(n*n);
  double* B5 = array::allocate<double>(n*n);
  double* C1 = array::allocate<double>(n*n);
  double* C2 = array::allocate<double>(n*n);
  double* C3 = array::allocate<double>(n*n);
  double* C4 = array::allocate<double>(n*n);
  double* C5 = array::allocate<double>(n*n);

  drng drng(seed);
  for (int i = 0; i < n*n; ++i) {
    AT1[i] = AT2[i] = AT3[i] = AT4[i] = AT5[i] = drng();
    B1[i] = B2[i] = B3[i] = B4[i] = B5[i] = drng();
  }

  REQUIRE( equal(n*n, AT1, AT2) );
  REQUIRE( equal(n*n, AT2, AT3) );
  REQUIRE( equal(n*n, AT3, AT4) );
  REQUIRE( equal(n*n, AT4, AT5) );
  REQUIRE( equal(n*n, B1, B2) );
  REQUIRE( equal(n*n, B2, B3) );
  REQUIRE( equal(n*n, B3, B4) );
  REQUIRE( equal(n*n, B4, B5) );

  matmul_gemm(n, n, n, C1, AT1, B1);
  matmul_fwht<multiply_add_shift_hash>(n, n, n, C2, AT2, B2, 5, 2048, seed+1);
  matmul_fwht<multiply_shift_hash>(n, n, n, C3, AT3, B3, 5, 2048, seed+2);
  matmul_fwht<tabulation_hash<8, uint32_t, uint32_t>>(n, n, n, C4, AT4, B4, 11, 2048, seed+3);
  matmul_fwht<tabulation_hash<16, uint32_t, uint32_t>>(n, n, n, C5, AT5, B5, 11, 1024, seed+4);

  REQUIRE( almost_equal(n*n, C1, C2, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C3, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C4, 1e-12) );
  REQUIRE( almost_equal(n*n, C1, C5, 1e-12) );

  array::deallocate(AT1);
  array::deallocate(AT2);
  array::deallocate(AT3);
  array::deallocate(AT4);
  array::deallocate(AT5);
  array::deallocate(B1);
  array::deallocate(B2);
  array::deallocate(B3);
  array::deallocate(B4);
  array::deallocate(B5);
  array::deallocate(C1);
  array::deallocate(C2);
  array::deallocate(C3);
  array::deallocate(C4);
  array::deallocate(C5);
}
#endif // COMPMATMUL_USE_FXT



#ifdef COMPMATMUL_USE_FXT
TEST_CASE( "perf_matmul_fwht", "[matmul][benchmark]" ) {
  int n = 512;
  double* AT = array::allocate<double>(n*n);
  double* B = array::allocate<double>(n*n);
  double* C = array::allocate<double>(n*n);

  uint64_t seed = 29123;
  drng rng(seed);
  for (int i = 0; i < n*n; ++i) {
    AT[i] = rng();
    B[i] = rng();
  }

  int b = 1024;
  int d = 13;
  

  BENCHMARK("matmul_fwht_multiply_shift 512x512x512 b=1024 d=13") {
    matmul_fwht<multiply_shift_hash>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fwht_multiply_add_shift 512x512x512 b=1024 d=13") {
    matmul_fwht<multiply_add_shift_hash>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fwht_tabulation8 512x512x512 b=1024 d=13") {
    matmul_fwht<tabulation_hash<8,uint32_t,uint32_t>>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  BENCHMARK("matmul_fwht_tabulation16 512x512x512 b=1024 d=13") {
    matmul_fwht<tabulation_hash<16,uint32_t,uint32_t>>(n, n, n, C, AT, B, d, b, seed + 1);
  };

  array::deallocate(AT);
  array::deallocate(B);
  array::deallocate(C);
}
#endif // COMPMATMUL_USE_FXT
