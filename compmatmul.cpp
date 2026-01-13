#include "matmul_fwht.hpp"
#include "matmul_fft.hpp"
#include "hashing.hpp"
#include "array.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;
using namespace compmatmul;

static void check_array_arguments(py::array C, py::array AT, py::array B) {
    py::buffer_info C_info = C.request();
    py::buffer_info AT_info = AT.request();
    py::buffer_info B_info = B.request();

    auto C_dtype = C.dtype();
    auto AT_dtype = AT.dtype();
    auto B_dtype = B.dtype();

    if (!C_dtype.is(py::dtype::of<double>())) {
        throw std::invalid_argument("compmatmul only supports double precision floats for now");
    }
    if (!AT_dtype.is(py::dtype::of<double>())) {
        throw std::invalid_argument("compmatmul only supports double precision floats for now");
    }
    if (!B_dtype.is(py::dtype::of<double>())) {
        throw std::invalid_argument("compmatmul only supports double precision floats for now");
    }

    if (C_info.ndim != 2) {
        throw std::invalid_argument("C must have ndim=2 (got " + 
          std::to_string(C_info.ndim) + ")");
    }
    if (AT_info.ndim != 2) {
        throw std::invalid_argument("AT must have ndim=2 (got " + 
          std::to_string(AT_info.ndim) + ")");
    }
    if (B_info.ndim != 2) {
        throw std::invalid_argument("B must have ndim=2 (got " + 
          std::to_string(B_info.ndim) + ")");
    }

    const auto C_CONTIGUOUS = py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_;
    if ((C.flags() & C_CONTIGUOUS) != C_CONTIGUOUS)
        throw std::invalid_argument("The arrays must be C-contiguous");

    if ((AT.flags() & C_CONTIGUOUS) != C_CONTIGUOUS)
        throw std::invalid_argument("The arrays must be C-contiguous");

    if ((B.flags() & C_CONTIGUOUS) != C_CONTIGUOUS)
        throw std::invalid_argument("The arrays must be C-contiguous");

    int p,q,r;
    q = AT.shape(0);
    p = AT.shape(1);
    r = B.shape(1);

    if (B.shape(0) != q || C.shape(0) != p || C.shape(1) != r) {
        throw std::invalid_argument("Shape mismatch! Note that AT is assumed "
            "to be transposed");
    }

    if (reinterpret_cast<uintptr_t>(AT.data()) % 64 != 0) {
        throw std::invalid_argument("AT is not aligned at a 64-byte boundary!");
    }
    if (reinterpret_cast<uintptr_t>(B.data()) % 64 != 0) {
        throw std::invalid_argument("B is not aligned at a 64-byte boundary!");
    }
    if (reinterpret_cast<uintptr_t>(C.data()) % 64 != 0) {
        throw std::invalid_argument("C is not aligned at a 64-byte boundary!");
    }
}

static void gemm(py::array C, py::array AT, py::array B) {
    int p,q,r;
    check_array_arguments(C,AT,B);
    q = AT.shape(0);
    p = AT.shape(1);
    r = B.shape(1);

    compmatmul::array::matmul_gemm<double>(p, q, r, static_cast<double*>(C.mutable_data()),
        static_cast<const double*>(AT.data()), static_cast<const double*>(B.data()));
}

static void check_other_arguments(int d, int b, 
    std::optional<std::string> hash) {
    if (b < 2 || (b&(b-1)) != 0) {
        throw std::invalid_argument(std::format("b must be power of 2 greater "
        "than 1 (got {})", b));
    }
    if (d < 1) {
        throw std::invalid_argument(std::format("d must be positive (got {})", 
            d));
    }
    if (hash) {
        if (*hash != "multiply shift" && *hash != "multiply add shift" && 
            *hash != "tabulation8" && *hash != "tabulation16")
            throw std::invalid_argument(std::format("Invalid hash `{}' "
                "provided; hash must be one of `multiply shift', "
                "`multiply add shift', `tabulation8', or `tabulation16'", 
                *hash));
    }
}

#ifdef COMPMATMUL_USE_FFTW
static void matmul_fft(py::array C, py::array AT, py::array B, int d, int b, 
    std::optional<uint64_t> seed = std::nullopt, 
    std::optional<std::string> hash = std::nullopt) {
    
    check_array_arguments(C,AT,B);
    check_other_arguments(d,b,hash);

    int p,q,r;
    q = AT.shape(0);
    p = AT.shape(1);
    r = B.shape(1);

    if (!hash || *hash == "multiply add shift")   
        matmul_fft<multiply_add_shift_hash>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "multiply shift")   
        matmul_fft<multiply_shift_hash>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "tabulation8")   
        matmul_fft<tabulation_hash<8,uint32_t,uint32_t>>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "tabulation16")   
        matmul_fft<tabulation_hash<16,uint32_t,uint32_t>>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else
        throw std::runtime_error("Tried to use an invalid hash");
}
#endif // COMPMATMUL_USE_FFTW

#ifdef COMPMATMUL_USE_FXT
static void matmul_fwht(py::array C, py::array AT, py::array B, int d, int b, 
    std::optional<uint64_t> seed = std::nullopt, 
    std::optional<std::string> hash = std::nullopt) {
    int p,q,r;
    check_array_arguments(C,AT,B);
    check_other_arguments(d,b,hash);
    q = AT.shape(0);
    p = AT.shape(1);
    r = B.shape(1);
    if (!hash || *hash == "multiply add shift") 
        matmul_fwht<multiply_add_shift_hash>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "multiply shift")
        matmul_fwht<multiply_shift_hash>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "tabulation8")
        matmul_fwht<tabulation_hash<8,uint32_t,uint32_t>>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else if (*hash == "tabulation16")
        matmul_fwht<tabulation_hash<16,uint32_t,uint32_t>>(p,q,r,
            static_cast<double*>(C.mutable_data()), 
            static_cast<const double*>(AT.data()), 
            static_cast<const double*>(B.data()),
            d, b, seed);
    else
        throw std::runtime_error("Tried to use an invalid hash");
}
#endif // COMPMATMUL_USE_FXT


PYBIND11_MODULE(compmatmul, m) {
    m.doc() = "Compressed Matrix Multiplication Python bindings";

    m.def("gemm", &gemm, "Computes C = A^T B using the DGEMM function of the "
        "underlying BLAS implementation",
        py::arg("C"), py::arg("AT"), py::arg("B"));

#ifdef COMPMATMUL_USE_FFTW
    m.def("matmul_fft", &::matmul_fft, "Computes C = A^T using compressed "
        "matrix multiplication with FFT",
        py::arg("C"), py::arg("AT"), py::arg("B"), py::arg("d"), py::arg("b"), 
        py::arg("seed") = py::none(), py::arg("hash") = py::none());
#endif // COMPMATMUL_USE_FFTW

#ifdef COMPMATMUL_USE_FXT
    m.def("matmul_fwht", &::matmul_fwht, "Computes C = A^T using compressed "
        "matrix multiplication with FWHT",
        py::arg("C"), py::arg("AT"), py::arg("B"), py::arg("d"), py::arg("b"), 
        py::arg("seed") = py::none(), py::arg("hash") = py::none());
#endif // COMPMATMUL_USE_FXT
}
