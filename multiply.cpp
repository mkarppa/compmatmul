#include "hashing.hpp"
#ifdef COMPMATMUL_USE_FXT
#include "matmul_fwht.hpp"
#endif // COMPMATMUL_USE_FXT
#ifdef COMPMATMUL_USE_FFTW
#include "matmul_fft.hpp"
#endif // COMPMATMUL_USE_FFTW
#include "array.hpp"
#include <cnpy.h>
#include <tclap/CmdLine.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <random>

// a test executable to simply compute the product using one of the various
// algorithms
// Matti Karppa 2024, 2025


static void printMatrix(FILE* f, const double* M, int rows, int cols) {
  const double* p = M;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      fprintf(f, "%s%2.0f", (j > 0 ? " " : ""), *p++);
    }
    fprintf(f, "\n");
  }
}

static void printMatrixT(FILE* f, const double* M, int rows, int cols) {
  for (int j = 0; j < cols; ++j) {
    const double* p = M + j;
    for (int i = 0; i < rows; ++i) {
      fprintf(f, "%s%2.0f", (i > 0 ? " " : ""), *p);
      p += cols;
    }
    fprintf(f, "\n");
  }
}


int main(int argc, char* argv[]) {
  try {
    TCLAP::CmdLine cmd("A simple matrix multiplication program. "
		       "Computes C=AB^T, that is, the right-hand operand is "
		       "assumed to be transposed.");
    TCLAP::UnlabeledValueArg<std::string> aFile("<A.npy>",
						".npy file containing the data "
						"for the left-hand side argument",
						true, "", "filename");
    TCLAP::UnlabeledValueArg<std::string> bFile("<B.npy>",
						".npy file containing the data "
						"for the right-hand side argument",
						true, "", "filename");
    TCLAP::UnlabeledValueArg<std::string> cFile("<C.npy>",
						"filename for the .npy file "
						"where the result will be stored",
						true, "", "filename");
    TCLAP::SwitchArg forceSwitch("f","force",
				 "Overwrite the output file (if it exists).");
    TCLAP::ValueArg<int> bArg("b", "buckets", "number of buckets per sketch "
			      "(the parameter b, must be a power of 2)",
			      false, -1, "int");
    TCLAP::ValueArg<int> dArg("d", "sketches", "number of sketches "
			      "(the parameter d, must be positive)",
			      false, -1, "int");
    std::vector<std::string> algoStrings { "gemm" };
#ifdef COMPMATMUL_USE_FFTW
    algoStrings.push_back("fft");
#endif // COMPMATMUL_USE_FFTW
#ifdef COMPMATMUL_USE_FXT
    algoStrings.push_back("fwht");
#endif // COMPMATMUL_USE_FXT
    TCLAP::ValuesConstraint<std::string> algoConstraint(algoStrings);
    TCLAP::ValueArg<std::string> algoArg("a", "algorithm", "Which algorithm to use", false,
					 "gemm", &algoConstraint);
    TCLAP::MultiSwitchArg verboseArg("v","verbose","Increase verbosity level");


    cmd.add(aFile);
    cmd.add(bFile);
    cmd.add(cFile);
    cmd.add(forceSwitch);
    cmd.add(algoArg);
    cmd.add(bArg);
    cmd.add(dArg);
    cmd.add(verboseArg);
    cmd.parse(argc, argv);

    if (std::filesystem::exists(std::filesystem::path(cFile.getValue())) && !forceSwitch.getValue()) {
      std::cerr << argv[0] << ": error: file `" << cFile.getValue()
		<< "' already exists!" << std::endl;
      return EXIT_FAILURE;
    }

    int verbose = verboseArg.getValue();

    int bParam = bArg.getValue();
    if (algoArg.getValue() != "gemm") {
      if (!bArg.isSet())
	throw std::runtime_error("Got algorithm `" + algoArg.getValue() +
				 "' but the parameter b is not set!");
      if (bParam < 1 || (bParam & (bParam-1)) != 0) 
	throw std::runtime_error("The parameter b must be a positive power of "
				 "2 (got " + std::to_string(bParam) + ")");
    }

    int dParam = dArg.getValue();
    if (algoArg.getValue() != "gemm") {
      if (!dArg.isSet())
	throw std::runtime_error("Got algorithm `" + algoArg.getValue() +
				 "' but the parameter d is not set!");
      if (dParam < 1) 
	throw std::runtime_error("The parameter d must be positive (got " +
				 std::to_string(dParam) + ")");
    }


    // we don't have a check for the datatype because that is not readily available in cnpy
    // we apologize for the inconvenience
    int p, q, r;
    auto a = cnpy::npy_load(aFile.getValue());
    if (a.shape.size() != 2)
      throw std::runtime_error("Dimension mismatch: A has " +
			       std::to_string(a.shape.size()) +
			       " dims, but 2 expected!");
    if (a.word_size != sizeof(double))
      throw std::runtime_error("Word size mismatch: A is expected to have word "
			       "size of " + std::to_string(sizeof(double)) +
			       ", but got " + std::to_string(a.word_size));

    q = a.shape[0];
    p = a.shape[1];

    if (a.fortran_order) {
      if (verbose > 0)
	std::cerr << "A was given in Fortran order, converting to C order" << std::endl;
      compmatmul::array::fortran_to_c(q,p,a.data<double>());
    }

    auto b = cnpy::npy_load(bFile.getValue());
    if (b.shape.size() != 2)
      throw std::runtime_error("Dimension mismatch: B has " +
			       std::to_string(a.shape.size()) +
			       " dims, but 2 expected!");
    if (b.word_size != sizeof(double))
      throw std::runtime_error("Word size mismatch: B is expected to have word "
			       "size of " + std::to_string(sizeof(double)) +
			       ", but got " + std::to_string(b.word_size));

    if (static_cast<int>(b.shape[0]) != q)
      throw std::runtime_error("Dimension mismatch: B is expected to have " +
			       std::to_string(p) + " rows, but got " +
			       std::to_string(b.shape[0]) + " instead");
    r = b.shape[1];
    if (b.fortran_order) {
      if (verbose > 0)
	std::cerr << "B was given in Fortran order, converting to C order" << std::endl;
      compmatmul::array::fortran_to_c(q,r,b.data<double>());
    }


    if (verbose > 0) {
      std::cerr << "A = " << aFile.getValue() << std::endl
		<< "B = " << bFile.getValue() << std::endl
		<< "C = " << cFile.getValue() << std::endl
		<< "Force overwrite? " << (forceSwitch.getValue() ? "Yes" : "No") << std::endl
		<< "Algorithm: " << algoArg.getValue() << std::endl
		<< "b = " << bParam << std::endl
		<< "d = " << dParam << std::endl;

      std::cerr << aFile.getValue() << ": " << p << "x" << q << " ("
		<< (a.fortran_order ? "Fortran" : "C") << " order, word size = "
		<< a.word_size << ")" << std::endl;
      if (verbose > 1) {
	      std::cerr << "A:" << std::endl;
	      printMatrixT(stderr, a.data<double>(), q, p);
      }
    }
    
    if (verbose > 0) {
      std::cerr << bFile.getValue() << ": " << q << "x" << r << " ("
        << (b.fortran_order ? "Fortran" : "C") << " order, word size = "
        << b.word_size << ")" << std::endl;
      if (verbose > 1) {
	      std::cerr << "B:" << std::endl;
	      printMatrix(stderr, b.data<double>(), q, r);
      }
    }

    double* c = compmatmul::array::allocate<double>(p*r);

    if (algoArg.getValue() == "gemm") {
      if (verbose > 0)
	std::cerr << "Using dgemm multiplication" << std::endl;
      compmatmul::array::matmul_gemm(p, q, r, c, a.data<double>(), b.data<double>());
    }
#ifdef COMPMATMUL_USE_FFTW
    else if (algoArg.getValue() == "fft") {
      if (verbose > 0)
      	std::cerr << "Using compressed FFT multiplication" << std::endl;
      compmatmul::matmul_fft<compmatmul::multiply_add_shift_hash>(p, q, r, c, 
        a.data<double>(), b.data<double>(), dParam, bParam);
    }
#endif // COMPMATMUL_USE_FFTW
#ifdef COMPMATMUL_USE_FXT
    else if (algoArg.getValue() == "fwht") {
      if (verbose > 0)
        std::cerr << "Using compressed FWHT multiplication" << std::endl;
      compmatmul::matmul_fwht<compmatmul::multiply_add_shift_hash>(p, q, r, c, 
        a.data<double>(), b.data<double>(), dParam, bParam);
    }
#endif // COMPMATMUL_USE_FXT
    else {
      assert(false);
    }

    if (verbose > 1) {
      std::cerr << "C:" << std::endl;
      printMatrix(stderr, c, p, r);
    }

    if (verbose > 0) {
      std::cerr << "Saving result to " << cFile.getValue() << std::endl;
    }
    cnpy::npy_save(cFile.getValue(), c, std::vector<size_t> {
	static_cast<size_t>(p), static_cast<size_t>(r) });
    
    compmatmul::array::deallocate(c);
  }
  catch (TCLAP::ArgException& e) {
    std::cerr << argv[0] << ": error: " << e.error() << " for arg "
	      << e.argId() << std::endl;
    return EXIT_FAILURE;
  }
  catch (std::runtime_error& e) {
    std::cerr << argv[0] << ": error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
