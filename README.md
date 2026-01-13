# Compressed Matrix Multiplication

This is a header-only library that implements Pagh's
compressed matrix multiplication [1], optionally with Fast Walsh Hadamard transform.

## Requirements

The library currently specifically targets the Intel MKL as the backend, although it is possible to also hack it to use OpenBLAS instead. For a successful build, you need
- Apptainer
- GCC (included in the Apptainer image)
- FFTW (downloaded automatically by the build process)
- FXT (downloaded automatically by the build process)
- TCLAP (included in the sources)
- pybind11 (included in the Apptainer image)

For using the `multiply` test program, you need also the cnpy library (included in the sources) and zlib (included in the Apptainer image). In general, the code should be very self-contained, assuming you use the provided Apptainer definition file.

## Building

Simply run

`$ apptainer build compmatmul.sif Apptainer.def`

You can then run a Python instance that imports the `compmatmul` module.

`$ apptainer run compmatmul.sif python3 -ic 'import compmatmul'`

Please see `test.py` on how to use the module.

## License

The code is licensed under the MIT license.

## Citing

TBD.

## References
[1] Rasmus Pagh. 2013. Compressed matrix multiplication. ACM Trans. Comput. Theory 5, 3, Article 9 (August 2013), 17 pages. https://doi.org/10.1145/2493252.2493254