import numpy as np
import compmatmul
from itertools import product

def is_aligned(arr, alignment= 64):
    return arr.ctypes.data % alignment == 0

def aligned_alloc(shape, dtype = np.float64, alignment = 64):
    elem_size = np.dtype(dtype).itemsize
    byte_size = np.prod(shape) * elem_size
    A = np.empty(byte_size + alignment, np.uint8)
       
    offset = (alignment - A.ctypes.data % alignment) % alignment
    A = A[offset:offset+byte_size].view(dtype).reshape(shape)
    return A

def as_aligned_array(arr, alignment = 64):
    if not is_aligned(arr,alignment):
        A = aligned_alloc(arr.shape, arr.dtype, alignment)
        A[:] = arr[:]
        return A
    else:
        return arr
    
def aligned_zeros(shape, dtype = np.float64, alignment = 64):
    A = aligned_alloc(shape,dtype,alignment)
    A.fill(0)
    return A

AT_4x4 = \
    as_aligned_array(np.ascontiguousarray(np.arange(1,17,dtype=np.float64) \
                                          .reshape(4,4).T))
B_4x4 = as_aligned_array(np.arange(17,33,dtype=np.float64).reshape(4,4))
C_4x4x4 = as_aligned_array(AT_4x4.T @ B_4x4)

AT_2x4 = \
    as_aligned_array(np.ascontiguousarray(np.arange(1,9,dtype=np.float64) \
                                          .reshape(4,2).T))
B_2x4 = as_aligned_array(np.arange(9,17,dtype=np.float64).reshape(2,4))
C_4x2x4 = as_aligned_array(AT_2x4.T @ B_2x4)

AT_4x2 = as_aligned_array(np.ascontiguousarray(np.arange(1,9,dtype=np.float64) \
                                               .reshape(2,4).T))
B_4x2 = as_aligned_array(np.arange(9,17,dtype=np.float64).reshape(4,2))
C_2x4x2 = as_aligned_array(AT_4x2.T @ B_4x2)

B_4x8 = as_aligned_array(np.arange(9,41,dtype=np.float64).reshape(4,8))
C_2x4x8 = as_aligned_array(AT_4x2.T @ B_4x8)


def test_multiply():
    seed = 13794
    params = {
        ('fft',None) : (1,256),
        ('fft','multiply shift') : (1,128),
        ('fft','multiply add shift') : (1,256),
        ('fft','tabulation8') : (1,128),
        ('fft','tabulation16') : (1,64),
        ('fwht',None) : (1,32),
        ('fwht','multiply shift') : (1,64),
        ('fwht','multiply add shift') : (1,32),
        ('fwht','tabulation8') : (1,64),
        ('fwht','tabulation16') : (1,64),
    }
    for (AT,B,C_correct) in \
        [(AT_4x4, B_4x4, C_4x4x4), (AT_4x2,B_4x2,C_2x4x2), 
         (AT_2x4,B_2x4,C_4x2x4), (AT_4x2,B_4x8,C_2x4x8)]:
        for (algo,hash_fun) in \
            product(['gemm','fft','fwht'],
                    [None,'multiply shift', 'multiply add shift','tabulation8',
                        'tabulation16']):
            if algo == 'gemm' and hash_fun is not None:
                continue
            C = aligned_alloc(C_correct.shape)
            if algo == 'gemm':
                assert hash_fun is None
                compmatmul.gemm(C, AT, B)
                assert np.array_equal(C,C_correct)
            elif algo == 'fft':
                d, b = params[(algo,hash_fun)]
                compmatmul.matmul_fft(C,AT,B,d,b,seed,hash_fun)
                assert np.allclose(C,C_correct)
            elif algo == 'fwht':
                d, b = params[(algo,hash_fun)]
                compmatmul.matmul_fwht(C,AT,B,d,b,seed,hash_fun)
                assert np.array_equal(C,C_correct)
            else:
                assert np.array_equal(C,C_correct)

    seed1 = 27548
    seed2 = 23340
    C1 = aligned_alloc(C_2x4x8.shape)
    C2 = aligned_alloc(C_2x4x8.shape)
    C3 = aligned_alloc(C_2x4x8.shape)
    d = 3
    b = 8
    compmatmul.matmul_fft(C1,AT_4x2,B_4x8,d,b,seed1)
    compmatmul.matmul_fft(C2,AT_4x2,B_4x8,d,b,seed2)
    compmatmul.matmul_fft(C3,AT_4x2,B_4x8,d,b,seed1)
    assert np.allclose(C1,C3)    
    assert not np.allclose(C1,C2)
    compmatmul.matmul_fft(C2,AT_4x2,B_4x8,d,b,seed1)
    assert np.allclose(C1,C2)
    compmatmul.matmul_fft(C3,AT_4x2,B_4x8,d,b,seed1,'multiply shift')
    compmatmul.matmul_fft(C2,AT_4x2,B_4x8,d,b,seed1,'multiply add shift')
    assert not np.allclose(C1,C3)
    assert np.allclose(C1,C2)

    compmatmul.matmul_fwht(C1,AT_4x2,B_4x8,d,b,seed1)
    compmatmul.matmul_fwht(C2,AT_4x2,B_4x8,d,b,seed2)
    compmatmul.matmul_fwht(C3,AT_4x2,B_4x8,d,b,seed1)
    assert np.array_equal(C1,C3)    
    assert not np.allclose(C1,C2)
    compmatmul.matmul_fwht(C2,AT_4x2,B_4x8,d,b,seed1)
    assert np.array_equal(C1,C2)
    compmatmul.matmul_fwht(C3,AT_4x2,B_4x8,d,b,seed1,'multiply shift')
    compmatmul.matmul_fwht(C2,AT_4x2,B_4x8,d,b,seed1,'multiply add shift')
    assert not np.allclose(C1,C3)
    assert np.array_equal(C1,C2)


    params = {
        ('fft',None) : (7,1024),
        ('fft','multiply shift') : (5,1024),
        ('fft','multiply add shift') : (7,1024),
        ('fft','tabulation8') : (9,2048),
        ('fft','tabulation16') : (9,1024),
        ('fwht',None) : (11,1024),
        ('fwht','multiply shift') : (9,1024),
        ('fwht','multiply add shift') : (11,1024),
        ('fwht','tabulation8') : (9,1024),
        ('fwht','tabulation16') : (11,2048),
    }

    rng = np.random.default_rng(27840)
    A = as_aligned_array(rng.integers(-10,11,(16,16)).astype(np.float64))
    B = as_aligned_array(rng.integers(-10,11,(16,16)).astype(np.float64))
    C_correct = A @ B
    AT = as_aligned_array(np.ascontiguousarray(A.T))

    for (algo,hash_fun) in \
        product(['gemm','fft','fwht'],
                [None,'multiply shift', 'multiply add shift','tabulation8',
                    'tabulation16']):
        if algo == 'gemm' and hash_fun is not None:
            continue
        C = aligned_alloc(C_correct.shape)
        if algo == 'gemm':
            assert hash_fun is None
            compmatmul.gemm(C, AT, B)
            assert np.array_equal(C,C_correct)
        elif algo == 'fft':
            d, b = params[(algo,hash_fun)]
            compmatmul.matmul_fft(C,AT,B,d,b,seed,hash_fun)
            assert np.allclose(C,C_correct)
        elif algo == 'fwht':
            d, b = params[(algo,hash_fun)]
            compmatmul.matmul_fwht(C,AT,B,d,b,seed,hash_fun)
            assert np.array_equal(C,C_correct)
        else:
            assert np.array_equal(C,C_correct)


if __name__ == '__main__':
    test_multiply()
