import numpy as np
import time
import pytest

@pytest.mark.parametrize("params", [(48,64,9),(768,1024,1)])
def test_A(params):
    print()
    print("params", params)
    H,W,K = params

    assert K<=H and K<=W

    input = np.random.randint( 255, size=(H,W), dtype=np.int32) - 128
    print( "input.shape", input.shape)
    kernels = np.random.randint( 255, size=(K,K), dtype=np.int32) - 128
    print( "kernels.shape", kernels.shape)

    before = time.time()
    output = np.zeros( (H-K+1,W-K+1), dtype=np.int32)


    for h in range(H-K+1):
        for w in range(W-K+1):
            sum = 0
            for i in range(K):
                for j in range(K):
                    sum += input[h+i][w+j] * kernels[i][j]
            output[h][w] = sum
    print( "Naive compute time:", time.time()-before)


    print( "output.shape", output.shape)
    unrolled_output = np.ndarray.view(output).reshape( ((H-K+1)*(W-K+1),))
    print( "unrolled_output", unrolled_output.shape)

    before = time.time()
    if K == 1:
        patches = input
    else:
        patches = np.zeros( ((H-K+1),(W-K+1),K,K), dtype=np.int32)
        for h in range(H-K+1):
            for w in range(W-K+1):
                patches[h][w] = input[h:h+K,w:w+K]
#                for i in range(K):
#                    for j in range(K):
#                        patches[h][w][i][j] = input[h+i][w+j]
    print( "Patch generation time:", time.time()-before)

    unrolled_patches = np.ndarray.view( patches).reshape( ((H-K+1)*(W-K+1),K*K))

    unrolled_kernels = np.ndarray.view( kernels).reshape( (K*K,))

    before = time.time()
    result = np.dot(unrolled_patches,unrolled_kernels)
    print( "GEMM time:", time.time()-before)
    print( "result.shape", result.shape)

    assert np.all( unrolled_output==result)

    rerolled_result = np.ndarray.view( result).reshape( (H-K+1,W-K+1))
    print( "rerolled_result.shape", rerolled_result.shape)

    assert np.all( output==rerolled_result)

