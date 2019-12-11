import numpy as np

import time

def test_A():
    H = 48
    W = 64
    K = 9
    IC = 3
    OC = 4

    assert K<=H and K<=W

    input = np.random.randint( 255, size=(H,W,IC), dtype=np.int32) - 128
    print( "input.shape", input.shape)
    kernels = np.random.randint( 255, size=(K,K,IC,OC), dtype=np.int32) - 128
    print( "kernels.shape", kernels.shape)

    before = time.time()
    output = np.zeros( (H-K+1,W-K+1,OC), dtype=np.int32)
    print( "output.shape", output.shape)

    for h in range(H-K+1):
        for w in range(W-K+1):
            for oc in range(OC):
                sum = 0
                for i in range(K):
                    for j in range(K):
                        for ic in range(IC):
                            sum += input[h+i][w+j][ic] * kernels[i][j][ic][oc]
                output[h][w][oc] = sum
    print( "Naive compute time:", time.time()-before)

    unrolled_output = np.ndarray.view(output).reshape( ((H-K+1)*(W-K+1),OC))
    print( "unrolled_output", unrolled_output.shape)

    before = time.time()
    patches = np.zeros( ((H-K+1),(W-K+1),K,K,IC), dtype=np.int32)
    for h in range(H-K+1):
        for w in range(W-K+1):
            patches[h][w] = input[h:h+K,w:w+K]

#            for i in range(K):
#                for j in range(K):
#                     patches[h][w][i][j] = input[h+i][w+j]

#                    for ic in range(IC):
#                        patches[h][w][i][j][ic] = input[h+i][w+j][ic]
    print( "Patch generation time:", time.time()-before)

    unrolled_patches = np.ndarray.view( patches).reshape( ((H-K+1)*(W-K+1),K*K*IC))

    unrolled_kernels = np.ndarray.view( kernels).reshape( (K*K*IC,OC))

    before = time.time()
    result = np.dot(unrolled_patches,unrolled_kernels)
    print( "GEMM time:", time.time()-before)
    print( "result.shape", result.shape)

    assert np.all( unrolled_output==result)

    rerolled_result = np.ndarray.view( result).reshape( (H-K+1,W-K+1,OC))
    print( "rerolled_result.shape", rerolled_result.shape)

    assert np.all( output==rerolled_result)

