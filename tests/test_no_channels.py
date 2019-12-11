import numpy as np

def test_A():
    H = 48
    W = 64
    K = 9

    assert K<=H and K<=W

    input = np.random.randint( 255, size=(H,W), dtype=np.int32) - 128
    print( "input.shape", input.shape)
    kernels = np.random.randint( 255, size=(K,K), dtype=np.int32) - 128
    print( "kernels.shape", kernels.shape)

    output = np.zeros( (H-K+1,W-K+1), dtype=np.int32)
    print( "output.shape", output.shape)

    for h in range(H-K+1):
        for w in range(W-K+1):
            sum = 0
            for i in range(K):
                for j in range(K):
                    sum += input[h+i][w+j] * kernels[i][j]
            output[h][w] = sum

    unrolled_output = np.ndarray.view(output).reshape( ((H-K+1)*(W-K+1),))
    print( "unrolled_output", unrolled_output.shape)

    patches = np.zeros( ((H-K+1),(W-K+1),K,K), dtype=np.int32)
    for h in range(H-K+1):
        for w in range(W-K+1):
            for i in range(K):
                for j in range(K):
                    patches[h][w][i][j] = input[h+i][w+j]

    unrolled_patches = np.ndarray.view( patches).reshape( ((H-K+1)*(W-K+1),K*K))

    unrolled_kernels = np.ndarray.view( kernels).reshape( (K*K,))

    result = np.dot(unrolled_patches,unrolled_kernels)
    print( "result.shape", result.shape)

    assert np.all( unrolled_output==result)

    rerolled_result = np.ndarray.view( result).reshape( (H-K+1,W-K+1))
    print( "rerolled_result.shape", rerolled_result.shape)

    assert np.all( output==rerolled_result)

