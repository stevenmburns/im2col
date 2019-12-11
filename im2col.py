import numpy as np

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

unroll_output = np.ndarray.view(output).reshape( ((H-K+1)*(W-K+1),))
print( "unroll_output", unroll_output.shape)

patches = np.zeros( ((H-K+1)*(W-K+1),K*K), dtype=np.int32)
print( "patches.shape", patches.shape)

for h in range(H-K+1):
    for w in range(W-K+1):
        for i in range(K):
            for j in range(K):
                patches[h*(W-K+1)+w][i*K+j] = input[h+i][w+j]

patches2 = np.zeros( ((H-K+1),(W-K+1),K,K), dtype=np.int32)
for h in range(H-K+1):
    for w in range(W-K+1):
        for i in range(K):
            for j in range(K):
                patches2[h][w][i][j] = input[h+i][w+j]

patches3 = np.ndarray.view( patches2).reshape( ((H-K+1)*(W-K+1),K*K))
assert np.all( patches3 == patches)

unroll_kernels = np.ndarray.view( kernels).reshape( (K*K,))

result = np.dot(patches,unroll_kernels)
print( "result.shape", result.shape)

assert np.all( unroll_output==result)

rerolled_result = np.ndarray.view( result).reshape( (H-K+1,W-K+1))
print( "rerolled_result.shape", rerolled_result.shape)

assert np.all( output==rerolled_result)

