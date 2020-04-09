#
#
# We want to take an input stream in: (row,col,chan) order (last field indices move fastest).
#
# We can read a block from memory aligned to the block size.
#
# We want to produce

import numpy as np
from itertools import product

class Cache:
    def __init__( self, capacity, *, linesize=8):
        self.tbl = {} # mem index -> age
        self.access_number = 0
        self.capacity = capacity
        self.linesize = linesize
        self.hits = 0
        self.misses = 0

    def cache_address( self, mem_index):
        return mem_index - (mem_index % self.linesize)

    def access( self, mem_index):
        ca = self.cache_address( mem_index)

        result = 'MISS'
        if ca in self.tbl:
            if self.access_number - self.tbl[ca] <= self.capacity:
                result = 'HIT'
        self.tbl[ca] = self.access_number
        self.access_number += 1
        if result == 'HIT':
            self.hits += 1
        else:
            self.misses += 1
        return result

def A( tag, params):

    H, W, IC, K, S = params

# H,W,IC
#    memory = list( product( range(H), range(W), range(IC)))
# IC,H,W
    memory = list( product( range(IC), range(H), range(W)))

    def a( ic,h,w):
# H,W,IC
#        return (h*W + w)*IC + ic
# IC,H,W
        return (ic*H + h)*W + w

    reshaped = []
    for h in range(H-K+1):
        if h % S != 0: continue
        for w in range(W-K+1):
            if w % S != 0: continue
            reshaped.append( [])
            for ic in range(IC):
                for i in range(K):
                    for j in range(K):
                        reshaped[-1].append( memory[a( ic, h+i, w+j)])
                        

    print(len(reshaped), len(reshaped[0]))

    miss_tally = {}

    for cs in [8,16,32,64,128,10000000]:
        ch = Cache( cs, linesize=8)

        b = 64
        m = len(reshaped)
        k = len(reshaped[0])
        for blk0 in range( (m+b-1)//b):
            for x in range( k):
#                print( 'cs', cs, 'Block', blk0, x)            
                for blk1 in range( b):
                    y = blk0*b+blk1
                    if y < m:
                        tup = reshaped[y][x]
                        mem_index = a( *tup)
                        status = ch.access( mem_index)
#                        print( tup, mem_index, status)
#                print()
        
        f = ch.misses/(ch.hits+ch.misses)
        perfect = params[3]*params[3]/(params[4]*params[4])
        miss_tally[cs] = f'Cache size: {cs} Hits: {ch.hits} Misses: {ch.misses} Fraction of misses: {f} {f*ch.linesize} {1/(f*ch.linesize)} {perfect}'
#        print( miss_tally[cs])

    for (k,v) in miss_tally.items():
        print( params, tag, v)

def test_A():
    params = [ 
        (224, 224,   3, 7, 2),
        (112, 112,  64, 3, 2),
        ( 56,  56,  64, 3, 1),
        ( 56,  56,  64, 1, 1),
        ( 56,  56, 128, 3, 2),
        ( 28,  28, 128, 3, 1),
        ( 28,  28, 128, 1, 1),
        ( 28,  28, 256, 3, 2),
        ( 14,  14, 256, 3, 1),
        ( 14,  14, 256, 1, 1),
        ( 14,  14, 512, 3, 2),
        ( 14,   7, 512, 3, 1),
        ( 14,   7, 512, 1, 1)
    ]
    for idx,param in enumerate( params):
        A( idx, param) 
