#efficient net standard scaling and block configs
#https://github.com/FluxML/Metalhead.jl/blob/master/src/convnets/efficientnet.jl
# n: # of block repetitions
# k: kernel size k x k
# s: stride
# e: expansion ratio
# i: block input channels
# o: block output channels
# this has been modified from k to (k,1)
const efficientnet_block_configs = [
#   (n,     k, s, e,   i,   o)   k
    (1, (3,1), 1, 1,  32,  16), #3
    (2, (3,1), 2, 6,  16,  24), #3
    (2, (5,1), 2, 6,  24,  40), #5
    (3, (3,1), 2, 6,  40,  80), #3
    (3, (5,1), 1, 6,  80, 112), #5
    (4, (5,1), 2, 6, 112, 192), #5
    (1, (3,1), 1, 6, 192, 320)  #3 <-- original
#    (1, (3,1), 1, 6, 192, 220)  #3 <-- hacking
]
# w: width scaling
# d: depth scaling
# r: image resolution
const efficientnet_global_configs = Dict(
#          (  r, (  w,   d))
    :b0 => (224, (1.0, 1.0)),
    :b1 => (240, (1.0, 1.1)),
    :b2 => (260, (1.1, 1.2)),
    :b3 => (300, (1.2, 1.4)),
    :b4 => (380, (1.4, 1.8)),
    :b5 => (456, (1.6, 2.2)),
    :b6 => (528, (1.8, 2.6)),
    :b7 => (600, (2.0, 3.1)),
    :b8 => (672, (2.2, 3.6))
)

