import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        accum_tile = nl.zeros(
            shape=(out_channels, out_pool_height, out_pool_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        for i in nl.sequential_range(filter_height):
            for j in nl.sequential_range(filter_width):

                X_tile = nl.ndarray((nl.par_dim(in_channels), out_height * out_width), dtype=X.dtype, buffer=nl.sbuf)
                # See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#nki-programming-model for optimized indexing
                for c_in in nl.sequential_range(in_channels):
                    for h in nl.sequential_range(out_height):
                        for w in nl.sequential_range(out_width):
                            flat_idx = h * out_width + w
                            X_tile[c_in, flat_idx] = nl.load(X[b, c_in, i + h, j + w])

                W_tile = nl.ndarray((out_channels, in_channels), dtype=W.dtype, buffer=nl.sbuf)
                for c_out in nl.sequential_range(out_channels):
                    for c_in in nl.affine_range(in_channels):
                        W_tile[c_out, c_in] = nl.load(W[c_out, c_in, i, j])
                
                result = nl.matmul(W_tile, X_tile, transpose_x=True)

                for c_out in nl.sequential_range(out_channels):
                    for h in nl.sequential_range(out_height):
                        for w in nl.sequential_range(out_width):
                            flat_idx = h * out_width + w
                            accum_tile[c_out, h, w] = nl.add(accum_tile[c_out, h, w], result[c_out, flat_idx])
    
        nl.store(X_out[b], value=accum_tile)
    return X_out

