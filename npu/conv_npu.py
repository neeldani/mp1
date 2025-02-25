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
    c_in_pmax = 128
    c_out_pmax = 128

    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax 

    W_tiled = nl.ndarray(
        shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    for c_out in nl.affine_range(n_tiles_c_out):         
        for c_in in nl.affine_range(n_tiles_c_in):
            W_tiled[c_out, :, c_in, :, :, :] = nl.load(W[nl.ds(c_out * 128, 128), nl.ds(c_in * 128, 128), :, :])

    W_permute = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    W_transposed = nl.ndarray(
        shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
        dtype=W.dtype,
        buffer=nl.sbuf
    )

    # nl.device_print(W_tiled.shape)
    # nl.device_print(W_permute.shape)
    # nl.device_print(W_transposed.shape)

    for fh in nl.affine_range(filter_height):
        for fw in nl.affine_range(filter_width):
            for c_out in nl.affine_range(n_tiles_c_out):
                for c_in in nl.affine_range(n_tiles_c_in):
                    W_permute[fh, fw, c_out, c_in] = nl.copy(W_tiled[c_out, :, c_in, :, fh, fw])
                    W_transposed[fh, fw, c_out, c_in] = nisa.nc_transpose(W_permute[fh, fw, c_out, c_in])

    # nl.device_print(W_permute[0])
    
    output_tile_height = 2
    input_tile_height = output_tile_height + filter_height - 1

    n_tiles_h = out_height // output_tile_height

    for b in nl.affine_range(batch_size):
        for tile_h in nl.affine_range(n_tiles_h):
            X_tile = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_tile_height, input_width), 
                dtype=X[b].dtype, 
                buffer=nl.sbuf)

            for c_in in nl.affine_range(n_tiles_c_in):
                # X_tile = nl.load(X[b, c_in * 128: c_in * 128 + 128, tile_h * output_tile_height: tile_h * output_tile_height + input_tile_height, :])
                X_tile[c_in, :, :, :] = nl.load(X[b, nl.ds(c_in * 128, 128), nl.ds(tile_h * output_tile_height, input_tile_height), :])
            
            for c_out in nl.affine_range(n_tiles_c_out):
                
                output_tile = nl.ndarray(shape=(nl.par_dim(c_out_pmax), output_tile_height, out_width), 
                    dtype=X_out[b].dtype, 
                    buffer=nl.sbuf)
                
                for out_row in nl.affine_range(output_tile_height):
                    acc_tile = nl.zeros((nl.par_dim(c_out_pmax), out_width), nl.float32, buffer=nl.psum)
                
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            for c_in in nl.affine_range(n_tiles_c_in):
                                x_slice = X_tile[c_in, :, out_row + i, j : j + out_width]
                                w_slice = W_transposed[i, j, c_out, c_in, :, :]
                                acc_tile += nl.matmul(w_slice, x_slice, transpose_x=True)
                    
                    output_tile[:, out_row, :] = acc_tile

                bias_tile = nl.load(bias[nl.ds(c_out * 128, 128)]).reshape((128, 1))
                output_tile = nisa.tensor_scalar(output_tile, np.add, bias_tile)
                
                h_start = tile_h * output_tile_height
                h_end = h_start + output_tile_height
                nl.store(
                    X_out[b, nl.ds(c_out * 128, 128), h_start : h_end, :],
                    value=output_tile,
                )
    return X_out
