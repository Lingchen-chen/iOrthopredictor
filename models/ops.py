# the code below is adapted from https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py

import numpy as np
import tensorflow as tf
from extern.dnnlib.upfirdn_2d_n import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from extern.dnnlib.fused_bias_act_n import fused_bias_act

# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolution or fully-connected layer.

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable(weight_var, shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense_layer(x, fmaps, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])]) #flatten
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Convolution layer with optional upsampling or downsampling.

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    if up:
        x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    elif down:
        x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:
        x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    return x

#----------------------------------------------------------------------------
# Apply bias and activation func.

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain) # linear means no activation fn

#----------------------------------------------------------------------------
# Modulated convolution layer.

def modulated_conv2d_layer(x, y, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight', mod_bias_var='mod_bias'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1
    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)
    ww = w[np.newaxis] # [BkkIO] Introduce minibatch dimension.

    with tf.name_scope("Modulate"):
        s = dense_layer(y, fmaps=x.shape[1].value, weight_var=mod_weight_var) # [BI] Transform incoming W to style.
        s = apply_bias_act(s, bias_var=mod_bias_var) + 1 # [BI] Add bias (initially 1)., no activation fn

        ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

    with tf.name_scope("Demodulate"):
        if demodulate:
            d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor., only the H, W, C_in
            ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

    with tf.name_scope("Convolution"):
        # Reshape/scale input.
        if fused_modconv:
            w = ww
        else:
            x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

        # Convolution with optional up/downsampling.
        # group conv
        B = x.get_shape()[0].value
        O = []
        for i in range(B):
            b = x[i:i+1, :, :, :]
            if up:
                b = upsample_conv_2d(b, tf.cast(w[i], b.dtype), data_format='NCHW', k=resample_kernel)
            elif down:
                b = conv_downsample_2d(b, tf.cast(w[i], b.dtype), data_format='NCHW', k=resample_kernel)
            else:
                b = tf.nn.conv2d(b, tf.cast(w[i], b.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
            O.append(b)

        # Reshape/scale output.
        if fused_modconv:
            x = tf.concat(O, axis=0) # Fused => reshape convolution groups back to minibatch.
        elif demodulate:
            x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
        return x


def modulated_conv2d_layer_n(x, fmaps, kernel, up=False, down=False, resample_kernel=None, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    assert not (up and down)
    assert kernel >= 1 and kernel % 2 == 1

    # Get weight.
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, lrmul=lrmul, weight_var=weight_var)

    with tf.name_scope("Weight_Normalization"):
        s = tf.get_variable("scale", shape=[fmaps], initializer=tf.initializers.ones()) * lrmul # (initially 1).
        w = tf.nn.l2_normalize(w, axis=[0, 1, 2], epsilon=1e-8)
        w *= tf.cast(s[np.newaxis, np.newaxis, np.newaxis, :], w.dtype)

    with tf.name_scope("Convolution"):
        # Convolution with optional up/downsampling.
        if up:
            x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
        elif down:
            x = conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
        else:
            x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
        return x

#----------------------------------------------------------------------------
# Minibatch standard deviation layer.

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    group_size = tf.minimum(group_size, x.get_shape()[0].value)     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y = tf.cast(y, tf.float32)                              # [GMncHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.cast(y, x.dtype)                                 # [Mn11]  Cast back to original data type.
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=1)                        # [NCHW]  Append as new fmap.


def latent_sample(p,
                  latent_size=128,
                  act="lrelu"):

    with tf.variable_scope("LatentSample"):
        p_pos = apply_bias_act(modulated_conv2d_layer_n(p, latent_size, 3), act=act)
        eps = tf.truncated_normal(p_pos.shape, mean=0.0, stddev=1.0)
        z_pos = p_pos + 1.0 * eps
        return p_pos, z_pos


def mapping(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    num_channels        = 3,            # Number of input color channels.
    resolution          = 256,          # Input resolution.
    fmap_min            = 32,           # Minimum number of feature maps in any layer.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    last_layer          = 2,
    architecture        = 'orig',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2 ** resolution_log2

    fmap_base = 2 ** resolution_log2 * fmap_min  # Overall multiplier for the number of feature maps.
    def nf(stage):
        return np.clip(int(fmap_base / (2.0 ** stage)), fmap_min, fmap_max)

    assert architecture in ['orig', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks for main layers.
    def layer(x, fmaps, kernel, down=False, up=False):
        x = modulated_conv2d_layer_n(x, fmaps=fmaps, kernel=kernel, down=down, up=up, resample_kernel=resample_kernel)
        return apply_bias_act(x, act=act)

    def block1(x, res):  # res = resolution_log2 --> last_layer+1
        t = x
        with tf.variable_scope('Conv0_down'):
            x = layer(x, fmaps=nf(res - 1), kernel=3, down=True)
        if architecture == 'resnet':
            with tf.variable_scope('Skip_down'):
                t = conv2d_layer(t, fmaps=nf(res - 1), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def fromrgb(x, res): # res = resolution_log2
        with tf.variable_scope('FromRGB'):
            x = layer(x, fmaps=nf(res), kernel=3)
            return x

    # Main layers.
    x = images_in
    y = []
    for res in range(resolution_log2, last_layer, -1):
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            if res == resolution_log2:
                x = fromrgb(x, res)
                y.append(x)
            x = block1(x, res)
            y.append(x)

    return y


def G_mapping(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    latent_size             = 128,          # Latent vector (Z) dimensionality.
    dlatent_size            = 128,          # latent (W) dimensionality for style modulation.
    mapping_layers          = 2,            # Number of mapping layers.
    mapping_fmaps           = 128,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    act = mapping_nonlinearity

    # Inputs.
    latents_in = tf.reduce_mean(latents_in, axis=[2, 3])
    latents_in.set_shape([None, latent_size])
    latents_in = tf.cast(latents_in, dtype)
    x = latents_in

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = apply_bias_act(dense_layer(x, fmaps=fmaps, lrmul=mapping_lrmul), act=act, lrmul=mapping_lrmul)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


def G_rendering(
    dlatents,                           # Input: extracted appearance feature map for bottleneck concatenation.
    latents_in,                         # Input: extracted appearance feature vector for style modulation.
    geometry_in,                        # Input: extracted hierarchical geometry feature maps list

    num_channels        = 3,            # Number of output color channels.
    resolution          = 256,          # Output resolution.
    fmap_min            = 32,           # Minimum number of feature maps in any layer.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.
    randomize_noise     = True,
    architecture        = 'orig',       # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    dlatent_size = int(np.log2(dlatents.get_shape()[-1].value))
    geo_size = int(np.log2(geometry_in[-1].shape[-1].value))
    fmap_base = 2 ** resolution_log2 * fmap_min  # Overall multiplier for the number of feature maps.

    assert geo_size == dlatent_size
    assert resolution == 2**resolution_log2

    def nf(stage):
        return np.clip(int(fmap_base / (2.0 ** stage)), fmap_min, fmap_max)

    assert architecture in ['orig', 'skip']
    act = nonlinearity

    # Style convolution layer: modconv + bias + act
    def layer_1(x, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, latents_in, fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel)
        if randomize_noise:
            noise = tf.random_normal([x.get_shape()[0].value, 1, x.shape[2].value, x.shape[3].value], dtype=x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Normal convolution layer with weight normalization
    def layer_2(x, fmaps, kernel, up=False):
        x = modulated_conv2d_layer_n(x, fmaps=fmaps, kernel=kernel, up=up, resample_kernel=resample_kernel)
        if randomize_noise:
            noise = tf.random_normal([x.get_shape()[0].value, 1, x.shape[2].value, x.shape[3].value], dtype=x.dtype)
            noise_strength = tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Building blocks for main layers.
    def block(x, res, geo, up=True): # res = dlatent_size..resolution_log2
        m_layer = layer_1 if latents_in is not None else layer_2
        if up:
            with tf.variable_scope('Conv0_up'):
                x = m_layer(x, fmaps=nf(res), kernel=3, up=True)

        with tf.variable_scope('Conv1'):
            x = tf.concat([x, geo], axis=1) # concatenate through feature channel
            x = m_layer(x, fmaps=nf(res), kernel=3)

        return x

    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel) #bilnear sampling

    # to do: fix the bug
    def torgb(x, y): # res = 2..resolution_log2
        with tf.variable_scope('ToRGB'):
            # if latents_in is not None:
            #     t = apply_bias_act(modulated_conv2d_layer(x, latents_in, fmaps=num_channels, kernel=1, demodulate=False))
            t = apply_bias_act(modulated_conv2d_layer_n(x, fmaps=num_channels, kernel=1))
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('{}x{}'.format(2 ** dlatent_size, 2 ** dlatent_size)):
        with tf.variable_scope('Conv'):
            x = block(dlatents, dlatent_size, geometry_in.pop(), up=False)
        if architecture == 'skip':
            y = torgb(x, y)

    for res in range(dlatent_size + 1, resolution_log2 + 1):
        with tf.variable_scope('%dx%d' % (2 ** res, 2 ** res)):
            x = block(x, res, geometry_in.pop(), up=True)
            if architecture == 'skip':
                y = upsample(y)
            if architecture == 'skip' or res == resolution_log2:
                y = torgb(x, y)

    images_out = y

    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


def D(
    images_in,                          # First input: Images [minibatch, channel, height, width].

    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 256,          # Input resolution. Overridden based on dataset.
    fmap_min            = 32,           # Minimum number of feature maps in any layer.
    fmap_max            = 128,          # Maximum number of feature maps in any layer.

    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    fmap_base = 2 ** resolution_log2 * fmap_min  # Overall multiplier for the number of feature maps.
    assert resolution == 2 ** resolution_log2

    def nf(stage):
        return np.clip(int(fmap_base / (2.0 ** stage)), fmap_min, fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    images_in = tf.cast(images_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res), kernel=1), act=act)
            return t if x is None else x + t

    def block(x, res): # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-1), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=1))
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out
