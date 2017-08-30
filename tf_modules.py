import tensorflow as tf
import numpy as np

ph = lambda shape, dtype=tf.float32, name=None: tf.placeholder(dtype, shape=shape, name=name)

def _set_static_size_(tensor, dim, size):
    shape = [ss.value for ss in tensor.get_shape()]
    shape[dim] = size
    tensor.set_shape(shape)


def _static_size(tensor, dim):
    out = tensor.get_shape()[dim].value
    assert out is not None, 'tensor %s dim %d is not static' % (tensor.name, dim)
    return out

def _rank(tensor):
    return len(tensor.get_shape())


def _he_init(shape):
    out = (np.random.randn(*shape) * np.sqrt(2. / np.prod(shape[:-1]))).astype(np.float32)
    return out


def _scope_variable(scope_name, var_name, shape, init=_he_init):
    with tf.variable_scope(scope_name) as scope:
        try:
            initializer = init(shape)
            var = tf.get_variable(name=var_name, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            var = tf.get_variable(name=var_name)
    return var


def _variable(name, shape, init=_he_init, reuse=False):
    initializer = init(shape)
    var = tf.get_variable(name=name, initializer=initializer)
    return var


def linear(x, dim_output, name='', reuse=False):
    assert _rank(x) == 2, 'x is rank %d; must be rank 2' % (_rank(x))
    dim_input = _static_size(x, 1)
    vs = 'linear'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs) as scope:
        weights = _variable('weights', [dim_input, dim_output], reuse=reuse)
        bias = _variable('bias', [dim_output], reuse=reuse)
        out = tf.add(tf.matmul(x, weights), bias)
        out.weights = weights
        out.bias = bias
    return out


def dropout(x, is_training, p_drop, name=''):
    vs = 'dropout'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs) as scope:
        def dropped():
            mask = tf.to_float(tf.random_uniform(tf.shape(x)) > p_drop)
            unscaled_dropped = x * mask
            dropped = unscaled_dropped * (1./(1.-p_drop))
            dropped.unscaled_dropped = unscaled_dropped
            dropped.mask = mask
            return dropped
    return tf.cond(is_training, lambda: dropped(), lambda: x)


def fc_layer(x, dim_output, act_fn=tf.nn.relu, p_drop=0., batch_norm=False, name='', reuse=False, is_training=None):
    if p_drop > 0. or batch_norm:
        assert is_training is not None, 'must pass is_training if using dropout or batch norm'

    vs = 'fc_layer'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs, reuse=reuse) as scope:
        out = pre_act = linear(x, dim_output, name=name, reuse=reuse)
        # TODO: fit batch_norm in
        # assert batch_norm is False, 'batch_norm not yet implemented'
        if batch_norm:
            out = batch_normed = batch_normalization(out, is_training=is_training)
        else:
            batch_normed = None
        out = act = act_fn(out, name=name)
        if p_drop > 0.:
            out = dropped = dropout(out, is_training, p_drop, name=name)
        else:
            dropped = None
        out.pre_act = pre_act
        out.batch_normed = batch_normed
        out.act = act
        out.dropped = dropped
    return out


def batch_normalization(x, is_training, loc_trainable=True, scale_trainable=True, decay=0.99, eps=1e-3, add_noise=False, reuse=False, name=''):
    """
        loc_trainable: bool, learnable means
        scale_trainable: bool, learnable vars
        decay: scalar in (0., 1.), how fast moving average updates (bigger is faster)
        eps: scalar in (0., inf), stability term for batch var
        name:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """

    def _assign_moving_average(orig_val, new_val, decay, name='_assign_moving_average'):
        with tf.name_scope(name):
            td = decay * (new_val - orig_val)
            return tf.assign_add(orig_val, td)

    vs = 'batch_normalization'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs, reuse=reuse) as scope:
        input_rank = _rank(x)
        pool_axes = np.arange(input_rank-1).tolist()
        n_out = _static_size(x, -1)
        moving_mean = tf.get_variable(name='moving_mean',
                                      shape=[n_out],
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        moving_var = tf.get_variable(name='moving_var',
                                     shape=[n_out],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
        loc = tf.get_variable(name='loc',
                              shape=[n_out],
                              initializer=tf.zeros_initializer(),
                              trainable=loc_trainable)
        scale = tf.get_variable(name='scale',
                                shape=[n_out],
                                initializer=tf.ones_initializer(),
                                trainable=scale_trainable)

        batch_mean, batch_var = tf.nn.moments(x, pool_axes, name='moments')
        def training():
            update_mm = _assign_moving_average(moving_mean, batch_mean, decay)
            update_mv = _assign_moving_average(moving_var, batch_var, decay)
            with tf.control_dependencies([update_mm, update_mv]):
                normed = tf.nn.batch_normalization(x, batch_mean, batch_var, loc, scale, eps)
                if add_noise:
                    normed += Normal(loc, scale).sample(tf.shape(normed)[0])
                return normed

        def testing():
            return tf.nn.batch_normalization(x, moving_mean, moving_var, loc, scale, eps)

        normed = tf.cond(is_training, training, testing)

        normed.batch_mean = batch_mean
        normed.batch_var = batch_var
        return normed


def dialated_conv2d(x, ch_out, khw, rate, padding, name='', reuse=None):
    vs = 'dialated_conv2d/'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs, reuse=reuse) as scope:
        kh, kw = khw

        ch_in = _static_size(x, 3)

        W_shape = [kh, kw, ch_in, ch_out]
        b_shape = [ch_out]
        W = _variable('weights', W_shape, reuse=reuse)
        b = _variable('bias', b_shape, reuse=reuse, init=tf.zeros_initializer())
        out = tf.nn.atrous_conv2d(x, W, rate, padding) + b
        out.W = W
        out.b = b

        return out


# (kw-1)*stride + curr_w
# def dialated_conv2d(x, ch_out, khw, rate, padding, name='', reuse=None):
def dialated_deconv2d(x, ch_out, khw, rate, padding, name='', reuse=None):
    vs = 'dialated_deconv2d/'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs, reuse=reuse) as scope:
        bs = tf.shape(x)[0]
        hh, ww = [_static_size(x, ii) for ii in [1,2]]
        ch_in = _static_size(x, 3)

        kh, kw = khw
        W_shape = [kh, kw, ch_in, ch_out]
        b_shape = [ch_out]
        W = _variable('weights', W_shape, reuse=reuse)
        b = _variable('bias', b_shape, reuse=reuse, init=tf.zeros_initializer())

        if padding == 'VALID':
            raise ValueError('only padding="SAME" currently implemented')
            # # FIXME: ugly 1d hack
            # # new_h = hh + (scale_h*2)
            # new_h = hh
            # new_w = (kw-1)*scale + ww
            # # new_w = ww + (scale_w*2)
        elif padding == 'SAME':
            new_h, new_w = hh, ww
        else:
            raise ValueError('invalid padding value; must be "VALID" or "SAME"')

        out = tf.nn.atrous_conv2d_transpose(x, W,
                                            output_shape=tf.stack([bs, new_h, new_w, ch_out]),
                                            rate=rate,
                                            padding=padding) + b
        out.set_shape((None, new_h, new_w, ch_out))
        out.W = W
        out.b = b
        return out


def conv2d(x, ch_out, khw, stride_hw, padding, name='', reuse=None):
    vs = 'conv2d/'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs, reuse=reuse) as scope:
        stride_h, stride_w = stride_hw
        stride = [1, stride_h, stride_w, 1]
        kh, kw = khw
        ch_in = _static_size(x, 3)

        W_shape = [kh, kw, ch_in, ch_out]
        b_shape = [ch_out]

        W = _variable('weights', W_shape, reuse=reuse)
        b = _variable('bias', b_shape, reuse=reuse, init=tf.zeros_initializer())

        out = tf.nn.conv2d(x, W, stride, padding) + b
        out.W = W
        out.b = b

        return out


def deconv2d(x, ch_out, khw, scale_hw, name='', reuse=None):
    vs = 'deconv2d/'
    if name: vs = name + '/' + vs
    with tf.variable_scope(vs) as scope:
        bs = tf.shape(x)[0]
        hh, ww = [_static_size(x, ii) for ii in [1,2]]
        ch_in = _static_size(x, 3)

        scale_h, scale_w = scale_hw
        kh, kw = khw

        # NOTE: be mindful of ch order
        W_shape = [kh, kw, ch_out, ch_in]
        b_shape = [ch_out]

        W = _variable('weights', W_shape, reuse=reuse)
        b = _variable('bias', b_shape, reuse=reuse, init=tf.zeros_initializer())

        kh, kw = khw
        new_h, new_w = scale_hw[0]*hh, scale_hw[1]*ww

        output_shape = [bs, new_h, new_w, ch_out]
        out = tf.nn.conv2d_transpose(x, W, output_shape, [1, scale_h, scale_w, 1], padding='SAME') + b
        _set_static_size_(out, 1, new_h)
        _set_static_size_(out, 2, new_w)

        out.W = W
        out.b = b
        return out


def logit(probs, pmin=1e-6, pmax=1.-1e-6, name=None):
    probs = tf.clip_by_value(probs, pmin, pmax)
    logits = -tf.log((1./probs) - 1., name=name)
    return logits

