import oneflow as flow

def _get_initializer():
    return flow.random_normal_initializer(mean=0.0, stddev=0.1)

def _get_regularizer(name):
    return None

def _prelu(inputs, data_format="NCHW", name=None):
    return flow.layers.prelu(
        inputs,
        alpha_initializer=flow.constant_initializer(0.25),
        alpha_regularizer=_get_regularizer("alpha"),
        shared_axes=[2, 3] if data_format == "NCHW" else [1, 2],
        name=name,
    )


def _batch_norm(
    inputs,
    epsilon,
    center=True,
    scale=True,
    is_training=True,
    data_format="NCHW",
    name=None,
):
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=3 if data_format == "NHWC" and len(inputs.shape) == 4 else 1,
        momentum=0.9,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=flow.zeros_initializer(),
        gamma_initializer=flow.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        moving_mean_initializer=flow.zeros_initializer(),
        moving_variance_initializer=flow.ones_initializer(),
        trainable=is_training,
        training=is_training,
        name=name,
    )

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num=1,
    data_format="NCHW",
    dilation_rate=1,
    activation=None,
    use_bias=False,
    weight_initializer=_get_initializer(),
    bias_initializer=flow.zeros_initializer(),
    weight_regularizer=_get_regularizer("weight"),
    bias_regularizer=_get_regularizer("bias"),
):
    return flow.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, groups=group_num, activation=activation, use_bias=use_bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer, kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer, name=name)

def residual_unit_v3(
    in_data, num_filter, stride, dim_match, bn_is_training, data_format, name
):

    suffix = ""
    use_se = 0
    bn1 = _batch_norm(
        in_data,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn1" % (name, suffix),
    )
    conv1 = _conv2d_layer(
        name="%s%s_conv1" % (name, suffix),
        input=bn1,
        filters=num_filter,
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn2 = _batch_norm(
        conv1,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn2" % (name, suffix),
    )
    prelu = _prelu(bn2, data_format=data_format, name="%s%s_relu1" % (name, suffix))
    conv2 = _conv2d_layer(
        name="%s%s_conv2" % (name, suffix),
        input=prelu,
        filters=num_filter,
        kernel_size=3,
        strides=stride,
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    bn3 = _batch_norm(
        conv2,
        epsilon=2e-5,
        is_training=bn_is_training,
        data_format=data_format,
        name="%s%s_bn3" % (name, suffix),
    )

    if dim_match:
        input_blob = in_data
    else:
        input_blob = _conv2d_layer(
            name="%s%s_conv1sc" % (name, suffix),
            input=in_data,
            filters=num_filter,
            kernel_size=1,
            strides=stride,
            padding="valid",
            data_format=data_format,
            use_bias=False,
            dilation_rate=1,
            activation=None,
        )
        input_blob = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=bn_is_training,
            data_format=data_format,
            name="%s%s_sc" % (name, suffix),
        )

    identity = flow.math.add(x=bn3, y=input_blob)
    return identity


def get_symbol(input_blob):
    filter_list = [64, 64, 128, 256, 512]
    num_stages = 4
    units = [3, 13, 30, 3]
    num_classes = 512
    bn_is_training = False
    data_format = "NHWC"
    if data_format.upper() == "NCHW":
        input_blob = flow.transpose(
        input_blob, name="transpose", perm=[0, 3, 1, 2]
    )
    input_blob = _conv2d_layer(
        name="conv0",
        input=input_blob,
        filters=filter_list[0],
        kernel_size=3,
        strides=[1, 1],
        padding="same",
        data_format=data_format,
        use_bias=False,
        dilation_rate=1,
        activation=None,
    )
    input_blob = _batch_norm(
        input_blob, epsilon=2e-5, is_training=bn_is_training, data_format=data_format, name="bn0"
    )
    input_blob = _prelu(input_blob, data_format=data_format, name="relu0")

    for i in range(num_stages):
        input_blob = residual_unit_v3(
            input_blob,
            filter_list[i + 1],
            [2, 2],
            False,
            bn_is_training=bn_is_training,
            data_format=data_format,
            name="stage%d_unit%d" % (i + 1, 1),
        )
        for j in range(units[i] - 1):
            input_blob = residual_unit_v3(
                input_blob,
                filter_list[i + 1],
                [1, 1],
                True,
                bn_is_training=bn_is_training,
                data_format=data_format,
                name="stage%d_unit%d" % (i + 1, j + 2),
            )
    body = _batch_norm(
            input_blob,
            epsilon=2e-5,
            is_training=False,
            data_format="NHWC",
            name="bn1"
    )
    body = flow.reshape(body, (body.shape[0], -1))
    pre_fc1 = flow.layers.dense(
            inputs=body,
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=_get_initializer(),
            bias_initializer=flow.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            trainable=True,
            name="pre_fc1",
    )
    fc1 = _batch_norm(
            pre_fc1,
            epsilon=2e-5,
            scale=True,
            center=True,
            is_training=False,
            data_format="NHWC",
            name="fc1",
    )
    return fc1
