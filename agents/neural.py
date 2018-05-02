from keras import layers


def dense(
    inputs, units, name=None, activation='lrelu', batch_normalization=True
):

    output = layers.Dense(units, use_bias=not batch_normalization)(inputs)

    if batch_normalization:
        output = layers.BatchNormalization()(output)

    if activation == 'lrelu':
        output = layers.LeakyReLU(alpha=0.2)(output)
    elif activation is None:
        pass
    else:
        output = layers.Activation(activation)(output)

    return output
