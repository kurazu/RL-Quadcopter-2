from keras import layers


def dense(inputs, units):
    output = layers.Dense(units, use_bias=False)(inputs)
    output = layers.BatchNormalization()(output)
    output = layers.LeakyReLU(alpha=0.1)(output)

    return output
