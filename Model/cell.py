import tensorflow as tf

class ConvGRUCell(tf.contrib.rnn.GRUCell):
    """
    Implement a GRU Cell
    """

    def __init__(self, num_units, filter_sizes, num_filters, embedding_size, reuse=None):
        super(ConvGRUCell, self).__init__(num_units=num_units, reuse=reuse, activation=tf.nn.relu)
        self._filter_sizes = filter_sizes
        self._num_filters = num_filters
        self._embedding_size = embedding_size

    def convolution(self, inputs):
        """
        Performs the convolution of the output
        """
        pooled_outputs = []
        print('Inputs', inputs)
        for i, filter_size in enumerate(self._filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % self._num_filters):
                # Convolution Layer
                conv = tf.layers.conv1d(
                    inputs,
                    filters=self._num_filters,
                    kernel_size=filter_size,
                    use_bias=True,
                    activation=tf.nn.relu,
                    name="conv-{i}".format(i=str(i)))
                print('Conv layer', conv)
                # Max-pooling over the outputs
                print('Input shape', inputs.get_shape())
                pooled = tf.layers.max_pooling1d(
                    conv,
                    pool_size=[conv.get_shape()[1]],
                    strides=filter_size,
                    name="pool")
                print('Pooled layer', pooled)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self._num_filters * len(self._filter_sizes)
        h_pool = tf.concat(pooled_outputs, 2)
        print('h_pool', h_pool)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print('h_pool_flat', h_pool_flat)
        return h_pool_flat

    def __call__(self, inputs, state, scope=None):
        """
        Extend the call method to first make a convolution of the inputs
        """
        inputs_convoluted = self.convolution(inputs)
        return super(ConvGRUCell, self).__call__(inputs=inputs_convoluted, state=state)
