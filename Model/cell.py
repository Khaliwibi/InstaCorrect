import tensorflow as tf

class ConvLSTMCell(tf.contrib.rnn.LSTMCell):
    """
    Implement a GRU Cell
    """

    def __init__(self, num_units, window_sizes, num_filters, embedding_size, reuse=None):
        super(ConvLSTMCell, self).__init__(num_units=num_units, reuse=reuse, activation=tf.nn.relu)
        self._window_sizes = window_sizes
        self._num_filters = num_filters
        # self._embedding_size = embedding_size

    def convolution(self, inputs):
        """
        Performs the convolution of the output
        args:
            - inputs: a tensor of shape [batch_size, max_word_length, embedding_size]
            filled with embeddings.
        """
        # Array containing all the results of the convolution.
        pooled_outputs = []
        print('Inputs', inputs)
        # For each 
        for i, window_size in enumerate(self._window_sizes):
            with tf.name_scope("conv-maxpool-%s" % self._num_filters):
                # Convolution Layer. 
                # Inputs are [batch_size, max_word_length, embedding_size]
                # Should apply a 1D convolution with padding = "SAME" to have
                # the same size as the input. Will return a tensor of shape
                # [batch_size, max_word_length, num_filters] on which we apply
                # a max over time pooling -> i.e. a reduce_max over the second
                # dimension.
                conv = tf.layers.conv1d(
                    inputs,
                    filters=self._num_filters, # the number of filters to apply
                    kernel_size=[window_size], # the kernel size.
                    use_bias=True,
                    padding="SAME",
                    activation=tf.nn.relu,
                    name="conv-{i}".format(i=str(i)))
                print('Conv layer', conv)
                print('Input shape', inputs.get_shape())
                # Max-pooling over the outputs. Just take the maximum for each
                # filter along the max_word_length dimension. Will return a tensor
                # of shape [batch_size, num_filters] -> so identitical whatever
                # the word length. Clever Kim.
                max_pooled = tf.reduce_max(conv, 1) 
                print('Max pooled', max_pooled)
#                pooled = tf.layers.max_pooling1d(
#                    conv,
#                    pool_size=[conv.get_shape()[1]],
#                    strides=filter_size,
#                    name="pool")
#                print('Pooled layer', pooled)
                # Append the result to the global results.
                pooled_outputs.append(max_pooled)
        # Combine all the pooled features
        # We now have an array of tensors of the shape [batch, num_filters]
        # Concatenate them to have [batch, num_filters*len(window_size)]
        # To be sure to understand, we now have each word embedded as vector
        # of size num_filters*len(window_size)
        concatenated_filters = tf.concat(pooled_outputs, 1)
        print('concaneted_filters', concatenated_filters)
        #num_filters_total = self._num_filters * len(self._filter_sizes)
        #☺print('h_pool', h_pool)
        #h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        #print('h_pool_flat', h_pool_flat)
        return concatenated_filters

    def __call__(self, inputs, state, scope=None):
        """
        Extend the call method to first make a convolution of the inputs
        """
        inputs_convoluted = self.convolution(inputs)
        return super(ConvLSTMCell, self).__call__(inputs=inputs_convoluted, state=state)
