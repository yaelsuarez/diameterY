import tensorflow as tf
from keras.layers import Layer, InputLayer, Conv2D, MaxPool2D, UpSampling2D, Dropout, GlobalAveragePooling2D, Dense, BatchNormalization
from keras.activations import relu, sigmoid, get as get_activation
from keras import Model

class ConvBlock(Layer):
    def __init__(self, filters, kernel_size=(3,3), padding='same', activation='relu', **kwargs):
        """Implements the Conv-Block described in Fig 1.

        Args:
            filters: number of filters of the convolution
            kernel_size: size of the convolution kernel. Defaults to (3,3).
            padding (str, optional): padding. Defaults to 'same'.
            activation: activation after Batch Norm. Defaults to relu.
            name (str, optional): name for the layer.
        """
        super(ConvBlock, self).__init__(**kwargs)
        # --- store configuration for serialization
        self.filters=filters, 
        self.kernel_size=kernel_size,
        self.padding=padding,
        self.activation=activation,
        self.kwargs = kwargs
        # --- create needed layers
        self.conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)
        self.batch_norm = BatchNormalization()
        self.activation_fn = get_activation(activation)

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        return self.activation(x)

    def get_config(self):
        return dict(
            filters=self.filters, 
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
            **self.kwargs
        )

class AttentionBlock(Layer):
    def __init__(self, units, **kwargs):
        super(self, AttentionBlock).__init__(**kwargs)
        self.global_avg_pooling = GlobalAveragePooling2D()
        self.dense = Dense(units=units)

    def __call__(self, inputs):
        x = self.global_avg_pooling(inputs)
        x = self.dense(x)
        x = relu(x)
        x = sigmoid(x)
        return tf.multiply(inputs, x)

class Midblock(Layer):
    def __init__(self, filters, attention_blocks, contraction_factor=2, **kwargs):
        super(self, AttentionBlock).__init__(**kwargs)
        # --- store configuration for serialization
        self.filters = filters
        self.attention_blocks = attention_blocks
        self.contraction_factor = contraction_factor

        # --- create all necessary layers
        self.conv_block_in = ConvBlock(filters=filters)
        self.attn_blocks = []
        for i in range(attention_blocks):
            units = int(filters/(contraction_factor ** i)) # divided by 1, 2, 4, ...
            self.attn_blocks.append(AttentionBlock(units))
        self.conv_block_out = ConvBlock(filters=filters)

    def __call__(self, inputs):
        x_in = self.conv_block_in(inputs)
        x = x_in
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        x = self.conv_block_out(x)
        return tf.multiply(inputs, x_in)

class LWBNAUnet(Model):
    def __init__(self, n_classes, filters, depth, midblock_attn_blocks, dropout=0.3, **kwargs):
        super(self, LWBNAUnet).__init__(**kwargs)
        # --- input shape is (height, width, channels)
        self.dropout = dropout
        self.depth = depth

        self.conv_blocks = {}
        self.attn_blocks = {}
        for d in depth:
            self.conv_blocks[f'contract_{d}0'] = ConvBlock(filters=filters, name=f'contract_{d}0')
            self.conv_blocks[f'contract_{d}1'] = ConvBlock(filters=filters)
            self.attn_blocks[f'contract_{d}'] = AttentionBlock(filters) # Is this correct ??
            self.conv_blocks[f'expand_{d}0'] = ConvBlock(filters=filters)
            self.conv_blocks[f'expand_{d}1'] = ConvBlock(filters=filters)
            self.attn_blocks[f'expand_{d}'] = AttentionBlock(filters) # Is this correct ??
        self.conv_blocks['post_midblock_0'] = ConvBlock(filters=filters)
        self.conv_blocks['post_midblock_1'] = ConvBlock(filters=filters)
        self.attn_blocks['post_midblock'] = AttentionBlock(filters) # Is this correct ??
        self.midblock = Midblock(filters=filters, attention_blocks=midblock_attn_blocks)
        self.output_conv = Conv2D(filters=n_classes, kernel_size=1, activation='sigmoid')

    def __call__(self, inputs):
        def dropout_conv_conv_attn(x, dir, depth):
            x = Dropout(self.dropout)(x)
            x = self.conv_blocks[f'{dir}_{depth}0'](x)
            x = self.conv_blocks[f'{dir}_{depth}1'](x)
            return self.attn_blocks[f'{dir}_{depth}'](x)

        # store skip connections
        skip = []

        # first contraction
        x = self.conv_blocks['contract_00'](x)
        x = self.conv_blocks['contract_00'](x) 
        skip[0] = self.attn_blocks['contract_0'](x)
        # the first contraction comes directly from the second Conv-Block
        x = MaxPool2D(name="max_pool_contract_0")(x)

        # intermediate contractions
        for d in range(1, self.depth):
            x = dropout_conv_conv_attn(x, 'contract', d)
            skip[d] = x  # store for later
            x = MaxPool2D(name="max_pool_contract_0")(x)
        
        # bottle-neck
        x = Dropout(self.dropout)(x)
        x = self.midblock(x)
        x = self.conv_blocks['post_midblock0'](x)
        x = self.conv_blocks['post_midblock1'](x)
        x = self.attn_blocks['post_midblock'](x)
        x = UpSampling2D()(x)

        # intermediate expansions
        for d in reversed(range(1, self.depth)):
            x = tf.add(x, skip[d]) # add skip connection
            x = dropout_conv_conv_attn(x, 'expand', d)
            x = UpSampling2D()(x)
        
        # final expansion
        x = tf.add(x, skip[0]) # add skip connection
        x = dropout_conv_conv_attn(x, 'expand', 0)
        return self.output_conv(x)
        


            


