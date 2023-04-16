import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


class Residual(tf.keras.Model):
    def __init__(self,num_channels,use_lxlconv=False,strides=1):
        super(Residual, self).__init__()
        #
