# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorpack as tp

from src.utils.params import Registrable
from tensorpack.tfutils.summary import add_moving_summary
from tensorflow.python.framework.tensor_spec import TensorSpec


class BaseModel(tp.ModelDesc, Registrable):
    def __init__(self, l2_norm, learning_rate, optim):
        self.l2_norm = l2_norm
        self.learning_rate = learning_rate
        self.optim = optim

    def inputs(self):
        raise NotImplementedError

    def build_graph(self):
        raise NotImplementedError

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=self.learning_rate,
                             trainable=False)
        add_moving_summary(lr)
        if self.optim['name'].lower() == 'adam':
            return tf.train.AdamOptimizer(
                learning_rate=lr, beta1=self.optim['beta1'],
                beta2=self.optim['beta2'], epsilon=self.optim['epsilon'])
        else:
            raise ValueError('Unknown optimizer')
