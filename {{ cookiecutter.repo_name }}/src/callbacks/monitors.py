import os
import numpy as np
import tensorflow as tf

from tensorpack.train import StopTraining
from tensorpack.callbacks import Callback


class EarlyStopping(Callback):
    def __init__(self, monitor_metric, mode, patient, min_delta=0,
                 baseline=None):
        assert mode in ['min', 'max']
        self._monitor_metric = monitor_metric
        self._patient = patient

        if mode == 'min':
            self._monitor_op = np.less
            self._min_delta = -min_delta
        else:
            self._monitor_op = np.greater
            self._min_delta = min_delta

        if baseline is None:
            self._best = np.Inf if mode == 'min' else -np.Inf
        else:
            self._best = baseline

        self._wait = 0

    def _trigger(self):
        curr_value = self.trainer.monitors.get_latest(self._monitor_metric)
        if self._monitor_op(curr_value - self._min_delta, self._best):
            self._best = curr_value
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patient:
                print('[EarlyStopping] Stop training at epoch %d' %
                      self.epoch_num)
                raise StopTraining()
