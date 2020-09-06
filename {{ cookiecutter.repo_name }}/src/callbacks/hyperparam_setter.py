import numpy as np

from tensorpack.callbacks import HyperParamSetter


class ExponentialDecayHyperParamSetter(HyperParamSetter):
    def __init__(self, param, init_value, decay_steps, decay_rate=0.95,
                 staircase=False):
        super(ExponentialDecayHyperParamSetter, self).__init__(param)
        self._init_value = init_value
        self._decay_steps = decay_steps
        self._decay_rate = decay_rate
        self._staircase = staircase

    def _get_value_to_set(self):
        p = self.global_step / self._decay_steps
        if self._staircase:
            p = np.floor(p)
        return self._init_value * np.power(self._decay_rate, p)


# Learning rate finder implementation
class AutoIncrementHyperParamSetter(HyperParamSetter):
    def __init__(self, param, num_iterate, init_value=1e-7,
                 final_value=1):
        super(AutoIncrementHyperParamSetter, self).__init__(param)
        self._init_value = init_value
        self._final_value = final_value
        self._num_iterate = num_iterate

        self._lr = init_value
        self._mult = (final_value / init_value) ** (1 / num_iterate)

    def _get_value_to_set(self):
        self._lr = self._lr * self._mult
        return self._lr


# One-cycle policy implementation
class OneCycleHyperParamSetter(HyperParamSetter):
    def __init__(self, param, num_epoch, num_cycle,
                 max_value=3e-3, div_init=100):
        super(OneCycleHyperParamSetter, self).__init__(param)
        self._max_value = max_value
        self._num_cycle = num_cycle
        self._half_cycle = self._num_cycle / 2

        self._init_value = self._max_value / div_init
        self._lr = self._init_value
        self._cycle_mult = div_init ** (1 / self._half_cycle)

        assert num_epoch >= num_cycle
        self._remain_epoch = num_epoch - num_cycle
        self._remain_mult = 100 ** (1 / self._remain_epoch)

    def _get_value_to_set(self):
        if self.epoch_num <= self._half_cycle:
            self._lr = self._lr * self._cycle_mult
        elif self.epoch_num <= self._num_cycle:
            self._lr = self._lr / self._cycle_mult
        else:
            self._lr = self._lr / self._remain_mult
        return self._lr


class StaircaseHyperParamSetter(HyperParamSetter):
    def __init__(self, param, epochs, values):
        super(StaircaseHyperParamSetter, self).__init__(param)

        assert len(epochs) == len(values)
        self._epochs = sorted(epochs)
        self._values = values

    def _get_value_to_set(self):
        for i, epoch in enumerate(self._epochs):
            if self.epoch_num <= epoch:
                return self._values[i]
        return self._values[-1]