# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf

from src.models.base import BaseModel


def get_dataflow():
    pass


def main(params, checkpoint_dir, recover=True, force=False):
    action = 'd' if force else 'k' if recover else None
    tp.logger.set_logger_dir(checkpoint_dir, action=action)

    dataflow_params = params['dataflow']
    model_params = params['model']
    trainer_params = params['trainer']

    train_ds = get_dataflow()

    params.to_file(os.path.join(tp.logger.get_logger_dir(), 'config.json'))

    model = BaseModel.from_params(model_params)

    trainer = tp.SyncMultiGPUTrainerParameterServer(
        gpus=trainer_params['num_gpus'], ps_device=None)
    train_config = tp.AutoResumeTrainConfig(
        always_resume=recover,
        model=model,
        data=train_ds,
        callbacks=callbacks,
        extra_callbacks=None,
        monitors=None,
        steps_per_epoch=trainer_params["steps_per_epoch"],
        max_epoch=trainer_params["max_epochs"]
    )

    tp.launch_train_with_config(train_config, trainer)
