# -*- coding: utf-8 -*-
import os

from tensorpack.predict import PredictConfig
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.export import ModelExporter

from src.models.base import BaseModel


def main(params, checkpoint_path, output_dir, export_type):
    model_params = params['model']
    model = BaseModel.from_params(model_params)

    pred_config = PredictConfig(
        session_init=get_model_loader(checkpoint_path),
        model=model,
        input_names=['input'],
        output_names=['prediction'])
    if export_type == 'compact':
        checkpoint_name = os.path.split(checkpoint_path)[1]
        ModelExporter(pred_config).export_compact(
            os.path.join(output_dir, checkpoint_name + '.pb'))
    else:
        ModelExporter(pred_config).export_serving(
            os.path.join(output_dir, 'exported'))
