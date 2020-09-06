import os

import mlflow
import tensorflow as tf

from tensorpack.callbacks import Callback


class MLflowLogging(Callback):
    def __init__(self, experiment_name, stats, params, artifacts=None,
                 trigger_every=1, tracking_uri="", google_auth_key=None):
        if not isinstance(stats, list):
            stats = [stats]
        self._stats = stats

        if artifacts is None:
            artifacts = []
        elif not isinstance(artifacts, list):
            artifacts = [artifacts]
        self.artifacts = artifacts

        self._trigger_every = trigger_every

        if google_auth_key:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_auth_key
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_obj = mlflow.start_run()

        mlflow.log_params(params)

    def _trigger(self):
        if self.epoch_num % self._trigger_every == 0:
            self._log_metrics()

    def _after_train(self):
        for artifact in self.artifacts:
            if not os.path.exists(artifact):
                print(f"Artifact '{artifact}' doesn't exsist")
                continue
            mlflow.log_artifact(artifact)

    def _log_metrics(self):
        m = self.trainer.monitors
        metrics = {k: m.get_latest(k) for k in self._stats}
        mlflow.log_metrics(metrics)


class LoadPretrainWeight(Callback):
    def __init__(self, checkpoint_path, scope=None):
        self._checkpoint_path = checkpoint_path
        self._scope = scope
        self._saver = None

    def _setup_graph(self):
        if self._scope:
            load_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)
        else:
            load_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES)

        self._saver = tf.train.Saver(load_weights)

    def _before_train(self):
        self._saver.restore(self.trainer.sess, self._checkpoint_path)
