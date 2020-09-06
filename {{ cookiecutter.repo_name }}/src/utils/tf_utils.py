import os
import re
import tensorflow as tf

from tensorflow.python.training import training, saver
from tensorflow.python.platform import gfile


def tf_float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tf_float_array_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_int64_array_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_example(feature_dict):
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def get_model_filenames(model_dir, regex=r'(^model-[\w\- ]+.ckpt-(\d+))'):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        print('No meta file found in the model directory (%s)' % model_dir)
        return None, None
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(regex, f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None, session=None):
    """ Check if the model is a model directory
    (containing a metagraph and a checkpoint file)
    or if it is a protobuf file with a frozen graph """
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(
            os.path.join(model_exp, meta_file), input_map=input_map)
        if session:
            saver.restore(session,
                          os.path.join(model_exp, ckpt_file))
        else:
            saver.restore(tf.get_default_session(),
                          os.path.join(model_exp, ckpt_file))


def get_tensor_variables_from_checkpoint(checkpoint_dir):
    """Returns list of all variables in the latest checkpoint.
    Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    Returns:
    List of tuples `(name, shape)`.
    """
    if gfile.IsDirectory(checkpoint_dir):
        checkpoint_file = saver.latest_checkpoint(checkpoint_dir)
    if not checkpoint_file:
        raise ValueError("Couldn't find 'checkpoint' file or checkpoints in "
                         "given directory %s" % checkpoint_dir)

    reader = training.NewCheckpointReader(checkpoint_file)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    result = []
    for name in names:
        result.append((name, variable_map[name]))
    return result


def get_all_scopes_from_checkpoint(checkpoint_dir):
    var_list = list_variables(checkpoint_dir)
    scopes = set()
    for name, dim in var_list:
        scope = name.split('/')[0]
        scopes.add(scope)
    return scope
