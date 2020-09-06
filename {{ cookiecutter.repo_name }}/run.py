# -*- coding: utf-8 -*-
import os
import sys

import logging
import pkgutil
import argparse
import importlib


def import_submodules(package_name: str) -> None:
    importlib.invalidate_caches()
    sys.path.append('.')

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, '__path__', [])
    path_string = '' if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path
        # `pkgutil.import_submodules` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)


import_submodules('src')
from src.utils.params import Params
from src.utils.args_parser import Subcommand, ArgumentParserWithDefaults

logger = logging.getLogger(__name__)


def main(subcommand_overrides={}):
    parser = ArgumentParserWithDefaults()

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
        'train': Train(),
        'hyperparams_search': HyperparamsSearch(),
        'evaluate': Evaluate(),
        'export': ExportModel(),
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()
    if 'func' in dir(args):
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


class Train(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Train the specified model on the specified dataset'''
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            'param_path', type=str,
            help='path to parameter file describing the model to be trained')
        subparser.add_argument(
            '-s', '--serialization-dir', type=str, default='',
            help='directory in which to save the model and its logs')
        subparser.add_argument(
            '-r', '--recover', action='store_true',
            help='recover training from the state in serialization_dir')
        subparser.add_argument(
            '-f', '--force', action='store_true',
            help='force override serialization dir')

        subparser.set_defaults(func=train_model)

        return subparser


def train_model(args):
    from src.train import main as train

    params = Params.from_file(args.param_path)
    serialization_dir = args.serialization_dir
    if not serialization_dir:
        param_filename = os.path.splitext(os.path.split(args.param_path)[1])[0]
        serialization_dir = os.path.join('train_logs', param_filename)
    return train(params, serialization_dir, args.recover, args.force)


class HyperparamsSearch(Subcommand):
    def add_subparser(self, name, subparsers):
        description = 'Run hyperparams search'
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            'config_dir', type=str,
            help='path to directory contains config files')
        subparser.add_argument(
            'log_dir', type=str,
            help=('directory in which to save the model and its logs.'))
        subparser.add_argument(
            '-r', '--recover', action='store_true',
            help='recover training from the state in serialization_dir')
        subparser.add_argument(
            '-f', '--force', action='store_true',
            help='force override serialization dir')
        subparser.add_argument(
            '-d', '--display_logs', action='store_true',
            help=('Options to display logs in shell')
        )

        subparser.set_defaults(func=hyperparams_search)
        return subparser


def hyperparams_search(args):
    from src.hyperparams_search import main as hp_search
    return hp_search(args.config_dir, args.log_dir, args.recover, args.force, args.display_logs)


class Evaluate(Subcommand):
    def add_subparser(self, name, subparsers):
        description = 'Run evaluation'
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            'dataset_path', type=str,
            help='path to evaluate dataset')
        subparser.add_argument(
            'checkpoint_path', type=str,
            help=('directory to the model checkpoint'))
        subparser.add_argument(
            '-f', '--force', action='store_true',
            help='force override evaluate result')

        subparser.set_defaults(func=evaluate_model)
        return subparser


def evaluate_model(args):
    from src.evaluate import main as evaluate
    return evaluate(args.dataset_path, args.checkpoint_path, args.force)


class ExportModel(Subcommand):
    def add_subparser(self, name, parser):
        description = '''Export model for serving'''
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            'checkpoint_path', type=str, help='path to model checkpoint')
        subparser.add_argument(
            '-o', '--output_dir', type=str, help='path to output dir')
        subparser.add_argument(
            '-e', '--export_type', choices=['compact', 'serving'],
            default='compact', help='choose export mode')

        subparser.set_defaults(func=export_model)

        return subparser


def export_model(args):
    from src.export import main as export
    params_path = os.path.join(os.path.split(args.checkpoint_path)[0],
                               'config.json')
    params = Params.from_file(params_path)
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.split(args.checkpoint_path)[0]
    return export(params, args.checkpoint_path, output_dir, args.export_type)


def run():
    main()


if __name__ == "__main__":
    run()
