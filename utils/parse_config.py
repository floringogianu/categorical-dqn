""" Functions and classes for parsing config files and command line arguments.
"""
import argparse
import yaml


def parse_cmd_args():
    """ Return parsed command line arguments.
    """
    p = argparse.ArgumentParser(description='')
    p.add_argument('-l', '--label', type=str, default="default_label",
                   metavar='label_name::str',
                   help='Label of the current experiment')
    p.add_argument('-id', '--id', type=int, default=0,
                   metavar='label_name::str',
                   help='Id of this instance running within the current' +
                        'experiment')
    p.add_argument('-cf', '--config', type=str, default="catch_dev",
                   metavar='path::str',
                   help='Path to the config file.')
    p.add_argument('-r', '--results', type=str, default="./results",
                   metavar='path::str',
                   help='Path of the results folder.')
    args = p.parse_args()
    return args


def to_namespace(d):
    """ Convert a dict to a namespace.
    """
    n = argparse.Namespace()
    for k, v in d.items():
        setattr(n, k, to_namespace(v) if isinstance(v, dict) else v)
    return n


def parse_config_file(args):
    f = open(args.config)
    config_data = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    return to_namespace(config_data)
