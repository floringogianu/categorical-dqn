""" Functions and classes for parsing config files and command line arguments.
"""
import argparse
import yaml


def parse_cmd_args():
    """ Return parsed command line arguments.
    """
    p = argparse.ArgumentParser(description='')
    p.add_argument('-cf', '--config_file', type=str, default="dev",
                   metavar='config_file_name::str', help='Config file name.')
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
    config_path = "./configs/%s.yaml" % args.config_file
    f = open(config_path)
    config_data = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    return to_namespace(config_data)
