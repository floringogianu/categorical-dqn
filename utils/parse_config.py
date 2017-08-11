""" Functions and classes for parsing config files and command line arguments.
"""
import argparse
import yaml
import os
from termcolor import colored as clr


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
    p.add_argument('-r', '--results', type=str, default="./experiments",
                   metavar='path::str',
                   help='Path of the results folder.')
    args = p.parse_args()
    return args


def to_namespace(d, args):
    """ Convert a dict to a namespace.
    """
    n = argparse.Namespace()
    for k, v in d.items():
        setattr(n, k, to_namespace(v) if isinstance(v, dict) else v)

    # inject some of the cmdl args into the config namespace
    setattr(n, "experiment_id", args.id)
    setattr(n, "results_path", args.results)
    return n


def check_paths(cmdl):
    if not os.path.exists(cmdl.results_path):
        print(
            clr("%s path for saving results does not exist. Please create it."
                % cmdl.results_path, 'red', attrs=['bold']))
        raise IOError
    else:
        print(clr("Warning, data in %s will be overwritten."
              % cmdl.results_path, 'red', attrs=['bold']))


def parse_config_file(args):
    f = open(args.config)
    config_data = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    cmdl = to_namespace(config_data, args)
    check_paths(cmdl)
    return cmdl
