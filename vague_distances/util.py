"""This module provides a set of utility functions.
Most noteably this module includes a wrapper for an easy access to a database,
which requires an ssh connection

Author: Jan Greulich
Date: 09.02.2018
"""

import argparse
import warnings
import functools
import configparser
import os
import sys
import re
import pickle

from sshtunnel import SSHTunnelForwarder
from pymongo import MongoClient

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "config.ini"))

SSH_HOST = config["SSH"]["SSH_HOST"]
SSH_PORT = int(config["SSH"]["SSH_PORT"])
LDAP_USER_NAME = config["SSH"]["LDAP_USER_NAME"]
LDAP_PASSWORD = config["SSH"]["LDAP_PASSWORD"]
MONGODB_HOST = config["MONGODB"]["MONGODB_HOST"]
MONGODB_PORT = int(config["MONGODB"]["MONGODB_PORT"])
MONGODB_AUTHENTICATION_DB = config["MONGODB"]["MONGODB_AUTHENTICATION_DB"]
MONGODB_USER_NAME = config["MONGODB"]["MONGODB_USER_NAME"]
MONGODB_PASSWORD = config["MONGODB"]["MONGODB_PASSWORD"]

DATABASE = config["DATABASE"]["DATABASE"]
NEWSGT_EDGES = config["DATABASE"]["NEWSGT_EDGES"]
NEWSGT_TRIPLES = config["DATABASE"]["NEWSGT_TRIPLES"]
WIKIDATA = config["DATABASE"]["WIKIDATA"]
WIKIDATA_EN = config["DATABASE"]["WIKIDATA_EN"]



def adrastea(*args, **kwargs):
    """Wrapper for the automatic ssh connection to the specified SSH-Port and
    MongoDB. The connection details can be set in the config.ini file.

    Possible kwargs:

    :param extra_args: function that accepts a configparser and adds
        parser arguments
    :type extra_args: function(parser): return parser

    Usage is as simple as::

        from eventflow.util import adrastea

        @adrastea()
        def foo():
    """
    def adrastea_inner(func):
        def ssh_connect(*args2, **kwargs2):
            with SSHTunnelForwarder((SSH_HOST, SSH_PORT),
                                    ssh_username=LDAP_USER_NAME,
                                    ssh_password=LDAP_PASSWORD,
                                    remote_bind_address=('localhost', MONGODB_PORT),
                                    local_bind_address=('localhost', MONGODB_PORT)
                                    ) as server:
                print('Connected via SSH and established port-forwarding')
                client = MongoClient(MONGODB_HOST, MONGODB_PORT)
                try:
                    client[MONGODB_AUTHENTICATION_DB].authenticate(
                        MONGODB_USER_NAME, MONGODB_PASSWORD)
                    env = dict()
                    env['client'] = client
                    if "extra_args" in kwargs:
                        parser = argparse.ArgumentParser()
                        env.update(vars(kwargs["extra_args"](parser)))
                    print('Authenticated on mongodb')
                    print('-' * 70)
                    print('')
                    args2 = list(args2)
                    args2.append(env)
                    result = func(*args2, **kwargs2)
                finally:
                    client.close()
                    print('')
                    print('-' * 70)
                    print('Connection closed')

            return result
        return ssh_connect
    return adrastea_inner


def date_parser(series):
    years = series.apply(lambda x: int(x.split("-")[0]))
    months = series.apply(lambda x: int(
        x.split("-")[1]) if len(x.split("-")) > 1 else 0)
    days = series.apply(lambda x: int(
        x.split("-")[2]) if len(x.split("-")) > 2 else 0)
    return years, months, days


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    '''
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func
