# -*- coding: utf-8 -*-

import gzip
import json
import os
import shutil
import datetime
import pickle
import codecs

def main_directory():
    """
    Returns root path of project/package
    """
    return os.path.join(os.path.abspath(__file__).split('FUI')[0],'FUI')

def dump_pickle(folder_path, file_name, df, verbose=False):
    """
    Function that pickles df in folder_path\file_name (creates folder_path if doesn't exist)
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, file_name), 'wb') as f_out:
        pickle.dump(df, f_out)
        if verbose:
            print("Wrote file '{}' with shape {} to disc".format(file_name, df.shape))


def dump_csv(folder_path, file_name, df, verbose=False):
    """
    Function that outputs df as csv in folder_path\file_name (creates folder_path if doesn't exist)
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df.to_csv(os.path.join(folder_path, file_name+'csv'), sep=',')
    if verbose:
        print("Wrote file '{}' with shape {} to disc".format(file_name, df.shape))


def flatten(mylist):
    newlist = [item for sublist in mylist for item in sublist]
    return newlist


def get_files_list(folder, suffix=''):
    """
    Function that returns list of files in folder ending with *.suffix
    """
    if suffix:
        ls_out = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and
                  f[-len(suffix):] == suffix]
    else:
        ls_out = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(ls_out)


def unzip_files(folder, suffix='.gz'):
    """
    A short utility for unzipping all files (e.g. .gz) in a folder
    """
    files_list = get_files_list(folder, suffix)
    for f in files_list:
        with gzip.open(os.path.join(folder, f), 'rb') as f_in:
            with open(os.path.join(folder, f)[:-len(suffix)], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def params(ls_paths=['scripts', 'input_params.json'], reset=False):
    """
    Function for loading parameters for the entire workflow relative to \\FUI\\
    """

    if reset is True:
        _Singleton._instance = None

    if _Singleton._instance is None:
        input_file = os.path.abspath(os.path.join(__file__, '..', '..', '..', *ls_paths))
        try:
            with codecs.open(input_file, 'r', encoding='utf-8-sig') as f:
                input_json = json.load(f)
                _Singleton._instance = _Singleton(input_json)
        except FileNotFoundError as e:
            print(e)
    return _Singleton._instance


class _Singleton:
    _instance = None

    def __init__(self, input_json):
        input_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        self.filenames = input_json['filenames']
        self.options = input_json['options']
        self.dicts = input_json['dicts']
        self.paths = {key: os.path.join(input_path, path) for key, path in input_json['paths'].items()}
        for _, folder_path in self.paths.items():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

def define_NB_colors():
    """
    Defines Nationalbankens' colors and update matplotlib to use those as default
    """
    c = cycler(
        'color',
        [
            (0/255, 123/255, 209/255),
            (146/255, 34/255, 156/255),
            (196/255, 61/255, 33/255),
            (223/255, 147/255, 55/255),
            (176/255, 210/255, 71/255),
            (102/255, 102/255, 102/255)
        ])
    plt.rcParams["axes.prop_cycle"] = c
    return c