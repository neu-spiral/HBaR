import os
from .path import *
import glob

def code_name(task, ttype, dtype, idx):
    if idx:
        filename = "{}-{}-{}-{:04d}.npy".format(task, ttype, dtype, idx)
    else:
        filename = "{}-{}-{}.npy".format(task, ttype, dtype)
    return filename

def get_log_filepath(filename, idx=None):
    filepath = "{}/assets/logs/{}".format(os.getcwd(), filename)
    return filepath

def get_model_path(filename, idx=None):
    if idx:
        filepath = "{}/assets/models/{}-{:04d}.pt".format(
            os.getcwd(), os.path.splitext(filename)[0], idx)
    else:
        filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath

