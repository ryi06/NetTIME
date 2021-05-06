import glob
import json
import logging
import os
import pickle
import time

import torch
import torch.nn as nn


########################
# CLASSES
########################
class AverageMeter(object):
    """
    Computes and stores average loss and scores; used for validation and
    prediction.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class CumulativeMeter(object):
    """Computes and stores cumulative training loss and scores."""

    def __init__(self, memory_gate):
        assert memory_gate <= 1 and memory_gate >= 0
        self.memory_gate = memory_gate
        self.cumulate = None

    def update(self, val):
        if self.cumulate is None:
            self.cumulate = val
        self.cumulate = (
            self.memory_gate * self.cumulate + (1 - self.memory_gate) * val
        )


class TimeTracker(object):
    """Track time."""

    def __init__(self):
        self.start_time = time.time()
        self.prev_time = time.time()

    def elapse(self):
        elapse = time.time() - self.start_time
        return time.strftime("%d-%H:%M:%S", time.gmtime(int(elapse)))

    def interval(self):
        interval = time.time() - self.prev_time
        self.prev_time = time.time()
        return time.strftime("%d-%H:%M:%S", time.gmtime(int(interval)))


class CheckpointTracker(object):
    """Track checkpoint files need to be evaluated."""

    def __init__(self, search_pattern):
        self.search_pattern = search_pattern
        self.__initialize_params()

    def __initialize_params(self):
        self.evaluated = set()
        self.reset_params()

    def reset_params(self):
        self.all_ckpts = set(glob.glob(self.search_pattern))
        self.remaining = self.all_ckpts - self.evaluated

    def add_evaluated(self, ckpt):
        self.evaluated.add(ckpt)


########################
# FUNCTIONS
########################
def load_pretrained_model(model, model_path):
    """Load a pretrained MultiBPE model"""
    params = torch.load(model_path)
    model.load_state_dict(params["state_dict"])
    return model


def reverse_embedding(index_file, merge=False):
    """ Reverse index_file to retrieve name from indices."""
    forward = pickle.load(open(index_file, "rb"))
    reverse = {}
    for key in forward.keys():
        reverse[key] = {v: k for k, v in forward[key].items()}
    if merge:
        merged = {}
        for key in forward.keys():
            merged[key] = {**forward[key], **reverse[key]}
        return merged
    else:
        return reverse


def set_logger(model_dir, model_name, mode, dtype, overwrite=True):
    """ Set up logging console and handler """
    fmt = logging.Formatter("[ %(asctime)s ] %(message)s", "%m/%d/%Y %H:%M:%S")
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_name = "{}_{}_{}.log".format(model_name, dtype.lower(), mode.lower())
    log_file = os.path.join(model_dir, "log", file_name)
    create_dirs(os.path.dirname(log_file))
    if overwrite and os.path.exists(log_file):
        os.remove(log_file)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.info("log file saved in {}".format(log_file))
    return logger


def create_dirs(directory, logger=None):
    """Create directory, display warning msg if directory already exists."""
    if os.path.exists(directory):
        msg = (
            "Directory {} already exists. Information may be overwritten."
        ).format(directory)
    else:
        os.makedirs(directory)
        msg = "Create directory {}".format(directory)
    print(msg) if logger is None else logger.info(msg)


def init_model_dir(output_dir, experiment_name):
    """Initialize model directory."""
    return os.path.join(output_dir, experiment_name)


def display_args(args, logger):
    """Display input argument."""
    logger.info(
        "CONFIG:\n{}".format(json.dumps(vars(args), indent=4, sort_keys=True))
    )
