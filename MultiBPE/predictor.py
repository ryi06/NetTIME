import json
import multiprocessing as mp
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    MultiBPEDataset,
    collate_samples,
    get_group_names,
    push_to_device,
)
from .model import MultiBPE
from .utils import *


class PredictWorkflow(object):
    """Define the MultiBPE prediction workflow."""

    def __init__(self):
        self.__MODE = "PREDICT"

        # Prediction parameters
        self.batch_size = None
        self.num_workers = None
        self.seed = None
        self.model_config = None
        self.best_ckpt = None
        self.eval_metric = None

        # Data
        self.output_key = None
        self.dataset = None
        self.dtype = None
        self.index_file = None
        self.exclude_groups = None
        self.include_groups = None
        self.no_target = None
        self.predict_groups = None

        # Save
        self.output_dir = None
        self.experiment_name = None
        self.result_dir = None

        # Args
        self.args = None

    def run(self):
        """Run prediction workflow."""
        # Initialize model dir and display configurations.
        self.model_dir = init_model_dir(self.output_dir, self.experiment_name)
        self.logger = set_logger(
            self.model_dir, self.experiment_name, self.__MODE, self.dtype
        )
        display_args(self.args, self.logger)

        # Set up GPU options
        if not torch.cuda.is_available():
            raise Exception("No GPU found.")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda")
        self.logger.info(
            "Use {} GPU(s) for prediction".format(torch.cuda.device_count())
        )

        # Initialize model
        self.load_model()
        self.softmax = nn.Softmax(dim=2)
        self.logger.info("MODEL ARCHITECTURE:\n{}".format(self.model))

        # Create prediction folder and load checkpoint.
        self.initialize_result_directories()
        self.load_checkpoint()

        # Start prediction.
        self.logger.info("Start prediction.")
        try:
            self.predict()
            self.logger.info("Prediction finished.")
        except KeyboardInterrupt:
            self.logger.warning("Prediction interrupted. Program exit.")

    ########################
    # Multiprocessing : main thread
    ########################
    def predict(self):
        """Make predictions for a set of conditions using a model checkpoint."""
        self.time_tracker = TimeTracker()

        # Get eligible conditions.
        group_names = get_group_names(
            self.dataset,
            self.output_key,
            self.exclude_groups,
            self.include_groups,
            self.no_target,
            self.predict_groups,
        )

        # Initialize data loading worker.
        preprocess_inqueue = mp.JoinableQueue(maxsize=8)
        preprocess_outqueue = mp.JoinableQueue(maxsize=64)
        preprocess_worker = mp.Process(
            name="preprocess",
            target=self.preprocess,
            args=(preprocess_inqueue, preprocess_outqueue),
        )
        preprocess_worker.start()

        # Set up dataloader
        embed_indices = reverse_embedding(self.index_file, merge=self.no_target)
        params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": collate_samples,
        }

        # Predict
        for group_name in group_names:
            predict_dset = MultiBPEDataset(
                self.dataset,
                embed_indices,
                self.output_key,
                group_name=group_name,
                ct_feature=self.ct_feature,
                tf_feature=self.tf_feature,
                no_target=self.no_target,
            )
            predict_iter = DataLoader(predict_dset, **params)
            num_batches = len(predict_iter)
            preprocess_inqueue.put((predict_iter))

            self.display_group(group_name, predict_dset.__len__())
            self.predict_group(group_name, preprocess_outqueue, num_batches)

        preprocess_worker.terminate()

    def predict_group(self, group_name, preprocess_queue, num_batches):
        """Make prediction for one group using a checkpoint."""
        evaluate_queue = mp.JoinableQueue(maxsize=64)
        evaluate_worker = mp.Process(
            name="evaluate_{}".format(group_name),
            target=self.evaluate,
            args=(group_name, evaluate_queue, num_batches),
        )
        evaluate_worker.start()

        self.model.eval()
        with torch.no_grad():
            for b in range(num_batches):
                dset = preprocess_queue.get()
                feature, _ = push_to_device(dset, self.device)
                pred = self.softmax(self.model(feature))
                evaluate_queue.put(pred.cpu())

        evaluate_queue.join()

    ########################
    # Multiprocessing : workers
    ########################
    # Preprocess worker
    def preprocess(self, in_queue, out_queue):
        """Retrieve data and populate data queue."""
        while True:
            dataloader = in_queue.get()
            for dset in dataloader:
                out_queue.put(dset)

    # Evaluate worker
    def evaluate(self, group_name, queue, num_batches):
        """Combine predictions and save to disk."""
        predictions = []

        for batch in range(num_batches):
            pred = queue.get()
            queue.task_done()
            predictions.append(pred)

        predictions = torch.cat(predictions).numpy()
        pred_path = os.path.join(
            self.pred_path, "{}.{}.npz".format(group_name, self.eval_metric)
        )
        np.savez_compressed(
            pred_path,
            step=self.step,
            group_name=group_name,
            metric=self.eval_metric,
            prediction=predictions,
            output_key=self.output_key,
        )

    ############################
    # Display and save predictions
    ############################
    def display_group(self, group_name, num_samples):
        """Display group name."""
        self.logger.info(
            "{} {} | eval metric {} | step {} | group {:<15} | {:7d} samples "
            "| time elapse: {:>12} |".format(
                self.__MODE,
                self.dtype,
                self.eval_metric,
                self.step,
                group_name,
                num_samples,
                self.time_tracker.elapse(),
            )
        )

    def initialize_result_directories(self):
        """Initialize prediction directory and path to best checkpoint."""
        if self.result_dir is not None:
            self.pred_path = self.result_dir
        else:
            self.pred_path = os.path.join(
                self.model_dir,
                "{}_{}".format(self.dtype.lower(), self.__MODE.lower()),
            )
        create_dirs(self.pred_path, logger=self.logger)

        if self.best_ckpt is not None:
            if self.best_ckpt.endswith(".json"):
                self.read_ckpt_json(self.best_ckpt)
            elif self.best_ckpt.endswith(".ckpt"):
                if self.eval_metric is None:
                    raise ValueError(
                        "--eval_metric can not be null when a "
                        ".ckpt file is used as best checkpoint."
                    )
                self.best_ckpt_path = self.best_ckpt
            else:
                raise RuntimeError(
                    "Invalid extension for best checkpoint file "
                    "{}".format(self.best_ckpt)
                )
        else:
            path = os.path.join(
                self.model_dir,
                "{}_{}".format("VALIDATION".lower(), "EVALUATE".lower()),
                "best_checkpoint",
                "best_aupr.json",
            )
            self.read_ckpt_json(path)

    def read_ckpt_json(self, filename):
        """Load a checkpoint .json file."""
        with open(filename, "r") as infile:
            data = json.load(infile)
        self.eval_metric = data["metric"]
        self.best_ckpt_path = data["ckpt_path"]

    ############################
    # Loading model and checkpoint
    ############################
    def load_model(self):
        """Load args from .config file and initialize model."""
        if self.model_config is not None:
            path = self.model_config
        else:
            path = os.path.join(
                self.model_dir, "{}.config".format(self.experiment_name)
            )
        params = torch.load(path, map_location=self.device)

        # Initialize model
        self.model = MultiBPE(params["args"])
        self.model.to(self.device)

        # Load model parameters
        self.ct_feature = params["args"].ct_feature
        self.tf_feature = params["args"].tf_feature

    def load_checkpoint(self):
        """Load model checkpoint state_dict."""
        ckpt_params = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt_params["state_dict"])
        self.step = ckpt_params["step"]
