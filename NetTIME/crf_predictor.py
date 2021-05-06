import json
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcrf import CRF
from tqdm import tqdm

from .dataset import (
    CRF_collate_samples,
    CRF_push_to_device,
    CRFDataset,
    merge_predictions,
)
from .utils import *


class CRFPredictWorkflow(object):
    def __init__(self):
        self.__MODE = "CRF_PREDICT"
        self.__METRIC = "loss_avg"

        # Prediction parameters
        self.batch_size = None
        self.num_workers = None
        self.seed = None
        self.model_config = None
        self.best_ckpt = None

        # Data
        self.prediction_dir = None
        self.dtype = None
        self.class_weight = None

        # Save
        self.output_dir = None
        self.experiment_name = None
        self.result_dir = None
        self.tmp_dir = None

        # Args
        self.args = None

    def run(self):
        """Run CRF prediction workflow."""
        # Initialize model dir and display configurations.
        self.model_dir = init_model_dir(self.output_dir, self.experiment_name)
        self.logger = set_logger(
            self.model_dir, self.experiment_name, self.__MODE, self.dtype
        )
        display_args(self.args, self.logger)

        # Set up GPUs
        if not torch.cuda.is_available():
            raise Exception("No GPU found.")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda")
        self.logger.info(
            "Use {} GPU(s) for training".format(torch.cuda.device_count())
        )

        # Initialize model
        self.load_model()
        self.logger.info("MODEL ARCHITECTURE:\n{}".format(self.model))

        # Create prediction folder and load checkpoint.
        self.initialize_result_directories()
        self.load_checkpoint()

        # Start prediction.
        self.logger.info("Start CRF prediction.")
        try:
            self.crf_predict()
            self.logger.info("CRF prediction finished.")
        except KeyboardInterrupt:
            self.logger.warning("Prediction interrupted. Program exit.")

    ########################
    # Multiprocessing : main thread
    ########################
    def crf_predict(self):
        """Make predictions for a set of conditions using a CRF checkpoint."""
        self.time_tracker = TimeTracker()

        # Initialize validation data processes
        preprocess_inqueue = mp.JoinableQueue(maxsize=8)
        preprocess_outqueue = mp.JoinableQueue(maxsize=32)
        preprocess_worker = mp.Process(
            name="preprocess",
            target=self.preprocess,
            args=(preprocess_inqueue, preprocess_outqueue),
        )
        preprocess_worker.start()

        # Set up dataloader
        params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": CRF_collate_samples,
        }
        file_path, group_names = merge_predictions(
            self.prediction_dir, self.logger, self.tmp_dir
        )

        # Predict
        for group_name in group_names:
            # Initialize DataLoader
            predict_dset = CRFDataset(
                file_path,
                self.class_weight,
                normalizer=self.normalizer,
                group_name=group_name,
            )
            predict_iter = DataLoader(predict_dset, **params)
            num_batches = len(predict_iter)
            preprocess_inqueue.put(predict_iter)

            # Evaluate
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
            for b in tqdm(range(num_batches)):
                dset = preprocess_queue.get()
                feature, _ = CRF_push_to_device(dset, self.device)
                classified = torch.tensor(self.model.decode(feature.detach()))
                evaluate_queue.put(classified.cpu())
        evaluate_queue.join()

    ########################
    # Multiprocessing : workers
    ########################
    # Data preprocessing
    def preprocess(self, in_queue, out_queue):
        """Set up multiprocessing data queue"""
        while True:
            dataloader = in_queue.get()
            for dset in dataloader:
                out_queue.put(dset)

    def evaluate(self, group_name, queue, num_batches):
        """Combine classifications and save to disk."""
        classifications = []

        for batch in range(num_batches):
            classified = queue.get()
            queue.task_done()
            classifications.append(classified)

        # save predictions
        classifications = torch.cat(classifications).numpy()
        classification_path = os.path.join(
            self.pred_path, "{}.{}.npz".format(group_name, self.__METRIC)
        )

        np.savez_compressed(
            classification_path,
            step=self.step,
            group_name=group_name,
            metric=self.__METRIC,
            classification=classifications,
        )

    ############################
    # Progress display and save
    ############################
    def display_group(self, group_name, num_samples):
        """Display group name."""
        self.logger.info(
            "{} {} | eval metric {} | step {} | group {:<15} | {:7d} samples "
            "| time elapse: {:>12} |".format(
                self.__MODE,
                self.dtype,
                self.__METRIC,
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
                self.best_ckpt_path = self.best_ckpt
            else:
                raise RuntimeError(
                    "Invalid extension for best checkpoint file "
                    "{}".format(self.best_ckpt)
                )
        else:
            path = os.path.join(
                self.model_dir,
                "{}_{}".format("VALIDATION".lower(), "CRF_EVALUATE".lower()),
                "best_checkpoint",
                "best_loss_avg.json",
            )
            self.read_ckpt_json(path)

    def read_ckpt_json(self, filename):
        """Load a checkpoint .json file."""
        with open(filename, "r") as infile:
            data = json.load(infile)
        self.best_ckpt_path = data["ckpt_path"]

    ############################
    # Loading model and checkpoint
    ############################
    def load_model(self):
        """Load args from .config file and initialize CRF classifier."""
        # Load model params
        if self.model_config is not None:
            path = self.model_config
        else:
            path = os.path.join(
                self.model_dir, "{}_crf.config".format(self.experiment_name)
            )
        params = torch.load(path, map_location="cpu")
        if self.class_weight is None:
            self.normalizer = params["normalizer"]
        else:
            self.normalizer = None

        # Initialize classifier
        self.model = CRF(params["output_size"], batch_first=True)
        self.model.to(self.device)

    def load_checkpoint(self):
        """Load CRF checkpoint state_dict."""
        ckpt_params = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt_params["state_dict"])
        self.step = ckpt_params["step"]
