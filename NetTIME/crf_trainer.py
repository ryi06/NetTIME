import json
import multiprocessing as mp
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchcrf import CRF

from .calculate_metrics import ScoreTracker
from .dataset import (NetTIMEDataset, Normalizer, collate_samples,
                      push_to_device)
from .model import NetTIME
from .utils import *


class CRFTrainWorkflow(object):
    def __init__(self):
        self.__MODE = "CRF_TRAIN"

        ######## Training parameters ########
        self.batch_size = None
        self.num_epochs = None
        self.num_workers = None
        self.nettime_config = None
        self.nettime_ckpt = None
        self.start_from_checkpoint = None

        self.learning_rate = None
        self.weight_decay = None
        self.seed = None
        self.loss_avg_ratio = None

        # Data
        self.dataset = None
        self.dtype = None
        self.class_weight = None
        self.output_key = None
        self.index_file = None
        self.exclude_groups = None
        self.include_groups = None

        # Display and save
        self.print_every = None
        self.evaluate_every = None
        self.output_dir = None
        self.experiment_name = None
        self.result_dir = None
        self.ckpt_dir = None

        # Args
        self.args = None

    def run(self):
        """Setting up CRF training run."""
        # Initialize model dir and display configurations.
        self.model_dir = init_model_dir(self.output_dir, self.experiment_name)
        self.logger = set_logger(
            self.model_dir, self.experiment_name, self.__MODE, self.dtype
        )
        display_args(self.args, self.logger)

        # Set up GPUs
        if not torch.cuda.is_available():
            raise Exception("No GPU found.")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda")
        self.logger.info(
            "Use {} GPU(s) for training".format(torch.cuda.device_count())
        )

        # Load NetTIME
        self.load_NetTIME_checkpoint()
        self.logger.info("NetTIME ARCHITECTURE:\n{}".format(self.nettime))

        # Load data
        self.logger.info("Loading data.")
        self.embed_indices = reverse_embedding(self.index_file)
        self.softmax = nn.Softmax(dim=2)
        self.normalizer = Normalizer(self.class_weight, device=self.device)

        params = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "collate_fn": collate_samples,
        }
        dset = NetTIMEDataset(
            self.dataset,
            self.embed_indices,
            self.output_key,
            ct_feature=self.ct_feature,
            tf_feature=self.tf_feature,
            exclude_groups=self.exclude_groups,
            include_groups=self.include_groups,
        )
        self.dset_iter = DataLoader(dset, **params)
        self.logger.info(
            "{:,} samples are used for training.".format(dset.__len__())
        )

        # Initialize model
        self.model = CRF(self.output_size, batch_first=True)
        if self.start_from_checkpoint is not None:
            self.model = load_pretrained_model(
                self.model, self.start_from_checkpoint
            )
            self.logger.info(
                "Training from existing model {}".format(
                    self.start_from_checkpoint
                )
            )
        self.model.to(self.device)
        self.logger.info("MODEL ARCHITECTURE:\n{}".format(self.model))

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Create training result folders.
        self.initialize_result_directories()

        # Start training
        self.logger.info("Start training CRF classifier.")
        try:
            self.crf_train()
            self.logger.info("CRF training finished.")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted. Program exit.")

    ########################
    # Multiprocessing : main thread
    ########################
    def crf_train(self):
        """Train a CRF classifier with multiprocessing."""
        # Initialize training data processes
        preprocess_queue = mp.JoinableQueue(maxsize=64)
        preprocess_worker = mp.Process(
            name="preprocess",
            target=self.preprocess,
            args=(preprocess_queue, self.dset_iter, self.num_epochs),
        )
        preprocess_worker.start()
        self.logger.info("Training data workder started.")

        # Initialize progress displaying processes
        num_batches = len(self.dset_iter)
        evaluate_queue = mp.JoinableQueue(maxsize=64)
        evaluate_worker = mp.Process(
            name="evaluate",
            target=self.evaluate,
            args=(evaluate_queue, num_batches),
        )
        evaluate_worker.start()
        self.logger.info("Training scoring workder started.")

        self.nettime.eval()
        self.model.train()
        for epoch in range(self.num_epochs):
            for b in range(num_batches):
                # Generate NetTIME predictions
                step, dset = preprocess_queue.get()
                preprocess_queue.task_done()
                pred, target = self.generate_predictions(dset)

                # Compute CRF loss
                self.model.zero_grad()
                loss = -self.model(pred, target, reduction="mean")
                loss.backward()
                self.optimizer.step()

                # Update and display training loss
                if step == 0 or step % self.print_every == 0:
                    data = (loss.item(), step, epoch, b)
                    evaluate_queue.put(("UPDATE_LOSS", data))

                # Calculate scores and save chedckpoint
                save_every = self.print_every * self.evaluate_every
                if step == 0 or step % save_every == 0:
                    binary = torch.tensor(self.model.decode(pred.detach()))
                    data = (binary, target.cpu(), loss.item(), step, epoch, b)
                    evaluate_queue.put(("UPDATE_SCORE", data))
                    self.save_checkpoint(step)

        # Terminate workers
        preprocess_worker.terminate()
        evaluate_queue.join()
        evaluate_worker.terminate()

    def generate_predictions(self, dset):
        """Generate NetTIME predictions."""
        with torch.no_grad():
            # Make predictions
            feature, target = push_to_device(dset, self.device)
            predictions = self.softmax(self.nettime(feature))
            normalized = self.normalizer.normalize(predictions)
            return normalized, target

    ########################
    # Multiprocessing : workers
    ########################
    # Preprocess worker
    def preprocess(self, queue, dataloader, num_epochs):
        """Set up multiprocessing data queue"""
        step = 0
        for epoch in range(num_epochs):
            for dset in dataloader:
                queue.put((step, dset))
                step += 1
        queue.join()

    # Evaluate worker
    def evaluate(self, queue, num_batches):
        """Compute cumulative loss, and save evaluations and checkpoints."""
        self.initiate_avg_trackers()
        self.time_tracker = TimeTracker()
        self.score_tracker = ScoreTracker(self.__MODE)

        while True:
            mode, data = queue.get()
            if mode == "UPDATE_LOSS":
                self.update_loss(data, num_batches)
            elif mode == "UPDATE_SCORE":
                self.update_score(data, num_batches)
            queue.task_done()

    ########################
    # Tracking evaluations
    ########################
    def initiate_avg_trackers(self):
        """Initialize cumulative loss and score trackers."""
        # Losses
        self.loss_avg = CumulativeMeter(self.loss_avg_ratio)

        # Scores
        self.iou_avg = CumulativeMeter(self.loss_avg_ratio)
        self.aupr_avg = CumulativeMeter(self.loss_avg_ratio)

    def update_loss(self, data, num_batches):
        """Update and display cumulative training loss."""
        loss, step, epoch, batch = data
        self.loss_avg.update(loss)
        self.display_loss(epoch, batch, num_batches)

    def update_score(self, data, num_batches):
        """Update and display cumulative training scores."""
        classified, target, loss, step, epoch, batch = data
        scores = self.score_tracker.calculate_scores(classified, target)
        if scores is not None:
            iou, aupr = scores
            self.iou_avg.update(iou)
            self.aupr_avg.update(aupr)
        self.save_evaluation(step, epoch, batch, loss)
        self.display_evaluation(step, epoch, batch, num_batches)

    ############################
    # Progress display
    ############################
    def display_loss(self, epoch, batch, num_batches):
        """Display training loss."""
        self.logger.info(
            "| epoch {:3d} | {:6d}/{:6d} batches | average loss {:8.4f} "
            "| time interval: {:>12} |".format(
                epoch,
                batch,
                num_batches,
                self.loss_avg.cumulate,
                self.time_tracker.interval(),
            )
        )

    def display_evaluation(self, step, epoch, batch, num_batches):
        """Display cumulative training loss and scores."""
        self.logger.info(
            "{} {} | step {:7d} | epoch {:3d} | {:6d}/{:6d} batches "
            "| average loss {:8.3f} | iou {:4.3f} | aupr {:4.3f} "
            "| time elapse: {:>12} |".format(
                self.__MODE,
                self.dtype,
                step,
                epoch,
                batch,
                num_batches,
                self.loss_avg.cumulate,
                self.iou_avg.cumulate,
                self.aupr_avg.cumulate,
                self.time_tracker.elapse(),
            )
        )

    ############################
    # Save CRF checkpoints
    ############################
    def initialize_result_directories(self):
        """Initialize output evaluation and checkpoint directories."""
        if self.result_dir is not None:
            self.eval_path = self.result_dir
        else:
            self.eval_path = os.path.join(
                self.model_dir,
                "{}_{}".format(self.dtype.lower(), self.__MODE.lower()),
            )
        create_dirs(self.eval_path, logger=self.logger)

        if self.ckpt_dir is not None:
            self.ckpt_path = self.ckpt_dir
        else:
            self.ckpt_path = os.path.join(self.model_dir, "crf_checkpoints")
        create_dirs(self.ckpt_path, logger=self.logger)

        self.save_model()

    def save_model(self):
        path = os.path.join(
            self.model_dir, "{}_crf.config".format(self.experiment_name)
        )
        params = {
            "args": self.args,
            "output_size": self.output_size,
            "state_dict": self.model.state_dict(),
            "class_weight": self.normalizer.class_weight,
        }
        torch.save(params, path)
        self.logger.info("Model configurations saved in {}".format(path))

    def save_evaluation(self, step, epoch, batch, current_loss):
        """Save a training checkpoint and the avg loss and scores."""
        eval_path = os.path.join(
            self.eval_path, "{}_{}.json".format(self.experiment_name, str(step))
        )
        eval_params = {
            "step": step,
            "epoch": epoch,
            "batch": batch,
            "train_loss_avg": self.loss_avg.cumulate,
            "train_iou_avg": self.iou_avg.cumulate,
            "train_aupr_avg": self.aupr_avg.cumulate,
        }
        with open(eval_path, "w") as outfile:
            json.dump(eval_params, outfile, indent=0)

    def save_checkpoint(self, step):
        """Save model state_dict in a checkpoint .ckpt file."""
        ckpt_path = os.path.join(
            self.ckpt_path, "{}_crf_{}.ckpt".format(self.experiment_name, step)
        )
        ckpt_params = {
            "step": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt_params, ckpt_path)

    ############################
    # Load NetTIME model and checkpoint
    ############################
    def load_NetTIME_checkpoint(self):
        """Load a NetTIME model and checkpoint params."""
        # Load model
        if self.nettime_config is not None:
            path = self.nettime_config
        else:
            path = os.path.join(
                self.model_dir, "{}.config".format(self.experiment_name)
            )
        params = torch.load(path, map_location=self.device)

        # Initialize model
        self.nettime = NetTIME(params["args"])
        self.nettime.to(self.device)

        # Load model params
        self.ct_feature = params["args"].ct_feature
        self.tf_feature = params["args"].tf_feature
        self.output_size = params["args"].output_size

        # Load checkpoint params
        if self.nettime_ckpt is not None:
            if self.nettime_ckpt.endswith(".json"):
                ckpt = self.read_ckpt_json(self.nettime_ckpt)
            elif self.nettime_ckpt.endswith(".ckpt"):
                ckpt = self.nettime_ckpt
            else:
                raise RuntimeError(
                    "Invalid extension for NetTIME checkpoint file "
                    "{}".format(self.nettime_ckpt)
                )
        else:
            path = os.path.join(
                self.model_dir,
                "{}_{}".format("VALIDATION".lower(), "EVALUATE".lower()),
                "best_checkpoint",
                "best_aupr.json",
            )
            ckpt = self.read_ckpt_json(path)
        self.logger.info("Loading NetTIME checkpoint {}".format(ckpt))
        ckpt_params = torch.load(ckpt, map_location=self.device)
        self.nettime.load_state_dict(ckpt_params["state_dict"])

    def read_ckpt_json(self, filename):
        """Load a checkpoint .json file."""
        with open(filename, "r") as infile:
            data = json.load(infile)
        return data["ckpt_path"]
