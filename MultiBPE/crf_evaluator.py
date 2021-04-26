import json
import multiprocessing as mp
import os

import torch
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


class CRFEvaluateWorkflow(object):
    def __init__(self):
        self.__MODE = "CRF_EVALUATE"
        self.__METRIC = "loss_avg"

        # Validation parameters
        self.batch_size = None
        self.num_workers = None
        self.seed = None
        self.multibpe_config = None
        self.ckpt_dir = None

        # Data
        self.dataset = None
        self.prediction_dir = None
        self.dtype = None
        self.class_weight = None

        # Save
        self.output_dir = None
        self.experiment_name = None
        self.result_dir = None
        self.tmp_dir = None

        ######## Configs ########
        self.args = None

    def run(self):
        # Set up logger and print out configurations
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

        # Initialize model.
        self.load_model()
        self.logger.info("MODEL ARCHITECTURE:\n{}".format(self.model))

        # Create evaluation result folders.
        self.initialize_result_directories()

        # Start evaluation.
        self.logger.info("Start evaluating CRF classifier.")
        try:
            self.crf_evaluate()
            self.logger.info("CRF classifier evaluation finished.")
        except KeyboardInterrupt:
            self.logger.warning("Evaluation interrupted. Program exit.")

    ########################
    # Multiprocessing : main thread
    ########################
    def crf_evaluate(self):
        """Evaluate all checkpoints for a trained CRF classifier."""
        ckpt_tracker = CheckpointTracker(os.path.join(self.ckpt_path, "*.ckpt"))
        self.time_tracker = TimeTracker()

        # Initialize validation dataset
        params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": CRF_collate_samples,
        }
        file_path, _ = merge_predictions(
            self.prediction_dir,
            self.logger,
            self.tmp_dir,
            target_path=self.dataset,
        )
        evaluate_dset = CRFDataset(file_path, self.class_weight)
        evaluate_iter = DataLoader(evaluate_dset, **params)
        num_batches = len(evaluate_iter)
        self.logger.info(
            "{:,} samples used for evaluation.".format(evaluate_dset.__len__())
        )

        # data preprocess worker
        preprocess_queue = mp.JoinableQueue(maxsize=128)
        preprocess_worker = mp.Process(
            name="preprocess",
            target=self.preprocess,
            args=(preprocess_queue, evaluate_iter),
        )
        preprocess_worker.start()
        self.logger.info("CRF evaluation data workder started")

        # Evaluate all checkpoints.
        while len(ckpt_tracker.remaining) > 0:
            for ckpt in ckpt_tracker.remaining:
                self.evaluate_checkpoint(ckpt, preprocess_queue, num_batches)
                ckpt_tracker.add_evaluated(ckpt)
            ckpt_tracker.reset_params()

        # Terminate data worker.
        preprocess_worker.terminate()

    def evaluate_checkpoint(self, checkpoint, preprocess_queue, num_batches):
        """Evaluate one checkpoint of a trained CRF classifier."""
        # Load checkpoint
        step = self.load_checkpoint(checkpoint)
        self.logger.info("Evaluating CRF step {}".format(step))

        # Return if the checkpoint is already evaluated.
        eval_path = os.path.join(
            self.eval_path, "{}_{}.json".format(self.experiment_name, step)
        )
        if os.path.exists(eval_path):
            self.logger.info("Step {} already evaluated".format(step))
            return

        # Initialize evaluation worker.
        evaluate_queue = mp.JoinableQueue(maxsize=64)
        evaluate_worker = mp.Process(
            name="evaluate_{}".format(step),
            target=self.evaluate,
            args=(checkpoint, evaluate_queue, eval_path, step, num_batches),
        )
        evaluate_worker.start()

        # Evaluate checkpoint.
        self.model.eval()
        with torch.no_grad():
            for b in tqdm(range(num_batches)):
                dset = preprocess_queue.get()
                feature, target = CRF_push_to_device(dset, self.device)
                loss = -self.model(feature, target, reduction="mean")
                evaluate_queue.put(loss.item())
        evaluate_queue.join()

    ########################
    # Multiprocessing : workers
    ########################
    # Preprocess worker
    def preprocess(self, queue, dataloader):
        """Set up multiprocessing data queue"""
        while True:
            for dset in dataloader:
                queue.put(dset)

    # Evaluate worker
    def evaluate(self, ckpt, queue, eval_path, step, num_batches):
        """Evaluate checkpoint and save evaluation to disk."""
        self.loss_avg = AverageMeter()
        for batch in range(num_batches):
            loss = queue.get()
            queue.task_done()
            self.loss_avg.update(loss)
        self.display_result(step)

        # Save evaluation
        result = {
            "step": step,
            "loss_avg": self.loss_avg.avg,
        }
        with open(eval_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

        # Update best checkpoint
        self.update_best_ckpt(result, ckpt)

    ############################
    # Display and save evaluations
    ############################
    def display_result(self, step):
        """Display average evaluation loss."""
        self.logger.info(
            "EVALUATE CRF | step {:8d} | avg loss {:8.4f} "
            "| time elapse: {:>12} |".format(
                step, self.loss_avg.avg, self.time_tracker.elapse()
            )
        )

    def update_best_ckpt(self, result, checkpoint):
        """Update best checkpoint metrics."""
        result["ckpt_path"] = checkpoint
        result["metric"] = self.__METRIC
        path = os.path.join(
            self.best_ckpt_path, "best_{}.json".format(self.__METRIC)
        )

        if os.path.exists(path):
            with open(path, "r") as infile:
                metrics = json.load(infile)
            if metrics[self.__METRIC] <= result[self.__METRIC]:
                return
        with open(path, "w") as outfile:
            json.dump(result, outfile, indent=4)

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
        self.best_ckpt_path = os.path.join(self.eval_path, "best_checkpoint")
        create_dirs(self.best_ckpt_path, logger=self.logger)

        if self.ckpt_dir is not None:
            self.ckpt_path = self.ckpt_dir
        else:
            self.ckpt_path = os.path.join(self.model_dir, "crf_checkpoint")

    ############################
    # Loading model and checkpoint
    ############################
    def load_model(self):
        """Load args from .config file and initialize CRF classifier."""
        # Load model params
        if self.multibpe_config is not None:
            path = self.multibpe_config
        else:
            path = os.path.join(
                self.model_dir, "{}.config".format(self.experiment_name)
            )
        params = torch.load(path, map_location="cpu")
        output_size = params["args"].output_size

        # Initialize classifier
        self.model = CRF(output_size, batch_first=True)
        self.model.to(self.device)

    def load_checkpoint(self, ckpt_path):
        """Load CRF checkpoint state_dict."""
        ckpt_params = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt_params["state_dict"])
        return ckpt_params["step"]
