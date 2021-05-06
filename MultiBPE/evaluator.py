import json
import multiprocessing as mp
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calculate_metrics import ScoreTracker, avg_sample_loss
from .dataset import MultiBPEDataset, collate_samples, push_to_device
from .model import MultiBPE
from .utils import *


class EvaluateWorkflow(object):
    """Define the MultiBPE evaluation workflow."""

    def __init__(self):
        self.__MODE = "EVALUATE"

        # Evaluation parameters
        self.batch_size = None
        self.num_workers = None
        self.seed = None
        self.model_config = None
        self.ckpt_dir = None

        # Data
        self.output_key = None
        self.dataset = None
        self.dtype = None
        self.index_file = None
        self.exclude_groups = None
        self.include_groups = None

        # Save
        self.output_dir = None
        self.experiment_name = None
        self.result_dir = None

        # Args
        self.args = None

    def run(self):
        """Run evaluation workflow."""
        # Initialize model dir and display configurations.
        self.model_dir = init_model_dir(self.output_dir, self.experiment_name)
        self.logger = set_logger(
            self.model_dir,
            self.experiment_name,
            self.__MODE,
            self.dtype,
            overwrite=False,
        )
        display_args(self.args, self.logger)

        # Set up GPUs
        if not torch.cuda.is_available():
            raise Exception("No GPU found.")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.device = torch.device("cuda")
        self.logger.info(
            "Use {} GPU(s) for evaluation".format(torch.cuda.device_count())
        )

        # Initialize model.
        self.load_model()
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.logger.info("MODEL ARCHITECTURE:\n{}".format(self.model))

        # Create evaluation result folders.
        self.initialize_result_directories()

        # Start evaluation.
        self.logger.info("Start evaluation.")
        try:
            self.evaluate_checkpoints()
            self.logger.info("Evaluation finished.")
        except KeyboardInterrupt:
            self.logger.warning("Evaluation interrupted. Program exit.")

    ########################
    # Multiprocessing: main thread
    ########################
    def evaluate_checkpoints(self):
        """Evaluate all checkpoints for a trained model."""
        ckpt_tracker = CheckpointTracker(os.path.join(self.ckpt_path, "*.ckpt"))
        self.time_tracker = TimeTracker()

        # Load data
        embed_indices = reverse_embedding(self.index_file)
        params = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "collate_fn": collate_samples,
        }
        evaluate_dset = MultiBPEDataset(
            self.dataset,
            embed_indices,
            self.output_key,
            ct_feature=self.ct_feature,
            tf_feature=self.tf_feature,
            exclude_groups=self.exclude_groups,
            include_groups=self.include_groups,
        )
        evaluate_iter = DataLoader(evaluate_dset, **params)
        num_batches = len(evaluate_iter)
        self.logger.info(
            "{:,} samples used for evaluation.".format(evaluate_iter.__len__())
        )

        # Initialize data loading worker.
        preprocess_queue = mp.JoinableQueue(maxsize=32)
        preprocess_worker = mp.Process(
            name="preprocess",
            target=self.preprocess,
            args=(preprocess_queue, evaluate_iter),
        )
        preprocess_worker.start()
        self.logger.info("Evaluation data workder started")

        # Evaluate all checkpoints.
        while len(ckpt_tracker.remaining) > 0:
            for ckpt in ckpt_tracker.remaining:
                self.evaluate_checkpoint(ckpt, preprocess_queue, num_batches)
                ckpt_tracker.add_evaluated(ckpt)
            ckpt_tracker.reset_params()

        # Terminate data worker.
        preprocess_worker.terminate()

    def evaluate_checkpoint(self, checkpoint, preprocess_queue, num_batches):
        """Evaluate one checkpoint of a trained model."""
        # Load checkpoint.
        step = self.load_checkpoint(checkpoint)
        self.logger.info("Evaluating step {}".format(step))

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
                feature, target = push_to_device(dset, self.device)
                pred = self.model(feature)
                loss = avg_sample_loss(
                    self.criterion(pred.transpose(1, 2), target)
                )
                evaluate_queue.put((pred.cpu(), target.cpu(), loss.item()))
        evaluate_queue.join()

    ########################
    # Multiprocessing: workers
    ########################
    # Preprocess worker
    def preprocess(self, queue, dataloader):
        """Retrieve data and populate data queue."""
        while True:
            for dset in dataloader:
                queue.put(dset)

    # Evaluate worker
    def evaluate(self, ckpt, queue, eval_path, step, num_batches):
        """Evaluate checkpoint and save evaluation to disk."""
        self.initiate_avg_trackers()
        score_tracker = ScoreTracker(self.__MODE)

        for batch in range(num_batches):
            pred, target, loss = queue.get()
            queue.task_done()

            # Update loss and scores
            self.loss_avg.update(loss)
            scores = score_tracker.calculate_scores(pred, target)
            if scores is not None:
                iou, aupr = scores
                self.iou_avg.update(iou)
                self.aupr_avg.update(aupr)

        self.display_result(step)

        result = {
            "step": step,
            "loss_avg": self.loss_avg.avg,
            "iou": self.iou_avg.avg,
            "aupr": self.aupr_avg.avg,
        }

        # Save evaluation
        with open(eval_path, "w") as outfile:
            json.dump(result, outfile, indent=4)

        self.update_best_ckpt(result, ckpt, "aupr")
        self.update_best_ckpt(result, ckpt, "loss_avg")

    ########################
    # Track, display and save evaluations
    ########################
    def initiate_avg_trackers(self):
        """Initialize average loss and score trackers."""
        # Losses
        self.loss_avg = AverageMeter()

        # Scores
        self.iou_avg = AverageMeter()
        self.aupr_avg = AverageMeter()

    def display_result(self, step):
        """Display average evaluation loss and scores."""
        self.logger.info(
            "{} {} | step {:8d} | avg loss {:8.4f} | iou {:4.3f} "
            "| aupr {:4.3f} | time elapse: {:>12} |".format(
                self.__MODE,
                self.dtype,
                step,
                self.loss_avg.avg,
                self.iou_avg.avg,
                self.aupr_avg.avg,
                self.time_tracker.elapse(),
            )
        )

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
            self.ckpt_path = os.path.join(self.model_dir, "checkpoints")

    def update_best_ckpt(self, result, checkpoint, metric):
        """Update best checkpoint metrics."""
        result["ckpt_path"] = checkpoint
        result["metric"] = metric
        path = os.path.join(self.best_ckpt_path, "best_{}.json".format(metric))

        if os.path.exists(path):
            with open(path, "r") as infile:
                metrics = json.load(infile)
            if metric in ["loss_avg"] and metrics[metric] <= result[metric]:
                return
            if metric in ["aupr"] and metrics[metric] >= result[metric]:
                return
        with open(path, "w") as outfile:
            json.dump(result, outfile, indent=4)

    ############################
    # Loading model and checkpoint
    ############################
    def load_model(self):
        """Load args from .config file and initialize model."""
        # Load args
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

    def load_checkpoint(self, ckpt_path):
        """Load model checkpoint state_dict."""
        ckpt_params = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt_params["state_dict"])
        return ckpt_params["step"]
