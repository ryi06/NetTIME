import warnings

import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning


class ScoreTracker(object):
    """Calculate model prediction AUPR and IOU scores."""

    def __init__(self, mode, num_batches=None, append_all=False):
        self.__LAST_DIM = 2
        self.__SOFTMAX = nn.Softmax(dim=self.__LAST_DIM)
        self.__VALID_MODES = {
            "TRAIN",
            "EVALUATE",
            "THRESHOLD_EVALUATE",
            "THRESHOLD_PREDICT",
            "CRF_TRAIN",
            "CRF_PREDICT",
        }
        if mode not in self.__VALID_MODES:
            raise ValueError("Invalid mode {} specified.".format(mode))
        if append_all and num_batches is None:
            raise ValueError(
                "num_batches needs to be specified when " "append_all is True."
            )

        self.mode = mode
        self.num_batches = num_batches
        self.append_all = append_all
        self.reset_params()

    def reset_params(self):
        """Reset prediction and target data used to calculate scores."""
        self.pred = []
        self.target = []

    def calculate_scores(self, pred, target, batch=None):
        """Calculate scores."""
        if self.append_all:
            scores = self.calculate_all(pred, target, batch)
        else:
            scores = self.calculate_current(pred, target)
        return scores

    def calculate_current(self, pred, target):
        """
        Calculate scores for the current batch; used for training and
        evaluation mode.
        """
        self.pred.append(pred)
        self.target.append(target)

        if (not target.numpy().any()) or target.numpy().all():
            print("Target all zero, do nothing.")
            return
        return self._calculate_scores()

    def calculate_all(self, pred, target, batch):
        """
        Calaulte scores after num_batches of predictions are collected; used
        for prediction mode.
        """
        self.pred.append(pred)
        self.target.append(target)

        if batch + 1 < self.num_batches:
            return

        return self._calculate_scores()

    def calculate_iou(self, y, x):
        """
        Calculate IOU score.
        """
        if not x.any() and not y.any():
            return 1
        elif not x.any() or not y.any():
            return 0

        summation = x + y

        union = (summation != 0).sum()
        intersection = (summation == 2).sum()

        return intersection / union

    def _calculate_scores(self):
        """Calculate IOU and AUPR scores."""
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        pred = torch.cat(self.pred)
        target = torch.cat(self.target)

        T = target.numpy().flatten()
        if pred.dim() != target.dim():
            softmax = self.__SOFTMAX(pred)
            C = softmax.argmax(dim=self.__LAST_DIM).numpy().flatten()
            P = softmax[..., 1].numpy().flatten()
        else:
            C = P = pred.numpy().flatten()

        # IOU
        iou = self.calculate_iou(T, C)

        # AUPR
        precision, recall, _ = metrics.precision_recall_curve(T, P)
        aupr = metrics.auc(recall, precision)

        self.reset_params()

        return (iou, aupr)

def avg_sample_loss(batch_loss):
	return batch_loss.sum(1).mean()
