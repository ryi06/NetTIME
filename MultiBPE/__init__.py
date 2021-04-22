"""
MultiBPE: A multitask learning model for prediction base-pair resolution
in vivo TF binding predictions using embeddings.
"""
from .trainer import TrainWorkflow
from .evaluator import EvaluateWorkflow
from .predictor import PredictWorkflow

# from .threshold_evaluator import EvaluateThresholdWorkflow
# from .threshold_predictor import PredictThresholdWorkflow

# from .crf_trainer import TrainCRFWorkflow
# from .crf_evaluator import EvaluateCRFWorkflow
# from .crf_predictor import PredictCRFWorkflow

__version__ = "0.1.0"
