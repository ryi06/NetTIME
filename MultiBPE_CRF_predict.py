import argparse

from MultiBPE import CRFPredictWorkflow

######## User Input ########
parser = argparse.ArgumentParser(
    "Make predictions using MultiBPE linear chain CRF classifier."
)

# Prediction parameters
parser.add_argument(
    "--batch_size",
    type=int,
    default=2700,
    help="Prediction batch size. Default: 2700",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=10,
    help="Number of workers used to perform multi-process data loading. "
    "Default: 10",
)
parser.add_argument(
    "--seed", type=int, default=1111, help="Random seed. Default: 1111"
)
parser.add_argument(
    "--model_config",
    type=str,
    default=None,
    help="Specify an alternative path to CRF .config file.",
)
parser.add_argument(
    "--best_ckpt",
    type=str,
    default=None,
    help="Specify an alternative path to a best model checkpoint .ckpt file or "
    "a best checkpoint evaluation .json file, which will be used to make "
    "predictions.",
)

# Data
parser.add_argument(
    "--prediction_dir",
    type=str,
    default=None,
    help="Path to MultiBPE prediction directory.",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="TEST",
    help="Dataset type. Default: TEST.",
)
parser.add_argument(
    "--class_weight",
    type=str,
    default=None,
    help="Path to a numpy .npy file specifying the class weight. "
    "Default: None, use class weight generated from training data.",
)

# Save
parser.add_argument(
    "--output_dir",
    type=str,
    default="experiments/",
    help="Root directory for saving experiment results. Default: experiments/",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="training_example",
    help="experiment name.",
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=None,
    help="Specify an alternative location to save prediction files.",
)
parser.add_argument(
    "--tmp_dir",
    type=str,
    default="/tmp",
    help="Temporary directory for saving merged prediction .h5 file. Default: "
    "/tmp",
)

args = parser.parse_args()

######## Configure workflow ########
workflow = CRFPredictWorkflow()

# Prediction parameters
workflow.batch_size = args.batch_size
workflow.num_workers = args.num_workers
workflow.seed = args.seed
workflow.model_config = args.model_config
workflow.best_ckpt = args.best_ckpt

# Data
workflow.prediction_dir = args.prediction_dir
workflow.dtype = args.dtype
workflow.class_weight = args.class_weight

# Save
workflow.output_dir = args.output_dir
workflow.experiment_name = args.experiment_name
workflow.result_dir = args.result_dir
workflow.tmp_dir = args.tmp_dir

# Args
workflow.args = args

######## Model Run ########
workflow.run()
