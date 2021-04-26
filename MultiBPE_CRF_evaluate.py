import argparse

from MultiBPE import CRFEvaluateWorkflow

######## User Input ########
parser = argparse.ArgumentParser(
    "Evaluate a linear chain CRF classifier trained on MultiBPE predictions."
)

# Validation parameters
parser.add_argument(
    "--batch_size",
    type=int,
    default=2700,
    help="Evaluation batch size. Default: 2700",
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
    "--multibpe_config",
    type=str,
    default=None,
    help="Specify an alternative path to MultiBPE .config file.",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help="Specify an alternative location to find checkpoints to be evaluated.",
)

# Data
parser.add_argument(
    "--dataset",
    type=str,
    default="data/datasets/training_example/validation_minOverlap200_maxUnion600_example.h5",
    help="Path to MultiBPE evaluation data containing target labels. Default: "
    "data/datasets/training_example/validation_minOverlap200_maxUnion600_example.h5",
)
parser.add_argument(
    "--prediction_dir",
    type=str,
    default=None,
    help="Path to MultiBPE prediction directory.",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="VALIDATION",
    help="Dataset type. Default: VALIDATION.",
)
parser.add_argument(
    "--class_weight",
    type=str,
    default="data/datasets/training_example/validation_minOverlap200_maxUnion600_example_weight.npy",
    help="Path to a numpy .npy file specifying the class weight. Default: "
    "data/datasets/training_example/validation_minOverlap200_maxUnion600_example_weight.npy",
)

# Save
parser.add_argument(
    "--output_dir",
    type=str,
    default="experiments/",
    help="Root directory for saving experiment results."
    "Default: experiments/",
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
    help="Specify an alternative location to save checkpoint evaluation files.",
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
workflow = CRFEvaluateWorkflow()

# Validation parameters
workflow.batch_size = args.batch_size
workflow.num_workers = args.num_workers
workflow.seed = args.seed
workflow.multibpe_config = args.multibpe_config
workflow.ckpt_dir = args.ckpt_dir

# Data
workflow.dataset = args.dataset
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
