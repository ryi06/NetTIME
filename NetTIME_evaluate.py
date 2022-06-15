import argparse

from NetTIME import EvaluateWorkflow

######## User Input ########
parser = argparse.ArgumentParser("Evaluating a trained NetTIME model.")

# Evaluation parameters
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
    "--model_config",
    type=str,
    default=None,
    help="Specify an alternative path to model .config file.",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help="Specify an alternative location to find checkpoints to be evaluated.",
)

# Data
parser.add_argument(
    "--output_key",
    type=str,
    default=["output_conserved"],
    nargs="+",
    help="A list of keys specifying the types of target labels to use for "
    "evaluation. Default: output_conserved",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="data/datasets/training_example/validation_minOverlap200_maxUnion600.h5",
    help="Path to evaluation data. Default: "
    "data/datasets/training_example/validation_minOverlap200_maxUnion600.h5",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="VALIDATION",
    help="Dataset type. Default: VALIDATION.",
)
parser.add_argument(
    "--index_file",
    type=str,
    default="data/embeddings/example.pkl",
    help="Path to a pickle file containing indices for TF and cell type labels."
    "Default: data/embeddings/example.pkl",
)
parser.add_argument(
    "--exclude_groups",
    type=str,
    default=None,
    nargs="+",
    help="List of group names to be excluded from evaluation. "
    "Default: None, all conditions in DATASET are included during evaluation.",
)
parser.add_argument(
    "--include_groups",
    type=str,
    default=None,
    nargs="+",
    help="List of group names to be included for evaluation."
    "Default: None, all conditions in DATASET are included during evaluation.",
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

args = parser.parse_args()

######## Configure workflow ########
workflow = EvaluateWorkflow()

# Evaluation parameters
workflow.batch_size = args.batch_size
workflow.num_workers = args.num_workers
workflow.seed = args.seed
workflow.model_config = args.model_config
workflow.ckpt_dir = args.ckpt_dir

# Data
workflow.output_key = args.output_key
workflow.dataset = args.dataset
workflow.dtype = args.dtype
workflow.index_file = args.index_file
workflow.exclude_groups = args.exclude_groups
workflow.include_groups = args.include_groups

# Save
workflow.output_dir = args.output_dir
workflow.experiment_name = args.experiment_name
workflow.result_dir = args.result_dir

# Args
workflow.args = args

######## Model Run ########
workflow.run()
