import argparse

from NetTIME import PredictWorkflow

######## User Input ########
parser = argparse.ArgumentParser(
    "Make predictions using a NetTIME model checkpoint."
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
    help="Specify an alternative path to model .config file.",
)
parser.add_argument(
    "--best_ckpt",
    type=str,
    default=None,
    help="Specify an alternative path to a best model checkpoint .ckpt file or "
    "a best checkpoint evaluation .json file, which will be used to make "
    "predictions.",
)
parser.add_argument(
    "--eval_metric",
    type=str,
    default=None,
    help="If BEST_CKPT is a .ckpt file, specify the evaluation metric used to "
    "select BEST_CKPT as the best checkpoint.",
)

# Data
parser.add_argument(
    "--output_key",
    type=str,
    default=["output_conserved"],
    nargs="+",
    help="A list of keys specifying the types of target labels to use for "
    "prediction. Default: output_conserved",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="data/datasets/training_example/test_minOverlap200_maxUnion600_example.h5",
    help="Path to prediction data. Default: "
    "data/datasets/training_example/test_minOverlap200_maxUnion600_example.h5",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="TEST",
    help="Dataset type. Default: TEST.",
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
    help="When target labels specified by OUTPUT_KEY in DATASET are available, "
    "specify a list of group names to be excluded from prediction. Default: "
    "None, all conditions in DATASET OUTPUT_KEY are included during prediction.",
)
parser.add_argument(
    "--include_groups",
    type=str,
    default=None,
    nargs="+",
    help="When target labels specified by OUTPUT_KEY in DATASET are available, "
    "specify a list of group names to be included for prediction. Default: "
    "None, all conditions in DATASET OUTPUT_KEY are included during prediction.",
)
parser.add_argument(
    "--no_target",
    action="store_true",
    help="Specify this flag when target labels are not available.",
)
parser.add_argument(
    "--predict_groups",
    type=str,
    default=None,
    nargs="+",
    help="When target labels are not available, specify a list of group names "
    "for which to make predictions. Default: None, no condition is included.",
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

args = parser.parse_args()

######## Configure workflow ########
workflow = PredictWorkflow()

# Prediction parameters
workflow.batch_size = args.batch_size
workflow.num_workers = args.num_workers
workflow.seed = args.seed
workflow.model_config = args.model_config
workflow.best_ckpt = args.best_ckpt
workflow.eval_metric = args.eval_metric

# Data
workflow.output_key = args.output_key
workflow.dataset = args.dataset
workflow.dtype = args.dtype
workflow.index_file = args.index_file
workflow.exclude_groups = args.exclude_groups
workflow.include_groups = args.include_groups
workflow.no_target = args.no_target
workflow.predict_groups = args.predict_groups

# Save
workflow.output_dir = args.output_dir
workflow.experiment_name = args.experiment_name
workflow.result_dir = args.result_dir

# Args
workflow.args = args

######## Model Run ########
workflow.run()
