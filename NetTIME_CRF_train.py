import argparse

from NetTIME import CRFTrainWorkflow

######## User Input ########
parser = argparse.ArgumentParser(
    "Train a linear chain CRF classifier on NetTIME predictions."
)

# Training parameters
parser.add_argument(
    "--batch_size",
    type=int,
    default=2700,
    help="Training batch size. Default: 2700",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=50,
    help="Number of training epoch. Default: 50",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=20,
    help="Number of workers used to perform multi-process data loading. "
    "Default: 20",
)
parser.add_argument(
    "--nettime_config",
    type=str,
    default=None,
    help="Specify an alternative path to NetTIME .config file.",
)
parser.add_argument(
    "--nettime_ckpt",
    type=str,
    default=None,
    help="Specify an alternative path to a best NetTIME checkpoint .ckpt file "
    "or a best NetTIME checkpoint evaluation .json file.",
)
parser.add_argument(
    "--start_from_checkpoint",
    type=str,
    default=None,
    help="Path to a pretrained model checkpoint from which to start training. "
    "Default: None, training starts from scratch.",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate, Default: 1e-4",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Oprimizer weight decay. Default: 0.0",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Random seed. Default: None"
)
parser.add_argument(
    "--loss_avg_ratio",
    type=float,
    default=0.9,
    help="Weight of loss history when calculating cumulative loss. Default: 0.9",
)

# Data
parser.add_argument(
    "--dataset",
    type=str,
    default="data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5",
    help="Path to NetTIME training data. Default: "
    "data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="TRAINING",
    help="Dataset type. Default: TRAINING.",
)
parser.add_argument(
    "--class_weight",
    type=str,
    default=None,
    help="Path to a numpy .npy file specifying the class weight. Default: "
    "None, class_weight will be calculated from target labels prior to training.",
)
parser.add_argument(
    "--output_key",
    type=str,
    default=["output_conserved"],
    nargs="+",
    help="A list of keys specifying the types of target labels to use for "
    "training. Default: output_conserved",
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
    help="List of group names to be excluded from training. "
    "Default: None, all conditions in DATASET are included during training.",
)
parser.add_argument(
    "--include_groups",
    type=str,
    default=None,
    nargs="+",
    help="List of group names to be included for training."
    "Default: None, all conditions in DATASET are included during training.",
)

# Display and save
parser.add_argument(
    "--print_every",
    type=int,
    default=10,
    help="Display training loss every PRINT_EVERY steps. Default: 10",
)
parser.add_argument(
    "--evaluate_every",
    type=int,
    default=50,
    help="Save a model checkpoint every (PRINT_EVERY * EVALUATE_EVERY) steps "
    "for evaluation. Default 50",
)
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
    help="Specify an alternative location to save model training loss files.",
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default=None,
    help="Specify an alternative location to save model checkpoint files.",
)

args = parser.parse_args()

######## Configure workflow ########
workflow = CRFTrainWorkflow()

# Training parameters
workflow.batch_size = args.batch_size
workflow.num_epochs = args.num_epochs
workflow.num_workers = args.num_workers
workflow.nettime_config = args.nettime_config
workflow.nettime_ckpt = args.nettime_ckpt
workflow.start_from_checkpoint = args.start_from_checkpoint

workflow.learning_rate = args.learning_rate
workflow.weight_decay = args.weight_decay
workflow.seed = args.seed
workflow.loss_avg_ratio = args.loss_avg_ratio

# Data
workflow.dataset = args.dataset
workflow.dtype = args.dtype
workflow.class_weight = args.class_weight
workflow.output_key = args.output_key
workflow.index_file = args.index_file
workflow.exclude_groups = args.exclude_groups
workflow.include_groups = args.include_groups


# Display and save
workflow.print_every = args.print_every
workflow.evaluate_every = args.evaluate_every
workflow.output_dir = args.output_dir
workflow.experiment_name = args.experiment_name
workflow.result_dir = args.result_dir
workflow.ckpt_dir = args.ckpt_dir

# Args
workflow.args = args

######## Model Run ########
workflow.run()
