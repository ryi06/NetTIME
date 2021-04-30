import argparse

from MultiBPE import TrainWorkflow


def frac2float(v):
    res = eval(v)
    assert isinstance(res, float)
    return res


######## User Input ########
parser = argparse.ArgumentParser("Training a MultiBPE model.")
parser.register("type", "frac2float", frac2float)

# Training parameters
parser.add_argument(
    "--batch_size",
    type=int,
    default=1800,
    help="Training batch size. Default: 1800",
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
    "--seed", type=int, default=1111, help="Random seed. Default: 1111"
)
parser.add_argument(
    "--loss_avg_ratio",
    type=float,
    default=0.9,
    help="Weight of loss history when calculating cumulative loss. Default: 0.9",
)
parser.add_argument(
    "--clip_threshold",
    type=float,
    default=None,
    help="The max_norm of the gradients to clip when using graduent clipping. "
    "Default: None, no gradient clipping.",
)

# Data
parser.add_argument(
    "--ct_feature",
    action="store_true",
    help="Include cell type-specific feature during training. Default: False.",
)
parser.add_argument(
    "--tf_feature",
    action="store_true",
    help="Include TF-specific feature during training. Default: False.",
)
parser.add_argument(
    "--output_key",
    type=str,
    default=["output_conserved", "output_relaxed"],
    nargs="+",
    help="A list of keys specifying the types of target labels to use for "
    "training. Default: output_conserved output_relaxed",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5",
    help="Path to training data. "
    "Default: data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="TRAINING",
    help="Dataset type. Default: TRAINING.",
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

# Model architecture
parser.add_argument(
    "--tf_vocab_size",
    type=int,
    default=256,
    help="TF vocabulary size. Default: 256",
)
parser.add_argument(
    "--ct_vocab_size",
    type=int,
    default=64,
    help="Cell type vocabulary size. Default: 64",
)
parser.add_argument(
    "--input_size",
    type=int,
    default=0,
    help="Sum of the number of cell type features and the number of TF features. "
    "Default: 0, no cell type and TF features included during training.",
)
parser.add_argument(
    "--output_size",
    type=int,
    default=2,
    help="Number of classes in target labels. Default 2",
)
parser.add_argument(
    "--seq_length",
    type=int,
    default=1000,
    help="Input sequence length, Default 1000",
)
parser.add_argument(
    "--embedding_size",
    type=int,
    default=50,
    help="Dimension of the embedding vectors for TFs and cell types. Default: 50",
)

parser.add_argument(
    "--fc_act_fn",
    type=str,
    default="ReLU",
    choices=["ReLU", "Tanh"],
    help="Name of the activation function for FC layers. Default: ReLU",
)
parser.add_argument(
    "--num_basic_blocks",
    type=int,
    default=2,
    help="Number of Basic Block layers. Default: 2",
)
parser.add_argument(
    "--cnn_act_fn",
    type=str,
    default="ReLU",
    choices=["ReLU", "Tanh"],
    help="Name of the activation function for CNN layers. Default: ReLU",
)
parser.add_argument(
    "--rnn_act_fn",
    type=str,
    default="Tanh",
    choices=["ReLU", "Tanh"],
    help="Activation function for RNN layers. Default: Tanh",
)
parser.add_argument(
    "--kernel_size",
    type=int,
    default=7,
    help="CNN kernal size. Default: 7",
)
parser.add_argument(
    "--stride",
    type=int,
    default=1,
    help="CNN stride size. Default 1.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout rate. Default 0.0",
)

args = parser.parse_args()

######## Configure workflow ########
workflow = TrainWorkflow()

# Training parameters
workflow.batch_size = args.batch_size
workflow.num_epochs = args.num_epochs
workflow.num_workers = args.num_workers
workflow.start_from_checkpoint = args.start_from_checkpoint

workflow.learning_rate = args.learning_rate
workflow.weight_decay = args.weight_decay
workflow.seed = args.seed
workflow.loss_avg_ratio = args.loss_avg_ratio
workflow.clip_threshold = args.clip_threshold

# Data
workflow.ct_feature = args.ct_feature
workflow.tf_feature = args.tf_feature
workflow.output_key = args.output_key
workflow.dataset = args.dataset
workflow.dtype = args.dtype
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
