import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import FilenameIterator, display_args, print_time

parser = argparse.ArgumentParser("Set paths for feature and target data files.")

parser.add_argument("input_file", type=str, help="Path to input pickle file.")
parser.add_argument("output_file", type=str, help="Path to output pickle file.")
parser.add_argument("output_folder", type=str, help="Path to output folder.")
parser.add_argument("metadata_file", type=str, help="ChIP-seq metadata file.")
parser.add_argument(
    "embedding_file",
    type=str,
    help="Path to pickle file storing embedding indices.",
)
parser.add_argument(
    "--ct_feature",
    type=str,
    default=None,
    help="A set of keys for cell type features, specified as 'ct1,ct2'. "
    "Default None, no cell type feature included.",
)
parser.add_argument(
    "--tf_feature",
    type=str,
    default=None,
    help="A set of keys for TF features, specified by 'tf1,tf2'. "
    "Default None, no TF feature included",
)
parser.add_argument(
    "--skip_target",
    action="store_true",
    help="Whether to skip setting paths for target labels. Default False.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=10000,
    help="The number of samples per data file. Default 10000.",
)

args = parser.parse_args()
display_args(args, __file__)


############## MAIN ##############
metadata = pd.read_csv(args.metadata_file, sep="\t")
TFs = np.unique(metadata["TF"])
CTs = np.unique(metadata["cell_type"])

start_time = time.time()
embedding = pickle.load(open(args.embedding_file, "rb"))
for tf in TFs:
    assert tf in embedding["tf"].keys()
for ct in CTs:
    assert ct in embedding["ct"].keys()
output_dict = pickle.load(open(args.input_file, "rb"))

# Create input feature path
input_root = os.path.join(args.output_folder, "feature")
os.makedirs(input_root, exist_ok=True)

print_time("Processing cell type features", start_time)
CT_features = args.ct_feature.split("+") if args.ct_feature is not None else []
for feature in CT_features:
    for ct in CTs:
        input_name = "{}.{}".format(feature, ct)
        input_prefix = os.path.join(input_root, input_name)
        coordinator = FilenameIterator(input_prefix, args.batch_size)
        for i in output_dict["input"].keys():
            if not feature in output_dict["input"][i].keys():
                output_dict["input"][i][feature] = {}
            output_dict["input"][i][feature][ct] = coordinator.get_next_entry()

print_time("Processing TF features", start_time)
TF_features = args.tf_feature.split("+") if args.tf_feature is not None else []
for feature in TF_features:
    for tf in TFs:
        input_name = "{}.{}".format(feature, tf)
        input_prefix = os.path.join(input_root, input_name)
        coordinator = FilenameIterator(input_prefix, args.batch_size)
        for i in output_dict["input"].keys():
            if not feature in output_dict["input"][i].keys():
                output_dict["input"][i][feature] = {}
            output_dict["input"][i][feature][tf] = coordinator.get_next_entry()

if not args.skip_target:
    print_time("Processing target", start_time)

    # Create output target path
    output_root = os.path.join(args.output_folder, "target")
    os.makedirs(output_root, exist_ok=True)

    output_dict["output_conserved"] = {}
    output_dict["output_relaxed"] = {}
    num_samples = len(output_dict["input"])
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        pair = "{}.{}".format(row["TF"], row["cell_type"])

        # Conserved
        output_prefix = os.path.join(output_root, "{}.conserved".format(pair))
        coordinator = FilenameIterator(output_prefix, args.batch_size)
        output_dict["output_conserved"][pair] = coordinator.get_batch_entries(
            num_samples,
            embedding["tf"][row["TF"]],
            embedding["ct"][row["cell_type"]],
        )

        # Relaxed
        output_prefix = os.path.join(output_root, "{}.relaxed".format(pair))
        coordinator = FilenameIterator(output_prefix, args.batch_size)
        output_dict["output_relaxed"][pair] = coordinator.get_batch_entries(
            num_samples,
            embedding["tf"][row["TF"]],
            embedding["ct"][row["cell_type"]],
        )

# Save path dictionary to output pickle file.
pickle.dump(output_dict, open(args.output_file, "wb"), protocol=4)
print_time("Path pickle dumped in {}".format(args.output_file), start_time)
