import argparse
import pickle
import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import FeatureReader, display_args, print_time

parser = argparse.ArgumentParser(
    "Get signal for input targets, including indices."
)

parser.add_argument("metadata_pkl", type=str, help="Path to example .pkl file.")
parser.add_argument("output_h5", type=str, help="Path to output h5 file.")
parser.add_argument(
    "--normalization",
    type=str,
    default="zscore",
    choices=["original", "zscore"],
    help="Key of the dataset to use in the data file. Default zscore.",
)
parser.add_argument(
    "--motif_threshold",
    type=float,
    default=1e-2,
    help="Threshold for motif enrichment feature. Default 1e-2.",
)
parser.add_argument(
    "--ct_feature",
    type=str,
    default=[],
    nargs="+",
    help="List of cell type features to include in dataset."
    "Default empty, no cell type feature signal track included.",
)
parser.add_argument(
    "--tf_feature",
    type=str,
    default=[],
    nargs="+",
    help="List of TF features to include in dataset."
    "Default empty, no tf feature signal track included.",
)
parser.add_argument(
    "--exclude_groups",
    type=str,
    default=[],
    nargs="+",
    help="List of group names to be excluded from training. "
    "Default empty, no group is excluded.",
)
parser.add_argument(
    "--compression",
    action="store_true",
    help="Whether to compress datasets in output hdf5 file. Default False.",
)
parser.add_argument(
    "--skip_target",
    action="store_true",
    help="Whether to skip target labels in output hdf5 file. Default False.",
)
parser.add_argument(
    "--condition_metadata",
    type=str,
    default=None,
    help="Path to condition metadata file. Specify when --skip_target is true.",
)
parser.add_argument(
    "--sequence_length", type=int, default=1000, help="Sample sequence length."
)

args = parser.parse_args()
display_args(args, __file__)


############## FUNCTION ##############
def return_indices(dset):
    return dset["start"], dset["stop"]


def add_target(metadata, h5, key):
    assert key in metadata.keys()
    h5_group = h5.create_group(key)
    print_time("Processing target {}".format(key), START_TIME)
    for group_name in tqdm(metadata[key].keys()):
        if group_name in args.exclude_groups:
            continue
        # Initialize dataset
        dataset = h5_group.create_dataset(
            group_name,
            (NUM_SAMPLES, args.sequence_length + 3),
            chunks=(1, args.sequence_length + 3),
            fillvalue=np.nan,
            **DATASET_PARAMS
        )
        # Fill values
        for file in metadata[key][group_name].keys():
            s, t = return_indices(metadata[key][group_name][file])
            batch = np.load(file)
            assert np.isnan(dataset[s:t]).all()
            assert batch["group_name"] == group_name
            assert batch["peak_type"] == key
            assert batch["start"] == s
            assert batch["stop"] == t
            dataset[s:t] = batch["data"]


def add_features(metadata, h5, key="input"):
    h5_group = h5.create_group(key)
    ct_reader, tf_reader = initialize_reader()
    for k in tqdm(metadata[key].keys()):
        sample_group = h5_group.create_group(str(k))
        sample_metadata = metadata[key][k]
        add_attribute(sample_metadata, sample_group)
        add_seq_feature(sample_metadata, sample_group)
        add_ct_feature(sample_metadata, sample_group, ct_reader)
        add_tf_feature(sample_metadata, sample_group, tf_reader)


def initialize_reader():
    ct_reader = {}
    for ct in CTs:
        ct_reader[ct] = {}
        for feat in args.ct_feature:
            ct_reader[ct][feat] = FeatureReader(args.normalization)

    tf_reader = {}
    for tf in TFs:
        tf_reader[tf] = {}
        for feat in args.tf_feature:
            tf_reader[tf][feat] = FeatureReader(
                args.normalization, threshold=args.motif_threshold
            )
    return ct_reader, tf_reader


def add_attribute(metadata, group):
    for attr in ATTRIBUTES:
        value = metadata[attr]
        group.attrs.create(
            attr, np.string_(value) if isinstance(value, str) else value
        )


def add_seq_feature(metadata, group):
    ids = metadata["fasta"]
    __write_feature(ids, group, "ids")
    one_hot = __one_hot_encoder(ids, (args.sequence_length, 4))
    __write_feature(one_hot, group, "fasta")


def add_ct_feature(metadata, group, reader):
    if len(args.ct_feature) == 0:
        return
    feature_group = group.create_group("ct_feature")
    for ct in CTs:
        arrays = []
        for feat in args.ct_feature:
            ary = reader[ct][feat].read_feature(metadata[feat][ct])
            arrays.append(ary)
        __write_feature(np.stack(arrays).squeeze(), feature_group, ct)


def add_tf_feature(metadata, group, reader):
    if len(args.tf_feature) == 0:
        return
    feature_group = group.create_group("tf_feature")
    for tf in TFs:
        arrays = []
        for feat in args.tf_feature:
            ary = reader[tf][feat].read_feature(metadata[feat][tf])
            arrays.append(ary)
        __write_feature(np.stack(arrays).squeeze(), feature_group, tf)


def __one_hot_encoder(index, shape):
    one_hot = np.zeros(shape)
    one_hot[np.arange(shape[0]), index] = 1
    return one_hot


def __write_feature(data, group, dset_name):
    if np.isnan(data).any():
        raise ValueError(
            "Group {} dataset {} has NaN values.".format(group.name, dset_name)
        )
    if data.shape[0] != args.sequence_length:
        second_dim = int(np.prod(data.shape) / args.sequence_length)
        data = np.reshape(data, (args.sequence_length, second_dim))
    group.create_dataset(dset_name, data.shape, data=data, **DATASET_PARAMS)


############## MAIN ##############
args.ct_feature.sort()
args.tf_feature.sort()
seq_dict = pickle.load(open(args.metadata_pkl, "rb"))

# Specitying global vars
NUM_SAMPLES = len(seq_dict["input"])
ATTRIBUTES = ["chrom", "start", "stop"]
START_TIME = time.time()
DATASET_PARAMS = {"compression": "gzip"} if args.compression else {}

# Get a list of included TFs and CTs
if not args.skip_target:
    group_names = seq_dict["output_conserved"].keys()
    group_names = [x for x in group_names if x not in args.exclude_groups]
    TFs = set([x.split(".")[0] for x in group_names])
    CTs = set([x.split(".")[1] for x in group_names])
else:
    conditions = pd.read_csv(args.condition_metadata, sep="\t")
    TFs = set(conditions["TF"])
    CTs = set(conditions["cell_type"])

# Create output hdf5 file
with h5py.File(args.output_h5, "w") as h5_file:
    # Add features
    print_time("Combining feature", START_TIME)
    add_features(seq_dict, h5_file)

    # Add target
    if not args.skip_target:
        print_time("Combining target", START_TIME)
        add_target(seq_dict, h5_file, "output_conserved")
        add_target(seq_dict, h5_file, "output_relaxed")

print_time("All sample processed!", START_TIME)
