import argparse
import pickle
import time
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

import utils


############## FUNCTION ##############
def return_indices(dset):
    return dset["start"], dset["stop"]


def add_target(
    metadata,
    h5,
    key,
    group_names,
    exclude_groups,
    dataset_params,
    num_samples,
    bucket_size,
    sequence_length,
    start_time,
):
    assert key in metadata.keys()
    h5_group = h5.create_group(key)
    utils.print_time("Processing target {}".format(key), start_time)
    for group_name in tqdm(group_names):
        # Initialize dataset
        num_buckets = math.ceil(num_samples / bucket_size)
        for bucket in range(num_buckets):
            _bucket_size = min(bucket_size, num_samples - bucket * bucket_size)
            print("Bucket: {} | Bucket size: {}".format(bucket, _bucket_size))
            h5_group.create_dataset(
                "{}/{}".format(group_name, bucket),
                (_bucket_size, sequence_length + 3),
                chunks=(1, sequence_length + 3),
                fillvalue=np.nan,
                **dataset_params
            )
        # Fill values.
        for file in metadata[key][group_name].keys():
            s, t = return_indices(metadata[key][group_name][file])
            batch = np.load(file)
            batch_data = batch["data"]
            bs = get_bucket_id(s, bucket_size)
            dataset = h5_group["{}/{}".format(group_name, bs)]

            ss = int(s % bucket_size)
            tt = ss + len(batch_data)

            assert np.isnan(dataset[ss:tt]).all()
            assert not np.isnan(batch_data).any()
            assert batch["group_name"] == group_name
            assert batch["peak_type"] == key
            assert batch["start"] == s
            assert batch["stop"] == t
            dataset[ss:tt] = batch_data


def add_features(
    metadata,
    h5,
    ct_feature,
    tf_feature,
    TFs,
    CTs,
    bucket_size,
    normalization,
    motif_threshold,
    sequence_length,
    dataset_params,
    num_samples,
    start_time,
    key="input",
    start_id=None,
    stop_id=None,
):
    h5_group = h5[key]
    ct_reader, tf_reader = initialize_reader(
        ct_feature, tf_feature, TFs, CTs, normalization, motif_threshold
    )

    start_key = 0 if start_id is None else max(0, start_id)
    if (stop_id is None) or (stop_id >= num_samples):
        stop_key = num_samples
    else:
        stop_key = stop_id

    utils.print_time(
        "Processing examples [{}, {}).".format(start_key, stop_key), start_time
    )
    for k in tqdm(range(start_key, stop_key)):
        bucket = get_bucket_id(k, bucket_size)
        sample_group = h5_group[bucket][str(k)]
        sample_metadata = metadata[key][k]
        add_ct_feature(
            sample_metadata,
            sample_group,
            ct_reader,
            ct_feature,
            CTs,
            sequence_length,
            dataset_params,
        )
        add_tf_feature(
            sample_metadata,
            sample_group,
            tf_reader,
            tf_feature,
            TFs,
            sequence_length,
            dataset_params,
        )


def initialize_reader(
    ct_feature, tf_feature, TFs, CTs, normalization, motif_threshold
):
    ct_reader = {}
    for ct in CTs:
        ct_reader[ct] = {}
        for feat in ct_feature:
            ct_reader[ct][feat] = utils.FeatureReader(normalization)

    tf_reader = {}
    for tf in TFs:
        tf_reader[tf] = {}
        for feat in tf_feature:
            tf_reader[tf][feat] = utils.FeatureReader(
                normalization, threshold=motif_threshold
            )
    return ct_reader, tf_reader


def add_ct_feature(
    metadata, group, reader, ct_feature, CTs, sequence_length, dataset_params
):
    if len(ct_feature) == 0:
        return
    feature_group = group.create_group("ct_feature")
    for ct in CTs:
        arrays = []
        for feat in ct_feature:
            ary = reader[ct][feat].read_feature(metadata[feat][ct])
            arrays.append(ary)
        __write_feature(
            np.stack(arrays).squeeze(),
            feature_group,
            ct,
            sequence_length,
            dataset_params,
        )


def add_tf_feature(
    metadata, group, reader, tf_feature, TFs, sequence_length, dataset_params
):
    if len(tf_feature) == 0:
        return
    feature_group = group.create_group("tf_feature")
    for tf in TFs:
        arrays = []
        for feat in tf_feature:
            ary = reader[tf][feat].read_feature(metadata[feat][tf])
            arrays.append(ary)
        __write_feature(
            np.stack(arrays).squeeze(),
            feature_group,
            tf,
            sequence_length,
            dataset_params,
        )


def __write_feature(data, group, dset_name, sequence_length, dataset_params):
    if np.isnan(data).any():
        raise ValueError(
            "Group {} dataset {} has NaN values.".format(group.name, dset_name)
        )
    if data.shape[0] != sequence_length:
        data = np.transpose(data)
        # second_dim = int(np.prod(data.shape) / sequence_length)
        # data = np.reshape(data, (sequence_length, second_dim))
    group.create_dataset(dset_name, data.shape, data=data, **dataset_params)


def _get_conditions(seq_dict, skip_target, condition_metadata, exclude_groups):
    if (skip_target and 
        (condition_metadata is None or not os.path.isfile(condition_metadata))):
        raise ValueError(
            "`condition_metadata` file must exist when `skip_target is True.`"
        )

    if condition_metadata is None or not os.path.isfile(condition_metadata):
        # When `condition_metadata` file path is not supplied. Get the list of
        # TF and cell types from the metadata dictionary.
        group_names = seq_dict["output_conserved"].keys()
    else:
        # Get TF and cell types from `condition_metadata` file.
        conditions = pd.read_csv(condition_metadata, sep="\t")
        tfs = conditions["TF"]
        cts = conditions["cell_type"]
        group_names = ['.'.join([x, y]) for x, y in zip(tfs, cts)]

    group_names = [x for x in group_names if x not in exclude_groups]
    TFs = set([x.split(".")[0] for x in group_names])
    CTs = set([x.split(".")[1] for x in group_names])
    
    print(f'TFs: {TFs} | CTs: {CTs} | group_names: {group_names}')

    return group_names, TFs, CTs


def get_bucket_id(sample_id, bucket_size):
    return str(int(sample_id // bucket_size))


############## MAIN ##############
def main(
    seq_dict,
    ct_feature,
    tf_feature,
    output_h5,
    output_types,
    compression,
    skip_feature,
    skip_target,
    condition_metadata,
    exclude_groups,
    sequence_length,
    normalization,
    motif_threshold,
    example_feature_start_id,
    example_feature_stop_id,
    bucket_size=utils.BUCKET_SIZE,
):
    num_samples = len(seq_dict["input"])
    start_time = time.time()
    dataset_params = {"compression": "gzip"} if compression else {}
    group_names, TFs, CTs = _get_conditions(
        seq_dict, skip_target, condition_metadata, exclude_groups
    )

    ct_feature.sort()
    tf_feature.sort()

    # Create output hdf5 file
    with h5py.File(output_h5, "a") as h5_file:
        # # Add features
        if not skip_feature:
            utils.print_time("Combining feature", start_time)
            add_features(
                seq_dict,
                h5_file,
                ct_feature,
                tf_feature,
                TFs,
                CTs,
                bucket_size,
                normalization,
                motif_threshold,
                sequence_length,
                dataset_params,
                num_samples,
                start_time,
                start_id=example_feature_start_id,
                stop_id=example_feature_stop_id,
            )

        # Add target
        if not skip_target:
            utils.print_time("Combining target", start_time)
            for output_type in output_types:
                add_target(
                    seq_dict,
                    h5_file,
                    output_type,
                    group_names,
                    exclude_groups,
                    dataset_params,
                    num_samples,
                    bucket_size,
                    sequence_length,
                    start_time,
                )

    utils.print_time("All samples processed!", start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Merge features signals (and targets labels) into an HDF5 file."
    )

    parser.add_argument(
        "metadata_pkl", type=str, help="Path to example .pkl file."
    )
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
        "--skip_feature",
        action="store_true",
        help="Whether to skip input features in output hdf5 file. Default False",
    )
    parser.add_argument(
        "--example_feature_start_id",
        type=int,
        default=None,
        help="example id to start processing input feature. Specify this param "
        "if you wish to process input feature parallelly and process a subset "
        "of examples per job.",
    )
    parser.add_argument(
        "--example_feature_stop_id",
        type=int,
        default=None,
        help="example id to stop processing input feature. Specify this param "
        "if you wish to process input feature parallelly and process a subset "
        "of examples per job.",
    )
    parser.add_argument(
        "--output_types",
        type=str,
        default=["output_conserved", "output_relaxed"],
        nargs="+",
        help="List of output types to include in the output hdf5 file. Default "
        "'output_conserved' 'output_relaxed'.",
    )
    parser.add_argument(
        "--condition_metadata",
        type=str,
        default=None,
        help="Path to condition metadata file. Specify when --skip_target is true.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Sample sequence length.",
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)

    seq_dict = pickle.load(open(args.metadata_pkl, "rb"))
    main(
        seq_dict,
        args.ct_feature,
        args.tf_feature,
        args.output_h5,
        args.output_types,
        args.compression,
        args.skip_feature,
        args.skip_target,
        args.condition_metadata,
        args.exclude_groups,
        args.sequence_length,
        args.normalization,
        args.motif_threshold,
        args.example_feature_start_id,
        args.example_feature_stop_id,
    )
