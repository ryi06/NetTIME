import argparse
import os
import pickle
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import FeatureWriter, SignalNormalizer, print_time, display_args

parser = argparse.ArgumentParser(
    "Retrieve TF motif enrichment scores for sample sequences."
)

parser.add_argument("metadata_pkl", type=str, help="Path to example .pkl file.")
parser.add_argument("fasta_file", type=str, help="Path to example .fa file.")
parser.add_argument("motif_file", type=str, help="Path to meme motif file.")
parser.add_argument("assay_type", type=str, help="Assay type.")
parser.add_argument("tf", type=str, help="Name of the TF.")
parser.add_argument(
    "zscore_folder", type=str, help="Output zscore params file path."
)
parser.add_argument(
    "--sequence_length", type=int, default=1000, help="Sample sequence length."
)
parser.add_argument(
    "--batch_size", type=int, default=10000, help="Data file batch size."
)
parser.add_argument(
    "--tmp_dir", type=str, default=".", help="Path to temp directory."
)
parser.add_argument(
    "--threshold", type=float, default=1e-2, help="FIMO p-value threshold"
)

args = parser.parse_args()
display_args(args, __file__)


############## FUNCTION ##############
def __fill_array_max_score(arr, fimo, start_id, strand_idx):
    for _, row in fimo.iterrows():
        start = row["start"] - start_id
        stop = row["stop"] - start_id
        enrichment = row["score"]

        arr[start:stop, strand_idx] = np.maximum(
            arr[start:stop, strand_idx], enrichment
        )


def generate_motif_enrichment_array(start, fimo_result):
    enrichment = np.zeros((1000, 2))
    if fimo_result is None or fimo_result.empty:
        return enrichment
    # strands
    plus_strand = fimo_result[fimo_result["strand"] == "+"]
    minus_strand = fimo_result[fimo_result["strand"] == "-"]

    __fill_array_max_score(enrichment, plus_strand, start, 0)
    __fill_array_max_score(enrichment, minus_strand, start, 1)

    return enrichment


def get_motif_enrichment(header, sequence, fasta_file, tmp_dir):
    # create tmp fasta file.
    with open(fasta_file, "w") as outfasta:
        outfasta.writelines([header, sequence])

    # Run FIMO.
    tmp_fimo = os.path.join(tmp_dir, "fimo.tsv")
    cmd = (
        "fimo --parse-genomic-coord --skip-matched-sequence --text "
        "--verbosity 1 --thresh {} {} {} > {}"
    ).format(args.threshold, args.motif_file, fasta_file, tmp_fimo)
    subprocess.run(cmd, check=True, shell=True)

    # Return fimo result.
    try:
        result = pd.read_csv(
            tmp_fimo,
            sep="\t",
            skipfooter=3,
            engine="python",
            dtype={"start": "int64", "stop": "int64"},
        )
        return result
    except pd.errors.EmptyDataError:
        return


def wrapper(fasta_file, seq_dict, tmp_dir, group_name, assay_type):
    tmp_fasta = os.path.join(tmp_dir, "tmp.fasta")
    num_samples = len(seq_dict["input"])
    with open(fasta_file, "r") as fasta:
        num_lines = len(fasta.read().split("\n")) - 1
    assert num_samples == round(num_lines / 2)
    writer = FeatureWriter(
        args.batch_size,
        args.sequence_length,
        num_samples,
        group_name,
        assay_type,
        num_features=NUM_FEATURES,
    )

    with open(fasta_file, "r") as fasta:
        for k in tqdm(seq_dict["input"].keys()):
            # Read the next fasta entry.
            label = fasta.readline()
            seq = fasta.readline()

            # Assert it's the right sample.
            _, chrom, start, stop = re.split(">|:|-", label.strip())
            assert chrom == seq_dict["input"][k]["chrom"]
            assert int(start) == seq_dict["input"][k]["start"]
            assert int(stop) == seq_dict["input"][k]["stop"]

            # Find sample motif enrichment using FIMO.
            fimo = get_motif_enrichment(label, seq, tmp_fasta, tmp_dir)
            enrichment = generate_motif_enrichment_array(int(start), fimo)

            # Write enrichment scores to disk.
            writer.write_feature(
                enrichment,
                k,
                seq_dict["input"][k][assay_type][group_name],
                threshold=args.threshold,
            )

    return writer


def merge_signal(writer, seq_length):
    original = np.empty((writer.counter, seq_length, NUM_FEATURES))
    original[:] = np.NaN
    for path in writer.paths:
        path = path.replace(
            ".npz", ".threshold{:.0e}.npz".format(args.threshold)
        )
        batch = np.load(path)
        print_time("[{}:{}]".format(batch["start"], batch["stop"]), start_time)
        original[batch["start"] : batch["stop"]] = batch["original"]
    return original


def zscore_signal(writer, original, save_params):
    # Retrieve/generate zscore params
    if os.path.isfile(save_params):
        params = np.load(save_params)
        normalizer = SignalNormalizer(
            "zscore", mu=params["mu"], std=params["std"]
        )
    else:
        mu = np.mean(original)
        std = np.std(original)
        normalizer = SignalNormalizer(
            "zscore", save_params=save_params, mu=mu, std=std
        )
    print_time(
        "mu={:.4f}, std={:.4f} obtained from {}".format(
            normalizer.mu, normalizer.std, save_params
        ),
        start_time,
    )
    zscore = normalizer.normalize(original)

    # Save zscored signal to disk.
    for path in writer.paths:
        path = path.replace(
            ".npz", ".threshold{:.0e}.npz".format(args.threshold)
        )
        batch = dict(np.load(path))
        batch["zscore"] = zscore[batch["start"] : batch["stop"]]
        np.savez_compressed(path, **batch)


############## MAIN ##############
seq_dict = pickle.load(open(args.metadata_pkl, "rb"))
start_time = time.time()
NUM_FEATURES = 2  # Plus and minus strands

print_time("Getting original feature signal", start_time)
signal_writer = wrapper(
    args.fasta_file, seq_dict, args.tmp_dir, args.tf, args.assay_type
)

print_time("Merging original data", start_time)
original_data = merge_signal(signal_writer, args.sequence_length)

print_time("Normalizing feature signal", start_time)
zscore_file = os.path.join(
    args.zscore_folder,
    "{}.{}.zscore_params.threshold{:.0e}.npz".format(
        args.assay_type, args.tf, args.threshold
    ),
)
zscore_signal(signal_writer, original_data, zscore_file)

print_time("All sample processed!", start_time)
