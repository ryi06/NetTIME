import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyBigWig
from tqdm import tqdm

import utils


############## FUNCTION ##############
def retrieve_signal(
    enrichment_file,
    seq_dict,
    group_name,
    assay_type,
    batch_size,
    threshold,
    sequence_length,
    num_features,
):
    enrichments = pyBigWig.open(enrichment_file)
    num_samples = len(seq_dict["input"])
    writer = utils.FeatureWriter(
        batch_size,
        sequence_length,
        num_samples,
        group_name,
        assay_type,
        num_features=num_features,
    )

    for k in tqdm(range(len(seq_dict["input"]))):
        # Initialize signal track.
        signal = np.zeros((sequence_length, num_features))

        # Construct BedTool input from sample sequence location.
        sample = utils.Sample(seq_dict["input"][k])

        # Get peaks that overlap with sample.
        sample_enrichments = enrichments.entries(
            sample.chrom, sample.start, sample.stop
        )

        if sample_enrichments is not None:
            sample_enrichments = set(sample_enrichments)
            for enrich_start, enrich_stop, values in sample_enrichments:
                s = max(0, enrich_start - sample.start)
                t = min(sequence_length, enrich_stop - sample.start)
                
                pval, _, strand = values.split("\t")
                assert strand in ["-", "+"]
                strand_index = 0 if strand == "+" else 1

                # Store max enrichment score.
                signal[s:t, strand_index] = np.maximum(
                    signal[s:t, strand_index], 1 - float(pval)
                )

        # Write enrichment scores to disk.
        writer.write_feature(
            signal,
            k,
            seq_dict["input"][k][assay_type][group_name],
            threshold=threshold,
        )

    return writer


def _add_threshold_to_filename(path, threshold):
    return path.replace(".npz", ".threshold{:.0e}.npz".format(threshold))


def get_zscore_params(writer, save_params, threshold, start_time):
    if os.path.isfile(save_params):
        params = np.load(save_params)
        normalizer = utils.SignalNormalizer(
            "zscore", mu=params["mu"], std=params["std"]
        )
    else:
        # Calculate mean.
        utils.print_time("Calculating sample mean.", start_time)
        mean_calculator = utils.BatchMeanCalculator()
        for path in writer.paths:
            path = _add_threshold_to_filename(path, threshold)
            batch = np.load(path)["original"]
            mean_calculator.add_batch_data(batch)
        mu = mean_calculator.get_mean()

        # Calculate standard deviation.
        utils.print_time("Calculating sample standard deviation.", start_time)
        var_calculator = utils.BatchVarianceCalculator(mu)
        for path in writer.paths:
            path = _add_threshold_to_filename(path, threshold)
            batch = np.load(path)["original"]
            var_calculator.add_batch_data(batch)
        std = var_calculator.get_standard_deviation()

        normalizer = utils.SignalNormalizer(
            "zscore", save_params=save_params, mu=mu, std=std
        )
    utils.print_time(
        "mu={:.4f}, std={:.4f} obtained from {}".format(
            normalizer.mu, normalizer.std, save_params
        ),
        start_time,
    )

    return normalizer


def zscore_signal(writer, save_params, threshold, start_time):
    # Retrieve/generate zscore params.
    normalizer = get_zscore_params(writer, save_params, threshold, start_time)

    # Save zscored signal to disk.
    for path in writer.paths:
        path = _add_threshold_to_filename(path, threshold)
        batch = dict(np.load(path))
        batch["zscore"] = normalizer.normalize(batch["original"])
        np.savez_compressed(path, **batch)


def main(
    seq_dict,
    fimo_bigbed,
    assay_type,
    tf,
    zscore_folder,
    sequence_length,
    threshold,
):
    start_time = time.time()
    num_features = 2  # Plus and minus strands
    batch_size = seq_dict["batch_size"]
    utils.print_time(f"Writing {batch_size} examples per file.", start_time)

    utils.print_time("Getting original feature signal", start_time)
    signal_writer = retrieve_signal(
        fimo_bigbed,
        seq_dict,
        tf,
        assay_type,
        batch_size,
        threshold,
        sequence_length,
        num_features,
    )

    utils.print_time("Normalizing feature signal", start_time)
    zscore_file = os.path.join(
        zscore_folder,
        "{}.{}.zscore_params.threshold{:.0e}.npz".format(
            assay_type, tf, threshold
        ),
    )
    zscore_signal(signal_writer, zscore_file, threshold, start_time)
    utils.print_time("All samples processed!", start_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Retrieve TF motif enrichment scores for sample sequences."
    )

    parser.add_argument(
        "metadata_pkl", type=str, help="Path to example .pkl file."
    )
    parser.add_argument(
        "fimo_bigbed",
        type=str,
        help="Path to a .bigbed file converted from FIMO result .tsv file.",
    )
    parser.add_argument("assay_type", type=str, help="Assay type.")
    parser.add_argument("tf", type=str, help="Name of the TF.")
    parser.add_argument(
        "zscore_folder", type=str, help="Output zscore params file path."
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Sample sequence length.",
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-2, help="FIMO p-value threshold"
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)

    seq_dict = pickle.load(open(args.metadata_pkl, "rb"))
    main(
        seq_dict,
        args.fimo_bigbed,
        args.assay_type,
        args.tf,
        args.zscore_folder,
        args.sequence_length,
        args.threshold,
    )
