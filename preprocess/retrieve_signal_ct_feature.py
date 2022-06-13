import argparse
import gc
import os
import pickle
import time
from subprocess import check_call

import numpy as np
import pybedtools
import pyBigWig
from tqdm import tqdm

import utils


############## FUNCTION ##############
def retrieve_signal(
    peak_file,
    bigWig_file,
    seq_dict,
    group_name,
    assay_type,
    batch_size,
    sequence_length,
):
    peaks = pyBigWig.open(peak_file)
    bigWig = pyBigWig.open(bigWig_file)
    num_samples = len(seq_dict["input"])
    writer = utils.FeatureWriter(
        batch_size,
        sequence_length,
        num_samples,
        group_name,
        assay_type,
    )

    for k in tqdm(range(len(seq_dict["input"]))):
        # Initialize signal track.
        signal = np.zeros(sequence_length)

        # Construct BedTool input from sample sequence location.
        sample = utils.Sample(seq_dict["input"][k])

        # Get peaks that overlap with sample.
        sample_peaks = peaks.entries(
            sample.chrom, sample.start, sample.stop, withString=False
        )

        # Retrieve sample bigwig signal that fall within peak regions.
        if sample_peaks is not None:
            for peak_start, peak_stop in sample_peaks:
                peak_signal = bigWig.intervals(
                    sample.chrom,
                    max(peak_start, sample.start),
                    min(peak_stop, sample.stop),
                )
                for s, t, v in peak_signal:
                    ss = max(peak_start, s, sample.start) - sample.start
                    tt = min(peak_stop, t, sample.stop) - sample.start
                    signal[ss:tt] = v

        # Write signal track to disk.
        writer.write_feature(
            signal, k, seq_dict["input"][k][assay_type][group_name]
        )
    return writer


def get_zscore_params(writer, save_params, start_time):
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
            batch = np.load(path)["original"]
            mean_calculator.add_batch_data(batch)
        mu = mean_calculator.get_mean()

        # Calculate standard deviation.
        utils.print_time("Calculating sample standard deviation.", start_time)
        var_calculator = utils.BatchVarianceCalculator(mu)
        for path in writer.paths:
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


def zscore_signal(writer, save_params, start_time):
    # Retrieve zscore params.
    normalizer = get_zscore_params(writer, save_params, start_time)

    # Zscore batch and save batch to disk.
    for path in tqdm(writer.paths):
        batch = dict(np.load(path))
        batch["zscore"] = normalizer.normalize(batch["original"])
        np.savez_compressed(path, **batch)


def main(
    seq_dict,
    peak_file,
    bigWig_file,
    cell_type,
    assay_type,
    zscore_folder,
    sequence_length,
):
    start_time = time.time()

    batch_size = seq_dict['batch_size']
    utils.print_time(f'Writing {batch_size} examples per file.', start_time)

    utils.print_time("Getting original feature signal", start_time)
    signal_writer = retrieve_signal(
        peak_file,
        bigWig_file,
        seq_dict,
        cell_type,
        assay_type,
        batch_size,
        sequence_length,
    )

    utils.print_time("Normalizing feature signal", start_time)
    zscore_file = os.path.join(
        zscore_folder,
        "{}.{}.zscore_params.npz".format(assay_type, cell_type),
    )
    zscore_signal(signal_writer, zscore_file, start_time)
    utils.print_time("All samples processed!", start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Retrieve cell type feature signals for sample sequences."
    )

    parser.add_argument(
        "metadata_pkl", type=str, help="Path to example .pkl file."
    )
    parser.add_argument("peak_file", type=str, help="Feature peak bigbed file.")
    parser.add_argument(
        "bigWig_file", type=str, help="Feature signal bigWig file."
    )
    parser.add_argument("assay_type", type=str, help="Assay type")
    parser.add_argument("cell_type", type=str, help="Name of the cell type.")
    parser.add_argument(
        "zscore_folder", type=str, help="Output zscore params file path."
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
        args.peak_file,
        args.bigWig_file,
        args.cell_type,
        args.assay_type,
        args.zscore_folder,
        args.sequence_length,
    )
