import argparse
import os
import pickle
import time

import numpy as np
import pyBigWig
from tqdm import tqdm

import utils


############## FUNCTION ##############
def __file_attribute(dset):
    sample_index = np.array([dset["TF_id"], dset["CT_id"]])
    return sample_index, dset["start"], dset["stop"]


def __check_exist(filename):
    if os.path.isfile(filename):
        data = np.load(filename)["data"]
        if not np.isnan(data).any():
            return True
    return False


def retrieve_peaks(
    peak_file, peak_kwd, group_name, seq_dict, sequence_length, start_time
):
    peaks = pyBigWig.open(peak_file)
    num_files = len(seq_dict[peak_kwd][group_name].keys())
    utils.print_time("{} batches to process".format(num_files), start_time)

    for file in seq_dict[peak_kwd][group_name].keys():
        if __check_exist(file):
            utils.print_time("{} exists -- skip".format(file), start_time)
            continue

        # Initialize output signal.
        sample_index, start, stop = __file_attribute(
            seq_dict[peak_kwd][group_name][file]
        )
        signal = np.empty((stop - start, sequence_length + 3))
        signal[:] = np.NaN
        signal[:, -3:-1] = sample_index

        # Find peaks that overlap with each sample sequence.
        for k in tqdm(range(start, stop)):
            ks = k - start
            signal[ks, -1] = k
            signal[ks, :sequence_length] = 0
            sample = utils.Sample(seq_dict["input"][k])
            try:
                sample_peaks = peaks.entries(
                    sample.chrom,
                    sample.start,
                    sample.stop,
                    withString=False,
                )
            except RuntimeError as e:
                # pyBigWig invokes RuntimeError in rare occasions when no peak
                # from `sample.chrom` is included in the peak (big)bed file.
                utils.print_time(
                    "Input peak (big)bed file does not contain any peak from "
                    f"chromosome {sample.chrom}. Original error message: {e}",
                    start_time,
                )
            if sample_peaks is None:
                continue
            for s, t in sample_peaks:
                sample_start = max(0, s - sample.start)
                sample_stop = min(sequence_length, t - sample.start)
                signal[ks, sample_start:sample_stop] = 1

        # Save batch data file to disk.
        np.savez_compressed(
            file,
            group_name=group_name,
            peak_type=peak_kwd,
            start=start,
            stop=stop,
            data=signal,
        )
        utils.print_time(
            "{} targets saved in {}".format(peak_kwd, file), start_time
        )


def main(
    metadata_pkl,
    TF,
    cell_type,
    assay_type,
    peak_bed,
    output_kwd,
    sequence_length,
):
    seq_dict = pickle.load(open(metadata_pkl, "rb"))
    assert TF != ""
    assert cell_type != ""
    assert assay_type == "ChIP"
    group = ".".join([TF, cell_type]).strip(".")

    start_time = time.time()
    retrieve_peaks(
        peak_bed, output_kwd, group, seq_dict, sequence_length, start_time
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Retrieve sample target labels from ChIP-seq peaks."
    )

    parser.add_argument(
        "metadata_pkl", type=str, help="Path to example metadata .pkl file."
    )
    parser.add_argument("peak_bed", type=str, help="ChIP-seq peak bigbed file.")
    parser.add_argument("assay_type", type=str, help="Assay type.")
    parser.add_argument("TF", type=str, help="TF name.")
    parser.add_argument("cell_type", type=str, help="Cell type name.")
    parser.add_argument("output_kwd", type=str, help="Output data keyword.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Sample sequence length.",
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)
    main(
        args.metadata_pkl,
        args.TF,
        args.cell_type,
        args.assay_type,
        args.peak_bed,
        args.output_kwd,
        args.sequence_length,
    )
