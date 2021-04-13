import argparse
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

import pybedtools
from utils import Sample, display_args, print_time

parser = argparse.ArgumentParser(
    "Retrieve sample target labels from ChIP-seq peaks."
)

parser.add_argument("metadata_pkl", type=str, help="Path to example .pkl file.")
parser.add_argument("peak_bed", type=str, help="ChIP-seq peak bed file.")
parser.add_argument("assay_type", type=str, help="Assay type.")
parser.add_argument("TF", type=str, help="TF name.")
parser.add_argument("cell_type", type=str, help="Cell type name.")
parser.add_argument("output_kwd", type=str, help="Output data keyword.")
parser.add_argument(
    "--sequence_length", type=int, default=1000, help="Sample sequence length."
)

args = parser.parse_args()
display_args(args, __file__)


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


def retrieve_peaks(peak_file, peak_kwd, group_name, seq_dict):
    peaks = pybedtools.BedTool(peak_file)
    num_files = len(seq_dict[peak_kwd][group_name].keys())
    print_time("{} batches to process".format(num_files), start_time)

    for file in seq_dict[peak_kwd][group_name].keys():
        if __check_exist(file):
            print_time("{} exists -- skip".format(file), start_time)
            continue

        # Initialize output signal.
        sample_index, start, stop = __file_attribute(
            seq_dict[peak_kwd][group_name][file]
        )
        signal = np.empty((stop - start, args.sequence_length + 3))
        signal[:] = np.NaN
        signal[:, -3:-1] = sample_index

        # Find peaks that overlap with each sample sequence.
        for k in tqdm(range(start, stop)):
            ks = k - start
            signal[ks, -1] = k
            signal[ks, : args.sequence_length] = 0
            sample = Sample(seq_dict["input"][k])
            entry = "{} {} {}".format(sample.chrom, sample.start, sample.stop)
            a = pybedtools.BedTool(entry, from_string=True)
            apeaks = a.intersect(peaks)
            for p in apeaks:
                s = p.start - sample.start
                t = p.stop - sample.start
                signal[ks, s:t] = 1
            if (k + 1) % 1000 == 0:
                pybedtools.cleanup(remove_all=True)

        # Save batch data file to disk.
        np.savez_compressed(
            file,
            group_name=group_name,
            peak_type=peak_kwd,
            start=start,
            stop=stop,
            data=signal,
        )
        print_time("{} targets saved in {}".format(peak_kwd, file), start_time)


############## MAIN ##############
seq_dict = pickle.load(open(args.metadata_pkl, "rb"))
assert args.TF != ""
assert args.cell_type != ""
assert args.assay_type == "ChIP"
group = ".".join([args.TF, args.cell_type]).strip(".")

start_time = time.time()
retrieve_peaks(args.peak_bed, args.output_kwd, group, seq_dict)
