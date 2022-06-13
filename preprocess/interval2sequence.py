import argparse
import h5py
import os
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import pybedtools
import utils

_NUCLEOTIDE_DICT = {"A": 0, "C": 1, "G": 2, "T": 3}
_VOCAB = {"A", "C", "G", "T", "a", "c", "g", "t"}
_VOCAB_SIZE = len(_NUCLEOTIDE_DICT)


############## FUNCTION ##############
def _str2int(s, map_dict):
    return map_dict[s]


def nucleotide2index(sequence):
    seq_array = np.array(list(sequence))
    return np.vectorize(_str2int)(seq_array, _NUCLEOTIDE_DICT)


def one_hot_encoder(index, shape):
    one_hot = np.zeros(shape)
    one_hot[np.arange(shape[0]), index] = 1
    return one_hot


def generate_hdf5(
    examples,
    output_path,
    output_bed,
    sequence_length,
    bucket_size=utils.BUCKET_SIZE,
):
    counter = 0
    with open(output_bed, "w") as out_file:
        with h5py.File(output_path, "w", libver="latest", swmr=True) as h5_file:
            input_group = h5_file.create_group("input")
            for entry in tqdm(examples):
                chrom, start, stop, seq = re.split(":|-|\n", entry.strip())
                seq = seq.upper()
                assert len(seq) == sequence_length

                chars = set(seq)
                if chars.issubset(_VOCAB):
                    example_path = os.path.join(
                        "/input",
                        str(int(counter // bucket_size)),
                        str(counter),
                    )
                    example_group = input_group.create_group(example_path)
                    example_group.attrs.create("chrom", chrom)
                    example_group.attrs.create("start", int(start))
                    example_group.attrs.create("stop", int(stop))
                    seq_fasta = nucleotide2index(seq)
                    example_group.create_dataset(
                        "ids",
                        (sequence_length,),
                        data=seq_fasta,
                        compression="gzip",
                    )
                    example_group.create_dataset(
                        "fasta",
                        (sequence_length, _VOCAB_SIZE),
                        data=one_hot_encoder(
                            seq_fasta, (sequence_length, _VOCAB_SIZE)
                        ),
                        compression="gzip",
                    )
                    counter += 1
                    out_file.write("{}\t{}\t{}\n".format(chrom, start, stop))
                    out_file.flush()
                else:
                    print("Poor quality sequence found.")


def generate_pkl(h5_path, pkl_path):
    seq_dict = {"input": {}}
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as h5_file:
        h5_input_group = h5_file["input"]
        bucket_keys = list(h5_input_group.keys())
        bucket_keys.sort(key=int)
        for bucket in tqdm(bucket_keys):
            example_keys = list(h5_input_group[bucket].keys())
            example_keys.sort(key=int)
            for key in example_keys:
                i = int(key)
                example_group = h5_input_group[bucket][key]
                seq_dict["input"][i] = {
                    "chrom": example_group.attrs["chrom"],
                    "start": example_group.attrs["start"],
                    "stop": example_group.attrs["stop"],
                }
    # Save sample sequence dictionary to output pickle file.
    pickle.dump(seq_dict, open(pkl_path, "wb"), protocol=4)


def generate_example_bed(
    input_bed, output_bed, sequence_length, chrom_size_path
):
    chrom_sizes = pd.read_csv(
        chrom_size_path, sep="\t", header=None, names=["chr", "size"]
    )
    seq_bed = pd.read_csv(
        input_bed, sep="\t", header=None, names=["chr", "start", "stop"]
    )

    with open(output_bed, "w") as out_file:
        for _, row in tqdm(seq_bed.iterrows(), total=len(seq_bed)):
            if row.stop - row.start == sequence_length:
                start, stop = row.start, row.stop
            else:
                midpoint = (row.stop - row.start) // 2 + row.start
                # Get peak sequence positions.
                start = midpoint - int(sequence_length // 2)
                stop = midpoint + int(sequence_length // 2)
            if (start < 0) or (
                stop > chrom_sizes[chrom_sizes.chr == row.chr]["size"].item()
            ):
                print(f"{row.chr}, {start}, {stop} index out of bound.")
                continue
            out_file.write("{}\t{:d}\t{:d}\n".format(row.chr, start, stop))
            out_file.flush()


def main(fasta_file, output_prefix, input_bed, sequence_length, chrom_sizes):
    fasta = pybedtools.BedTool(fasta_file)
    # Generate example bed file and write to disk.
    output_bed = output_prefix + ".bed"
    generate_example_bed(input_bed, output_bed, sequence_length, chrom_sizes)

    # Get fasta of example sequences.
    example_bed = pybedtools.BedTool(output_bed)
    example_fasta = example_bed.sequence(fi=fasta)
    examples = open(example_fasta.seqfn).read().split(">")[1:]

    # Write sample sequence to dictionary.
    output_h5 = output_prefix + "_seq_feature.h5"
    generate_hdf5(examples, output_h5, output_bed, sequence_length)

    # Write metadata into a separate pkl file.
    output_pkl = output_prefix + "_metadata.pkl"
    generate_pkl(output_h5, output_pkl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generate sample sequences each of size <sequence_length> from a set "
        "of intervals saved in a bed file. Save sequence genomic location as "
        "well as fasta information into a pickle dictionary."
    )

    parser.add_argument("input_bed", type=str, help="Path to input bed file.")
    parser.add_argument(
        "output_prefix", type=str, help="Prefix of output files."
    )
    parser.add_argument(
        "fasta_file", type=str, help="Path to genome fasta file."
    )
    parser.add_argument(
        "chrom_sizes", type=str, help="Path to chromosome size file."
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1e3,
        help="Output sequence length. (Default 1000.)",
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)
    main(
        args.fasta_file,
        args.output_prefix,
        args.input_bed,
        args.sequence_length,
        args.chrom_sizes,
    )
