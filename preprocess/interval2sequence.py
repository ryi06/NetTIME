import argparse
import pickle

import numpy as np
from tqdm import tqdm

import pybedtools
from utils import display_args

parser = argparse.ArgumentParser(
    "Generate sample sequences each of size <sequence_length> from a set of "
    "intervals saved in a bed file. Save sequence genomic location as well as "
    "fasta information into a pickle dictionary."
)

parser.add_argument("input_bed", type=str, help="Path to input bed file.")
parser.add_argument("output_prefix", type=str, help="Prefix of output files.")
parser.add_argument("fasta_file", type=str, help="Path to genome fasta file.")
parser.add_argument(
    "--sequence_length",
    type=int,
    default=1e3,
    help="Output sequence length. (Default 1000.)",
)

args = parser.parse_args()
display_args(args, __file__)


############## FUNCTION ##############
def _str2int(s, map_dict):
    return map_dict[s]


def nucleotide2index(sequence, nucleotide_dict):
    seq_array = np.array(list(sequence))
    return np.vectorize(_str2int)(seq_array, nucleotide_dict)


############## MAIN ##############
nucleotide_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
vocab = {"A", "C", "G", "T", "a", "c", "g", "t"}
fasta = pybedtools.BedTool(args.fasta_file)
seq_dict = {}
counter = 0

with open(args.input_bed, "r") as seq_file:
    num_lines = len(seq_file.read().split("\n")) - 1

with open(args.input_bed, "r") as seq_file:
    with open(args.output_prefix + ".bed", "w") as out_file:
        for line in tqdm(seq_file, total=num_lines):
            record = line.strip().split("\t")
            midpoint = (int(record[2]) - int(record[1])) // 2 + int(record[1])

            # Get peak sequence positions.
            chrom = record[0]
            start = midpoint - int(args.sequence_length // 2)
            stop = midpoint + int(args.sequence_length // 2)

            # Get peak sequence fasta.
            entry = " ".join([chrom, str(start), str(stop)])
            tmp = pybedtools.BedTool(entry, from_string=True).sequence(fi=fasta)
            seq = open(tmp.seqfn).read().strip().split("\n")[1].upper()

            # Write sample sequence to dictionary.
            assert len(seq) == args.sequence_length
            chars = set(seq)
            if chars.issubset(vocab):
                seq_dict[counter] = {
                    "chrom": chrom,
                    "start": start,
                    "stop": stop,
                    "fasta": nucleotide2index(seq, nucleotide_dict),
                }
                out_file.write("{}\t{:d}\t{:d}\n".format(chrom, start, stop))
                out_file.flush()
                counter += 1
            else:
                print("Poor quality sequence found.")

# Save sample sequence dictionary to output pickle file.
output_dict = {"input": seq_dict}
pickle.dump(output_dict, open(args.output_prefix + ".pkl", "wb"), protocol=4)
