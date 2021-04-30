import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

from utils import display_args, print_time

parser = argparse.ArgumentParser("Compute class weight in target labels.")

parser.add_argument(
    "input_h5", type=str, help="Path to .h5 file containing target labels."
)
parser.add_argument(
    "output_npy", type=str, help="Path to output class occurrence .npy file."
)

args = parser.parse_args()
display_args(args, __file__)

############## MAIN ##############
# Constant
OUTPUT_SIZE = 2
OUTPUT_KEY = "output_conserved"

start_time = time.time()
count = np.zeros(OUTPUT_SIZE)
with h5py.File(args.input_h5, "r") as label_dset:
    label_group = label_dset[OUTPUT_KEY]
    for group_name in tqdm(label_group.keys()):
        target = label_group[group_name][:, :-3]
        unique = np.unique(target, return_counts=True)
        if len(unique[0]) != OUTPUT_SIZE:
            raise RuntimeError(
                "Invalid number of classes in target labels. Expect {}, got {} "
                "instead".format(OUTPUT_SIZE, len(unique[0]))
            )
        count += unique[1]

weight = count[::-1]
weight = weight / weight.sum()
np.save(args.output_npy, weight)

print_time("All samples processed.", start_time)
