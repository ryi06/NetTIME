import argparse
import time

import h5py
import numpy as np
from tqdm import tqdm

import utils


def main(input_h5, output_npy):
    # Constant
    _output_size = 2
    _output_key = "output_conserved"
    start_time = time.time()

    count = np.zeros(_output_size)
    with h5py.File(input_h5, "r") as label_dset:
        label_group = label_dset[_output_key]
        for group_name in tqdm(label_group.keys()):
            for bucket in (label_group[group_name].keys()):
                target = label_group[group_name][bucket][:, :-3]
                unique = np.unique(target, return_counts=True)
                if len(unique[0]) != _output_size:
                    raise RuntimeError(
                        "Invalid number of classes in target labels. Expect {}, "
                        "got {} instead".format(_output_size, len(unique[0]))
                    )
                count += unique[1]

    weight = count[::-1]
    weight = weight / weight.sum()
    np.save(output_npy, weight)
    utils.print_time("All samples processed.", start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute class weight in target labels.")

    parser.add_argument(
        "input_h5", type=str, help="Path to .h5 file containing target labels."
    )
    parser.add_argument(
        "output_npy",
        type=str,
        help="Path to output class occurrence .npy file.",
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)
    main(args.input_h5, args.output_npy)
