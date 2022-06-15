import collections

import copy
import numpy as np
import pandas as pd

EXPECTED_EXAMPLE_BED = pd.DataFrame(
    [
        ["chr1", 11, 21],
        ["chr1", 10, 20],
        ["chr1", 249, 259],  # contains n.
        ["chr2", 0, 10],  # contains N.
        ["chr2", 32, 42],
        ["chr2", 50, 60],
        ["chr2", 39, 49],
        ["chr2", 490, 500],
        ["chr3", 1, 11],
        ["chr3", 85, 95],
    ]
)

EXPECTED_EXAMPLE_SEQ_FEATURE = [
    # (dataset_path, fasta_ids)
    ("input/0/0", np.array([1, 3, 2, 2, 3, 1, 3, 1, 0, 0])),  # 'ctggtctcaa'
    ("input/0/1", np.array([2, 1, 3, 2, 2, 3, 1, 3, 1, 0])),  # 'gctggtctca'
    ("input/0/2", np.array([0, 3, 0, 0, 3, 2, 0, 1, 3, 0])),  # 'ATAATGACTA'
    ("input/0/3", np.array([1, 0, 1, 0, 2, 0, 3, 0, 3, 3])),  # 'CACAGATATT'
    ("input/0/4", np.array([1, 3, 0, 0, 3, 0, 3, 1, 1, 0])),  # 'CTAATATCCA'
    ("input/1/5", np.array([0, 2, 3, 3, 2, 0, 3, 3, 3, 0])),  # 'agttgattta'
    ("input/1/6", np.array([0, 1, 3, 0, 1, 0, 1, 1, 3, 3])),  # 'actACacctt'
    ("input/1/7", np.array([0, 0, 3, 0, 0, 2, 2, 0, 2, 0])),  # 'AATAAGGAGA'
]


EXPECTED_EXAMPLE_METADATA = collections.OrderedDict(
    {
        "input": {
            0: {"chrom": "chr1", "start": 11, "stop": 21},
            1: {"chrom": "chr1", "start": 10, "stop": 20},
            2: {"chrom": "chr2", "start": 32, "stop": 42},
            3: {"chrom": "chr2", "start": 50, "stop": 60},
            4: {"chrom": "chr2", "start": 39, "stop": 49},
            5: {"chrom": "chr2", "start": 490, "stop": 500},
            6: {"chrom": "chr3", "start": 1, "stop": 11},
            7: {"chrom": "chr3", "start": 85, "stop": 95},
        },
        "batch_size": 5,
    }
)


def expected_example_metadata(output_folder):
    # result = collections.OrderedDict(
    #     {"input": copy.deepcopy(EXPECTED_EXAMPLE_METADATA["input"])}
    # )
    result = copy.deepcopy(EXPECTED_EXAMPLE_METADATA)
    for i in range(8):
        div, mod = divmod(i, 5)
        result["input"][i]["ctf1"] = {
            "ct1": {
                "path": f"{output_folder}/feature/ctf1.ct1.{div}.npz",
                "index": mod,
            },
            "ct2": {
                "path": f"{output_folder}/feature/ctf1.ct2.{div}.npz",
                "index": mod,
            },
        }
        result["input"][i]["ctf2"] = {
            "ct1": {
                "path": f"{output_folder}/feature/ctf2.ct1.{div}.npz",
                "index": mod,
            },
            "ct2": {
                "path": f"{output_folder}/feature/ctf2.ct2.{div}.npz",
                "index": mod,
            },
        }
        result["input"][i]["tff1"] = {
            "tf1": {
                "path": f"{output_folder}/feature/tff1.tf1.{div}.npz",
                "index": mod,
            },
            "tf2": {
                "path": f"{output_folder}/feature/tff1.tf2.{div}.npz",
                "index": mod,
            },
            "tf3": {
                "path": f"{output_folder}/feature/tff1.tf3.{div}.npz",
                "index": mod,
            },
        }

    result["output_conserved"] = {
        "tf1.ct1": {
            f"{output_folder}/target/tf1.ct1.conserved.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 0,
                "CT_id": 0,
            },
            f"{output_folder}/target/tf1.ct1.conserved.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 0,
                "CT_id": 0,
            },
        },
        "tf2.ct1": {
            f"{output_folder}/target/tf2.ct1.conserved.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 1,
                "CT_id": 0,
            },
            f"{output_folder}/target/tf2.ct1.conserved.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 1,
                "CT_id": 0,
            },
        },
        "tf3.ct2": {
            f"{output_folder}/target/tf3.ct2.conserved.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 2,
                "CT_id": 1,
            },
            f"{output_folder}/target/tf3.ct2.conserved.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 2,
                "CT_id": 1,
            },
        },
    }
    result["output_relaxed"] = {
        "tf1.ct1": {
            f"{output_folder}/target/tf1.ct1.relaxed.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 0,
                "CT_id": 0,
            },
            f"{output_folder}/target/tf1.ct1.relaxed.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 0,
                "CT_id": 0,
            },
        },
        "tf2.ct1": {
            f"{output_folder}/target/tf2.ct1.relaxed.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 1,
                "CT_id": 0,
            },
            f"{output_folder}/target/tf2.ct1.relaxed.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 1,
                "CT_id": 0,
            },
        },
        "tf3.ct2": {
            f"{output_folder}/target/tf3.ct2.relaxed.0.npz": {
                "start": 0,
                "stop": 5,
                "TF_id": 2,
                "CT_id": 1,
            },
            f"{output_folder}/target/tf3.ct2.relaxed.1.npz": {
                "start": 5,
                "stop": 8,
                "TF_id": 2,
                "CT_id": 1,
            },
        },
    }
    return result


PEAKS_BED = pd.DataFrame(
    [
        ["chr1", 13, 19, "name", 1, "."],  # ["chr1", 11, 21] & ["chr1", 10, 20]
        ["chr2", 25, 40, "name", 2, "."],  # ["chr2", 32, 42] & ["chr2", 39, 49]
        ["chr2", 55, 70, "name", 5, "."],  # ["chr2", 50, 60]
        ["chr2", 480, 490, "name", 7, "."],  # No overlap
        ["chr3", 0, 20, "name", 9, "."],  # ["chr3", 1, 11]
        ["chr3", 80, 86, "name", 2, "."],  # ["chr3", 85, 95]
        ["chr3", 90, 100, "name", 2, "."],  # ["chr3", 85, 95]
    ]
)

EXPECTED_CHIPSEQ_DATA = np.array(
    [
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0],  # ["chr1", 11, 21]
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1],  # ["chr1", 10, 20]
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 2],  # ["chr2", 32, 42]
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 3],  # ["chr2", 50, 60]
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 4],  # ["chr2", 39, 49]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 5],  # ["chr2", 490, 500]
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 6],  # ["chr3", 1, 11]
        [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 7],  # ["chr3", 85, 95]
    ],
    dtype=np.float,
)

SIGNAL_BEDGRAPH = pd.DataFrame(
    [
        ["chr1", 0, 10, 3],
        ["chr1", 10, 20, 5],
        ["chr1", 20, 30, 2],
        ["chr1", 30, 40, 4],
        ["chr2", 0, 10, 2],
        ["chr2", 20, 30, 9],
        ["chr2", 30, 35, 20],
        ["chr2", 35, 50, 19],
        ["chr2", 50, 60, 12],
        ["chr2", 60, 70, 16],
        ["chr2", 480, 500, 15],
        ["chr3", 0, 10, 8],
        ["chr3", 10, 20, 7],
        ["chr3", 80, 91, 13],
        ["chr3", 91, 100, 6],
    ]
)

EXPECTED_CT_FEATURE_DATA = np.array(
    [
        [0, 0, 5, 5, 5, 5, 5, 5, 0, 0],  # ["chr1", 11, 21]
        [0, 0, 0, 5, 5, 5, 5, 5, 5, 0],  # ["chr1", 10, 20]
        [20, 20, 20, 19, 19, 19, 19, 19, 0, 0],  # ["chr2", 32, 42]
        [0, 0, 0, 0, 0, 12, 12, 12, 12, 12],  # ["chr2", 50, 60]
        [19, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ["chr2", 39, 49]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ["chr2", 490, 500]
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 7],  # ["chr3", 1, 11]
        [13, 0, 0, 0, 0, 13, 6, 6, 6, 6],  # ["chr3", 85, 95]
    ],
    dtype=np.float,
)

EXPECTED_CT_FEATURE_MU = 5.2875
EXPECTED_CT_FEATURE_STD = 6.380034776550986
EXPECTED_ZSCORED_CT_FEATURE_DATA = (
    EXPECTED_CT_FEATURE_DATA - EXPECTED_CT_FEATURE_MU
) / EXPECTED_CT_FEATURE_STD

SIGNAL_BED = pd.DataFrame(
    [
        ["chr1", 7, 15, 0.9, 100, "+"],
        ["chr1", 17, 20, 0.85, 100, "-"],
        ["chr2", 35, 42, 0.2, 100, "-"],
        ["chr2", 492, 496, 0.6, 100, "+"],
        ["chr3", 5, 8, 0.3, 100, "-"],
        ["chr3", 80, 100, 0.4, 100, "+"],
    ]
)


EXPECTED_TF_FEATURE_DATA = np.array(
    [
        [
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0, 0],
            [0, 0],
            [0, 0.15],
            [0, 0.15],
            [0, 0.15],
            [0, 0],
        ],  # ["chr1", 11, 21]
        [
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0.1, 0],
            [0, 0],
            [0, 0],
            [0, 0.15],
            [0, 0.15],
            [0, 0.15],
        ],  # ["chr1", 10, 20]
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
        ],  # ["chr2", 32, 42]
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],  # ["chr2", 50, 60]
        [
            [0, 0.8],
            [0, 0.8],
            [0, 0.8],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],  # ["chr2", 39, 49]
        [
            [0, 0],
            [0, 0],
            [0.4, 0],
            [0.4, 0],
            [0.4, 0],
            [0.4, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],  # ["chr2", 490, 500]
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0.7],
            [0, 0.7],
            [0, 0.7],
            [0, 0],
            [0, 0],
            [0, 0],
        ],  # ["chr3", 1, 11]
        [
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
            [0.6, 0],
        ],  # ["chr3", 85, 95]
    ]
)

EXPECTED_TF_FEATURE_MU = 0.121875
EXPECTED_TF_FEATURE_STD = 0.2494799277998132

EXPECTED_ZSCORED_TF_FEATURE_DATA = (
    EXPECTED_TF_FEATURE_DATA - EXPECTED_TF_FEATURE_MU
) / EXPECTED_TF_FEATURE_STD
