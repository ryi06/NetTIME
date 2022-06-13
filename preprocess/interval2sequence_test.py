import os
import pickle
import tempfile

import h5py
import numpy as np
import pandas as pd
import pybedtools
import pytest

import interval2sequence

from ..testdata import expected_data

_INPUT_BED_PATH = "../testdata/peaks.bed"
_OUTPUT_BED_NAME = "examples.bed"
_SEQUENCE_LENGTH = 10
_SEQ_FEATURE_NAME = "seq_feature.h5"
_METADATA_NAME = "metadata.pkl"

_CHROM_SIZES = "../testdata/chrom.sizes"
_GENOME_FA = "../testdata/genome.fa"


def _generate_output_bed_path(path):
    return os.path.join(path, _OUTPUT_BED_NAME)


def _generate_seq_feature_path(path):
    return os.path.join(path, _SEQ_FEATURE_NAME)


def _generate_metadata_path(path):
    return os.path.join(path, _METADATA_NAME)


@pytest.fixture
def _example_fasta(tmp_path):
    output_bed_path = _generate_output_bed_path(tmp_path)
    interval2sequence.generate_example_bed(
        _INPUT_BED_PATH, output_bed_path, _SEQUENCE_LENGTH, _CHROM_SIZES
    )
    example_bed = pybedtools.BedTool(output_bed_path)
    example_fasta = example_bed.sequence(fi=_GENOME_FA)
    return open(example_fasta.seqfn).read().split(">")[1:]


def test_generate_example_bed(tmp_path):
    output_bed_path = _generate_output_bed_path(tmp_path)
    interval2sequence.generate_example_bed(
        _INPUT_BED_PATH, output_bed_path, _SEQUENCE_LENGTH, _CHROM_SIZES
    )
    output_bed = pd.read_csv(output_bed_path, sep="\t", header=None)
    pd.testing.assert_frame_equal(
        output_bed, expected_data.EXPECTED_EXAMPLE_BED
    )


def test_generate_hdf5(tmp_path, _example_fasta):
    output_hdf5_path = _generate_seq_feature_path(tmp_path)
    interval2sequence.generate_hdf5(
        _example_fasta,
        output_hdf5_path,
        _generate_output_bed_path(tmp_path),
        _SEQUENCE_LENGTH,
        bucket_size=5,
    )

    with h5py.File(output_hdf5_path, "r") as h5:
        for path, ids in expected_data.EXPECTED_EXAMPLE_SEQ_FEATURE:
            np.testing.assert_array_equal(h5[path]["ids"][:], ids)


def test_generate_pkl(tmp_path, _example_fasta):
    output_hdf5_path = _generate_seq_feature_path(tmp_path)
    interval2sequence.generate_hdf5(
        _example_fasta,
        output_hdf5_path,
        _generate_output_bed_path(tmp_path),
        _SEQUENCE_LENGTH,
        bucket_size=5,
    )

    output_pkl_path = _generate_metadata_path(tmp_path)
    interval2sequence.generate_pkl(output_hdf5_path, output_pkl_path)

    metadata = pickle.load(open(output_pkl_path, "rb"))
    print(metadata['input'])
    print(expected_data.EXPECTED_EXAMPLE_METADATA["input"])
    assert metadata["input"] == expected_data.EXPECTED_EXAMPLE_METADATA["input"]


def test_main_output_names(tmp_path):
    output_prefix = os.path.join(tmp_path, "prefix")
    interval2sequence.main(
        _GENOME_FA,
        output_prefix,
        _INPUT_BED_PATH,
        _SEQUENCE_LENGTH,
        _CHROM_SIZES,
    )
    assert os.path.isfile(f"{output_prefix}_seq_feature.h5")
    assert os.path.isfile(f"{output_prefix}_metadata.pkl")


def test_one_hot_encoder():
    index = np.array([1, 3, 2, 2, 3, 1, 3, 1, 0, 0])
    shape = (_SEQUENCE_LENGTH, 4)
    expected_output = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(
        interval2sequence.one_hot_encoder(index, shape), expected_output
    )
