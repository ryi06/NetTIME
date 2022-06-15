import os
import pickle

import pytest

import set_path

from ..testdata import expected_data

_METADATA_TXT = "../testdata/metadata_target.txt"
_EMBEDDING_PATH = "../testdata/embedding.pkl"
_OUTPUT_PKL_NAME = "metadata.pkl"
_BATCH_SIZE = 5

_OUTPUT_CONSERVED = "output_conserved"
_OUTPUT_RELAZED = "output_relaxed"


@pytest.fixture
def _metadata_pkl():
    return expected_data.EXPECTED_EXAMPLE_METADATA


def _generate_output_pkl_path(path):
    return os.path.join(path, _OUTPUT_PKL_NAME)


def test_main_skip_target(tmp_path, _metadata_pkl):
    output_pkl_path = _generate_output_pkl_path(tmp_path)
    set_path.main(
        _METADATA_TXT,
        _EMBEDDING_PATH,
        _metadata_pkl,
        tmp_path,
        output_pkl_path,
        ["ctf1", "ctf2"],
        ["tff1"],
        _BATCH_SIZE,
        True,
    )
    output_dict = pickle.load(open(output_pkl_path, "rb"))
    assert _OUTPUT_CONSERVED not in output_dict.keys()
    assert _OUTPUT_RELAZED not in output_dict.keys()


def test_main(tmp_path, _metadata_pkl):
    output_pkl_path = _generate_output_pkl_path(tmp_path)
    set_path.main(
        _METADATA_TXT,
        _EMBEDDING_PATH,
        _metadata_pkl,
        tmp_path,
        output_pkl_path,
        ["ctf1", "ctf2"],
        ["tff1"],
        _BATCH_SIZE,
        False,
    )
    output_dict = pickle.load(open(output_pkl_path, "rb"))
    expected_output = expected_data.expected_example_metadata(str(tmp_path))
    assert output_dict == expected_output
