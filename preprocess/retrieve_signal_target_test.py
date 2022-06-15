import os
import subprocess

import numpy as np
import pytest

import retrieve_signal_target

from ..testdata import expected_data

_SEQUENCE_LENGTH = 10
_CHROM_SIZES = "../testdata/chrom.sizes"


def _metadata_pkl(path):
    return expected_data.expected_example_metadata(str(path))


def _create_peak_bigbed(path):
    data = expected_data.PEAKS_BED
    bed_path = os.path.join(path, "chipseq_peaks.bed")
    data.to_csv(bed_path, sep="\t", index=False, header=False)
    # Convert to bigbed.
    bigbed_path = bed_path.replace(".bed", ".bigbed")
    cmd = f"bedToBigBed {bed_path} {_CHROM_SIZES} {bigbed_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigbed_path


@pytest.mark.parametrize(
    "target_name,condition_name,file_suffix,expected_output",
    [
        (
            "output_conserved",
            "tf1.ct1",
            "target/tf1.ct1.conserved.0.npz",
            (np.array([0, 0]), 0, 5),
        ),
        (
            "output_conserved",
            "tf2.ct1",
            "target/tf2.ct1.conserved.1.npz",
            (np.array([1, 0]), 5, 8),
        ),
        (
            "output_relaxed",
            "tf2.ct1",
            "target/tf2.ct1.relaxed.0.npz",
            (np.array([1, 0]), 0, 5),
        ),
        (
            "output_relaxed",
            "tf3.ct2",
            "target/tf3.ct2.relaxed.1.npz",
            (np.array([2, 1]), 5, 8),
        ),
    ],
)
def test_file_attribute(
    tmp_path, target_name, condition_name, file_suffix, expected_output
):
    metadata = _metadata_pkl(tmp_path)
    sample_index, start, stop = retrieve_signal_target.__file_attribute(
        metadata[target_name][condition_name][
            os.path.join(tmp_path, file_suffix)
        ]
    )
    expected_index, expected_start, expected_stop = expected_output
    print(f"sample_index: {sample_index} | expected_index: {expected_index}")
    np.testing.assert_array_equal(sample_index, expected_index)
    assert start == expected_start
    assert stop == expected_stop


@pytest.mark.parametrize(
    "dset,expected",
    [
        (np.array([1, 2, 3, 4, 5]), True),
        (np.array([1, 2, np.NaN, 4, 5]), False),
        (None, False),
    ],
)
def test_check_exist(tmp_path, dset, expected):
    filename = os.path.join(tmp_path, "filename.npz")
    if dset is not None:
        np.savez_compressed(filename, data=dset)
    assert retrieve_signal_target.__check_exist(filename) == expected


def test_retrieve_peaks(tmp_path):
    peak_bed = _create_peak_bigbed(tmp_path)
    metadata_pkl = _metadata_pkl(tmp_path)
    expected_peak_kwd = "output_conserved"
    expected_group_name = "tf3.ct2"

    # Create target folder.
    output_dir = os.path.join(tmp_path, "target")
    os.mkdir(output_dir)

    retrieve_signal_target.retrieve_peaks(
        peak_bed,
        expected_peak_kwd,
        expected_group_name,
        metadata_pkl,
        _SEQUENCE_LENGTH,
        0,
    )

    def _assert_metadata(dset, expected_start, expected_stop):
        assert dset["group_name"].item() == expected_group_name
        assert dset["peak_type"].item() == expected_peak_kwd
        assert dset["start"] == expected_start
        assert dset["stop"] == expected_stop

    file_batch1 = os.path.join(
        output_dir, f"{expected_group_name}.conserved.0.npz"
    )
    batch1 = np.load(file_batch1)
    np.testing.assert_array_equal(
        batch1["data"], expected_data.EXPECTED_CHIPSEQ_DATA[:5]
    )
    _assert_metadata(batch1, 0, 5)

    file_batch2 = os.path.join(
        output_dir, f"{expected_group_name}.conserved.1.npz"
    )
    batch2 = np.load(file_batch2)
    np.testing.assert_array_equal(
        batch2["data"], expected_data.EXPECTED_CHIPSEQ_DATA[5:]
    )
    _assert_metadata(batch2, 5, 8)
