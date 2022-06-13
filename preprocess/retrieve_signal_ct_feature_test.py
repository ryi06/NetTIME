import os
import subprocess

import numpy as np
import pytest

import retrieve_signal_ct_feature

from ..testdata import expected_data

_CHROM_SIZES = "../testdata/chrom.sizes"
_SEQUENCE_LENGTH = 10
_BATCH_SIZE = 5

_EXPECTED_CELL_TYPE = "ct2"
_EXPECTED_ASSAY_TYPE = "ctf1"


def _metadata_pkl(path):
    return expected_data.expected_example_metadata(str(path))


def _create_peak_bigbed(path):
    data = expected_data.PEAKS_BED
    bed_path = os.path.join(path, "ct_feature_peaks.bed")
    data.to_csv(bed_path, sep="\t", index=False, header=False)
    # Convert to bigbed.
    bigbed_path = bed_path.replace(".bed", ".bigbed")
    cmd = f"bedToBigBed {bed_path} {_CHROM_SIZES} {bigbed_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigbed_path


def _create_signal_bigwig(path):
    data = expected_data.SIGNAL_BEDGRAPH
    bedgraph_path = os.path.join(path, "ct_feature_signal.bedgraph")
    data.to_csv(bedgraph_path, sep="\t", index=False, header=False)
    # Convert to bigwig.
    bigwig_path = bedgraph_path.replace(".bedgraph", ".bigwig")
    cmd = f"bedGraphToBigWig {bedgraph_path} {_CHROM_SIZES} {bigwig_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigwig_path


def _create_input_data(path):
    peak_bigbed = _create_peak_bigbed(path)
    metadata_pkl = _metadata_pkl(path)
    signal_bigwig = _create_signal_bigwig(path)
    return peak_bigbed, metadata_pkl, signal_bigwig


def test_retrieve_signal(tmp_path):
    peak_bigbed, metadata_pkl, signal_bigwig = _create_input_data(tmp_path)

    # Create feature folder.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)

    retrieve_signal_ct_feature.retrieve_signal(
        peak_bigbed,
        signal_bigwig,
        metadata_pkl,
        _EXPECTED_CELL_TYPE,
        _EXPECTED_ASSAY_TYPE,
        _BATCH_SIZE,
        _SEQUENCE_LENGTH,
    )

    def _assert_metadata(dset, expected_start, expected_stop):
        assert dset["group_name"] == _EXPECTED_CELL_TYPE
        assert dset["feature_type"] == _EXPECTED_ASSAY_TYPE
        assert dset["start"] == expected_start
        assert dset["stop"] == expected_stop

    file_batch1 = os.path.join(
        output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.0.npz"
    )
    batch1 = np.load(file_batch1)
    np.testing.assert_array_equal(
        batch1["original"], expected_data.EXPECTED_CT_FEATURE_DATA[:5]
    )
    _assert_metadata(batch1, 0, 5)

    file_batch2 = os.path.join(
        output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.1.npz"
    )
    batch2 = np.load(file_batch2)
    np.testing.assert_array_equal(
        batch2["original"], expected_data.EXPECTED_CT_FEATURE_DATA[5:]
    )
    _assert_metadata(batch2, 5, 8)


def test_get_zscore_params(tmp_path):
    peak_bigbed, metadata_pkl, signal_bigwig = _create_input_data(tmp_path)

    # Create folders.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    zscore_file = os.path.join(zscore_dir, "zscore_params.npz")
    os.mkdir(zscore_dir)

    writer = retrieve_signal_ct_feature.retrieve_signal(
        peak_bigbed,
        signal_bigwig,
        metadata_pkl,
        _EXPECTED_CELL_TYPE,
        "ctf1",
        _BATCH_SIZE,
        _SEQUENCE_LENGTH,
    )

    retrieve_signal_ct_feature.get_zscore_params(writer, zscore_file, 0)
    data = np.load(zscore_file)

    assert data["mu"] == pytest.approx(expected_data.EXPECTED_CT_FEATURE_MU)
    assert data["std"] == pytest.approx(expected_data.EXPECTED_CT_FEATURE_STD)


def test_zscore_signal(tmp_path):
    peak_bigbed, metadata_pkl, signal_bigwig = _create_input_data(tmp_path)

    # Create folders.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    zscore_file = os.path.join(zscore_dir, "zscore_params.npz")
    os.mkdir(zscore_dir)

    writer = retrieve_signal_ct_feature.retrieve_signal(
        peak_bigbed,
        signal_bigwig,
        metadata_pkl,
        _EXPECTED_CELL_TYPE,
        _EXPECTED_ASSAY_TYPE,
        _BATCH_SIZE,
        _SEQUENCE_LENGTH,
    )

    retrieve_signal_ct_feature.zscore_signal(writer, zscore_file, 0)

    file_batch1 = os.path.join(
        output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.0.npz"
    )
    batch1 = np.load(file_batch1)
    np.testing.assert_array_almost_equal(
        batch1["zscore"], expected_data.EXPECTED_ZSCORED_CT_FEATURE_DATA[:5]
    )

    file_batch2 = os.path.join(
        output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.1.npz"
    )
    batch2 = np.load(file_batch2)
    np.testing.assert_array_almost_equal(
        batch2["zscore"], expected_data.EXPECTED_ZSCORED_CT_FEATURE_DATA[5:]
    )


def test_main(tmp_path):
    peak_bigbed, metadata_pkl, signal_bigwig = _create_input_data(tmp_path)

    # Create folders.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    os.mkdir(zscore_dir)

    retrieve_signal_ct_feature.main(
        metadata_pkl,
        peak_bigbed,
        signal_bigwig,
        _EXPECTED_CELL_TYPE,
        _EXPECTED_ASSAY_TYPE,
        zscore_dir,
        _SEQUENCE_LENGTH,
    )

    assert os.path.isfile(
        os.path.join(
            output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.0.npz"
        )
    )
    assert os.path.isfile(
        os.path.join(
            output_dir, f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.1.npz"
        )
    )
    assert os.path.isfile(
        os.path.join(
            zscore_dir,
            f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_CELL_TYPE}.zscore_params.npz",
        )
    )
