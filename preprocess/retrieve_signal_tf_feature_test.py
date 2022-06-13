import os
import subprocess

import numpy as np

import retrieve_signal_tf_feature

from ..testdata import expected_data

_CHROM_SIZES = "../testdata/chrom.sizes"
_SEQUENCE_LENGTH = 10
_BATCH_SIZE = 5
_THRESHOLD = 1e-02
_NUM_FEATURES = 2

_EXPECTED_TF = "tf2"
_EXPECTED_ASSAY_TYPE = "tff1"


def _metadata_pkl(path):
    return expected_data.expected_example_metadata(str(path))


def _create_enrichment_bigbed(path):
    data = expected_data.SIGNAL_BED
    bed_path = os.path.join(path, "tf_feature_signal.bed")
    data.to_csv(bed_path, sep="\t", index=False, header=False)
    # Convert to bigbed.
    bigbed_path = bed_path.replace(".bed", ".bigbed")
    cmd = f"bedToBigBed {bed_path} {_CHROM_SIZES} {bigbed_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigbed_path


def _create_input_data(path):
    metadata_pkl = _metadata_pkl(path)
    signal_bigbed = _create_enrichment_bigbed(path)
    return metadata_pkl, signal_bigbed


def test_retrieve_signal(tmp_path):
    metadata_pkl, signal_bigbed = _create_input_data(tmp_path)

    # Create feature folder.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)

    retrieve_signal_tf_feature.retrieve_signal(
        signal_bigbed,
        metadata_pkl,
        _EXPECTED_TF,
        _EXPECTED_ASSAY_TYPE,
        _BATCH_SIZE,
        _THRESHOLD,
        _SEQUENCE_LENGTH,
        _NUM_FEATURES,
    )

    def _assert_metadata(dset, expected_start, expected_stop):
        assert dset["group_name"] == _EXPECTED_TF
        assert dset["feature_type"] == _EXPECTED_ASSAY_TYPE
        assert dset["threshold"] == _THRESHOLD
        assert dset["start"] == expected_start
        assert dset["stop"] == expected_stop

    file_batch1 = os.path.join(
        output_dir,
        f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.0."
        f"threshold{_THRESHOLD:.0e}.npz",
    )
    batch1 = np.load(file_batch1)
    np.testing.assert_almost_equal(
        batch1["original"], expected_data.EXPECTED_TF_FEATURE_DATA[:5]
    )
    _assert_metadata(batch1, 0, 5)

    file_batch2 = os.path.join(
        output_dir,
        f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.1."
        f"threshold{_THRESHOLD:.0e}.npz",
    )
    batch2 = np.load(file_batch2)
    np.testing.assert_almost_equal(
        batch2["original"], expected_data.EXPECTED_TF_FEATURE_DATA[5:]
    )
    _assert_metadata(batch2, 5, 8)


def test_zscore_signal(tmp_path):
    metadata_pkl, signal_bigbed = _create_input_data(tmp_path)

    # Create folders.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    zscore_file = os.path.join(zscore_dir, "zscore_params.npz")
    os.mkdir(zscore_dir)

    writer = retrieve_signal_tf_feature.retrieve_signal(
        signal_bigbed,
        metadata_pkl,
        _EXPECTED_TF,
        _EXPECTED_ASSAY_TYPE,
        _BATCH_SIZE,
        _THRESHOLD,
        _SEQUENCE_LENGTH,
        _NUM_FEATURES,
    )

    retrieve_signal_tf_feature.zscore_signal(writer, zscore_file, _THRESHOLD, 0)

    file_batch1 = os.path.join(
        output_dir,
        f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.0."
        f"threshold{_THRESHOLD:.0e}.npz",
    )
    batch1 = np.load(file_batch1)
    np.testing.assert_almost_equal(
        batch1["zscore"], expected_data.EXPECTED_ZSCORED_TF_FEATURE_DATA[:5]
    )

    file_batch2 = os.path.join(
        output_dir,
        f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.1."
        f"threshold{_THRESHOLD:.0e}.npz",
    )
    batch2 = np.load(file_batch2)
    np.testing.assert_almost_equal(
        batch2["zscore"], expected_data.EXPECTED_ZSCORED_TF_FEATURE_DATA[5:]
    )


def test_main(tmp_path):
    metadata_pkl, signal_bigbed = _create_input_data(tmp_path)

    # Create folders.
    output_dir = os.path.join(tmp_path, "feature")
    os.mkdir(output_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    os.mkdir(zscore_dir)

    retrieve_signal_tf_feature.main(
        metadata_pkl,
        signal_bigbed,
        _EXPECTED_ASSAY_TYPE,
        _EXPECTED_TF,
        zscore_dir,
        _SEQUENCE_LENGTH,
        _THRESHOLD,
    )

    assert os.path.isfile(
        os.path.join(
            output_dir,
            f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.0.threshold{_THRESHOLD:.0e}"
            ".npz",
        )
    )

    assert os.path.isfile(
        os.path.join(
            output_dir,
            f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.1.threshold{_THRESHOLD:.0e}"
            ".npz",
        )
    )

    assert os.path.isfile(
        os.path.join(
            zscore_dir,
            f"{_EXPECTED_ASSAY_TYPE}.{_EXPECTED_TF}.zscore_params.threshold"
            f"{_THRESHOLD:.0e}.npz",
        )
    )
