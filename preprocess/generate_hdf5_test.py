import os
import pickle
import subprocess

import h5py
import numpy as np
import pybedtools
import pytest

import generate_hdf5
import interval2sequence
import retrieve_signal_ct_feature
import retrieve_signal_tf_feature
import retrieve_signal_target

from ..testdata import expected_data

_BATCH_SIZE = 5
_SEQUENCE_LENGTH = 10
_THRESHOLD = 1e-02
_INPUT_BED_PATH = "../testdata/peaks.bed"
_CHROM_SIZES = "../testdata/chrom.sizes"
_GENOME_FA = "../testdata/genome.fa"
_EMBEDDING_PATH = "../testdata/embedding.pkl"
_CONDITION_METADATA = "../testdata/metadata_target.txt"


def _metadata_pkl(path):
    return expected_data.expected_example_metadata(str(path))


def _create_peak_bigbed(path, filename, data):
    bed_path = os.path.join(path, filename)
    data.to_csv(bed_path, sep="\t", index=False, header=False)
    # Convert to bigbed.
    bigbed_path = bed_path.replace(".bed", ".bigbed")
    cmd = f"bedToBigBed {bed_path} {_CHROM_SIZES} {bigbed_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigbed_path


@pytest.fixture
def _example_fasta(tmp_path):
    output_bed_path = os.path.join(tmp_path, "output.bed")
    interval2sequence.generate_example_bed(
        _INPUT_BED_PATH, output_bed_path, _SEQUENCE_LENGTH, _CHROM_SIZES
    )
    example_bed = pybedtools.BedTool(output_bed_path)
    example_fasta = example_bed.sequence(fi=_GENOME_FA)
    return open(example_fasta.seqfn).read().split(">")[1:]


def _create_signal_bigwig(path):
    data = expected_data.SIGNAL_BEDGRAPH
    bedgraph_path = os.path.join(path, "ct_feature_signal.bedgraph")
    data.to_csv(bedgraph_path, sep="\t", index=False, header=False)
    # Convert to bigwig.
    bigwig_path = bedgraph_path.replace(".bedgraph", ".bigwig")
    cmd = f"bedGraphToBigWig {bedgraph_path} {_CHROM_SIZES} {bigwig_path}"
    subprocess.run(cmd, check=True, shell=True)
    return bigwig_path


@pytest.mark.parametrize(
    "condition_metadata", ["/file/path/not/exist.txt", None]
)
def test_get_conditions_error(condition_metadata):
    with pytest.raises(ValueError):
        generate_hdf5._get_conditions(
            _metadata_pkl, True, condition_metadata, []
        )


@pytest.mark.parametrize(
    "skip_target,condition_metadata,exclude_groups,expected_output",
    [
        (
            False,
            None,
            [],
            (
                ["tf1.ct1", "tf2.ct1", "tf3.ct2"],
                {"tf1", "tf2", "tf3"},
                {"ct1", "ct2"},
            ),
        ),
        (
            False,
            "/path/doesnt/exist.txt",
            [],
            (
                ["tf1.ct1", "tf2.ct1", "tf3.ct2"],
                {"tf1", "tf2", "tf3"},
                {"ct1", "ct2"},
            ),
        ),
        (
            False,
            _CONDITION_METADATA,
            [],
            (
                ["tf1.ct1", "tf2.ct1", "tf3.ct2"],
                {"tf1", "tf2", "tf3"},
                {"ct1", "ct2"},
            ),
        ),
        (
            False,
            None,
            ["tf2.ct1"],
            (["tf1.ct1", "tf3.ct2"], {"tf1", "tf3"}, {"ct1", "ct2"}),
        ),
        (
            False,
            _CONDITION_METADATA,
            ["tf3.ct2"],
            (["tf1.ct1", "tf2.ct1"], {"tf1", "tf2"}, {"ct1"}),
        ),
    ],
)
def test_get_conditions(
    skip_target, condition_metadata, exclude_groups, expected_output
):
    result = generate_hdf5._get_conditions(
        _metadata_pkl("tmp_dir"),
        skip_target,
        condition_metadata,
        exclude_groups,
    )
    assert result == expected_output


def test_main(tmp_path, _example_fasta):
    # Generate data.
    conditions = ["tf1.ct1", "tf2.ct1", "tf3.ct2"]
    cell_types = ["ct1", "ct2"]
    ct_features = ["ctf1", "ctf2"]
    tfs = ["tf1", "tf2", "tf3"]
    tf_features = ["tff1"]

    metadata_pkl = _metadata_pkl(tmp_path)
    target_peak_bed = _create_peak_bigbed(
        tmp_path, "chipseq_peaks.bed", expected_data.PEAKS_BED
    )
    ct_feature_peak_bed = _create_peak_bigbed(
        tmp_path, "ct_feature_peaks.bed", expected_data.PEAKS_BED
    )
    tf_feature_signal_bed = _create_peak_bigbed(
        tmp_path, "tf_feature_signal.bed", expected_data.SIGNAL_BED
    )
    signal_bigwig = _create_signal_bigwig(tmp_path)
    output_bed = os.path.join(tmp_path, "output.bed")
    output_h5 = os.path.join(tmp_path, "output.h5")

    # Create folders.
    target_dir = os.path.join(tmp_path, "target")
    os.mkdir(target_dir)
    feature_dir = os.path.join(tmp_path, "feature")
    os.mkdir(feature_dir)
    zscore_dir = os.path.join(tmp_path, "zscore_params")
    os.mkdir(zscore_dir)

    # Generate seq feature.
    interval2sequence.generate_hdf5(
        _example_fasta,
        output_h5,
        output_bed,
        _SEQUENCE_LENGTH,
        bucket_size=_BATCH_SIZE,
    )

    # Create targets.
    for condition_name in conditions:
        retrieve_signal_target.retrieve_peaks(
            target_peak_bed,
            "output_conserved",
            condition_name,
            metadata_pkl,
            _SEQUENCE_LENGTH,
            0,
        )

    # Create ct features.
    for cell_type in cell_types:
        for assay_type in ct_features:
            retrieve_signal_ct_feature.main(
                metadata_pkl,
                ct_feature_peak_bed,
                signal_bigwig,
                cell_type,
                assay_type,
                zscore_dir,
                _SEQUENCE_LENGTH,
            )

    # Create tf features.
    for tf in tfs:
        for assay_type in tf_features:
            retrieve_signal_tf_feature.main(
                metadata_pkl,
                tf_feature_signal_bed,
                assay_type,
                tf,
                zscore_dir,
                _SEQUENCE_LENGTH,
                _THRESHOLD,
            )

    generate_hdf5.main(
        seq_dict=metadata_pkl,
        ct_feature=ct_features,
        tf_feature=tf_features,
        output_h5=output_h5,
        output_types=["output_conserved"],
        compression=True,
        skip_feature=False,
        skip_target=False,
        condition_metadata=None,
        exclude_groups=[],
        sequence_length=_SEQUENCE_LENGTH,
        normalization="zscore",
        motif_threshold=_THRESHOLD,
        example_feature_start_id=None,
        example_feature_stop_id=None,
        bucket_size=_BATCH_SIZE,
    )

    embedding = pickle.load(open(_EMBEDDING_PATH, "rb"))
    with h5py.File(output_h5, "r") as h5:
        for condition in conditions:
            tf, ct = condition.split(".")
            expected = expected_data.EXPECTED_CHIPSEQ_DATA
            expected[:, -3] = float(embedding["tf"][tf])
            expected[:, -2] = float(embedding["ct"][ct])
            np.testing.assert_array_equal(
                h5[f"output_conserved/{condition}/0"][:], expected[:5]
            )
            np.testing.assert_array_equal(
                h5[f"output_conserved/{condition}/1"][:], expected[5:]
            )

        for i in range(8):
            for cell_type in cell_types:
                for assay_type in ct_features:
                    expected = expected_data.EXPECTED_ZSCORED_CT_FEATURE_DATA[i]
                    expected = np.repeat(expected[:, np.newaxis], 2, axis=1)
                    path = (
                        f"input/{i // _BATCH_SIZE}/{i}/ct_feature/{cell_type}"
                    )
                    np.testing.assert_array_almost_equal(
                        h5[path][:],
                        expected,
                    )
            for tf in tfs:
                for assay_type in tf_features:
                    expected = expected_data.EXPECTED_ZSCORED_TF_FEATURE_DATA[i]
                    path = (
                        f"input/{i // _BATCH_SIZE}/{i}/tf_feature/{tf}"
                    )
                    np.testing.assert_array_almost_equal(
                        h5[path][:],
                        expected,
                    )
