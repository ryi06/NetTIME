import os
import subprocess

import numpy as np
import pybedtools
import pytest

import compute_class_weight
import generate_hdf5
import interval2sequence
import retrieve_signal_target

from ..testdata import expected_data

_BATCH_SIZE = 5
_SEQUENCE_LENGTH = 10
_INPUT_BED_PATH = "../testdata/peaks.bed"
_CHROM_SIZES = "../testdata/chrom.sizes"
_GENOME_FA = "../testdata/genome.fa"


@pytest.fixture
def _example_fasta(tmp_path):
    output_bed_path = os.path.join(tmp_path, "output.bed")
    interval2sequence.generate_example_bed(
        _INPUT_BED_PATH, output_bed_path, _SEQUENCE_LENGTH, _CHROM_SIZES
    )
    example_bed = pybedtools.BedTool(output_bed_path)
    example_fasta = example_bed.sequence(fi=_GENOME_FA)
    return open(example_fasta.seqfn).read().split(">")[1:]


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


def _create_target_hdf5(path, example_fasta):
    conditions = ["tf1.ct1", "tf2.ct1", "tf3.ct2"]
    metadata_pkl = _metadata_pkl(path)
    target_peak_bed = _create_peak_bigbed(path)

    output_bed = os.path.join(path, "output.bed")
    output_h5 = os.path.join(path, "output.h5")

    target_dir = os.path.join(path, "target")
    os.mkdir(target_dir)

    # Generate seq feature.
    interval2sequence.generate_hdf5(
        example_fasta,
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

    generate_hdf5.main(
        seq_dict=metadata_pkl,
        ct_feature=[],
        tf_feature=[],
        output_h5=output_h5,
        output_types=["output_conserved"],
        compression=True,
        skip_feature=True,
        skip_target=False,
        condition_metadata=None,
        exclude_groups=[],
        sequence_length=_SEQUENCE_LENGTH,
        normalization="zscore",
        motif_threshold=None,
        example_feature_start_id=None,
        example_feature_stop_id=None,
        bucket_size=_BATCH_SIZE,
    )

    return output_h5


def test_main(tmp_path, _example_fasta):
    output_h5 = _create_target_hdf5(tmp_path, _example_fasta)
    output_npy = os.path.join(tmp_path, "output.npy")
    compute_class_weight.main(output_h5, output_npy)

    data = np.load(output_npy)
    expected = expected_data.EXPECTED_CHIPSEQ_DATA[:, :-3]
    expected = np.unique(expected, return_counts=True)[1]
    expected = expected[::-1] / expected.sum()
    np.testing.assert_array_almost_equal(data, expected)

