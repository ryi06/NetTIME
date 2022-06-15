import os

import pandas as pd

import merge_peaks

_MIN_OVERLAP = 3
_MAX_UNION = 10
_INPUT_BED = pd.DataFrame(
    [
        ["chr1", 1, 3],
        ["chr1", 2, 8],
        ["chr1", 4, 15],
        ["chr1", 11, 18],
        ["chr1", 15, 20],
        ["chr1", 17, 25],
        ["chr1", 30, 42],
        ["chr1", 31, 38],
        ["chr2", 2, 15],
        ["chr2", 4, 10],
        ["chr2", 8, 17],
        ["chr2", 13, 19],
        ["chr2", 13, 20],
        ["chr2", 17, 22],
    ]
)
_INTERVAL_DICT = {
    "chr1": [
        [1, 3],
        [2, 8],
        [4, 15],
        [11, 18],
        [15, 20],
        [17, 25],
        [30, 42],
        [31, 38],
    ],
    "chr2": [[2, 15], [4, 10], [8, 17], [13, 19], [13, 20], [17, 22]],
}
_MERGED_DICT = {
    "chr1": [[1, 3], [2, 8], [4, 15], [11, 20], [17, 25], [30, 42]],
    "chr2": [[2, 15], [8, 17], [13, 22]],
}
_MERGED_BED = pd.DataFrame(
    [
        ["chr1", 1, 3],
        ["chr1", 2, 8],
        ["chr1", 4, 15],
        ["chr1", 11, 20],
        ["chr1", 17, 25],
        ["chr1", 30, 42],
        ["chr2", 2, 15],
        ["chr2", 8, 17],
        ["chr2", 13, 22],
    ]
)


def _create_input_bed(path):
    bed_file = os.path.join(path, "input.bed")
    _INPUT_BED.to_csv(bed_file, sep="\t", index=False, header=False)
    return bed_file


def _output_bed_path(path):
    return os.path.join(path, "output.bed")


def test_generate_intervals(tmp_path):
    bed_file = _create_input_bed(tmp_path)
    interval_dict = merge_peaks.generate_intervals(bed_file)
    assert interval_dict == _INTERVAL_DICT


def test_merge_intervals():
    merged_dict = merge_peaks.merge_intervals(
        _INTERVAL_DICT, _MIN_OVERLAP, _MAX_UNION
    )
    assert merged_dict == _MERGED_DICT


def test_generate_bed(tmp_path):
    output_bed = _output_bed_path(tmp_path)
    merge_peaks.generate_bed(_MERGED_DICT, output_bed)
    data = pd.read_csv(output_bed, sep="\t", header=None)
    pd.testing.assert_frame_equal(data, _MERGED_BED)


def test_main(tmp_path):
    input_bed = _create_input_bed(tmp_path)
    output_bed = _output_bed_path(tmp_path)
    merge_peaks.main(input_bed, _MIN_OVERLAP, _MAX_UNION, output_bed)
    data = pd.read_csv(output_bed, sep="\t", header=None)
    pd.testing.assert_frame_equal(data, _MERGED_BED)
