#!/bin/bash

USAGE=$'
Example usage:

./preprocess_v1-3_merge_dataset.sh [--example_pickle EXAMPLE_PICKLE] \
	[--seq_feature_hdf5 SEQ_FEATURE_HDF5] [--output_hdf5 OUTPUT_HDF5] \
	[--generate_hdf5_extra_args GENERATE_HDF5_EXTRA_ARGS]

Flags:
--example_pickle Path to example.pkl file. (Default: ../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example.pkl)
--seq_feature_hdf5 Path to hdf5 file containing sequence feature. (Default: ../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example_seq_feature.h5)
--output_hdf5 Path to output hdf5 file.  (Default: ../data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5)
--generate_hdf5_extra_args Flags for generate_hdf5.py, specified as "flag1,param1,flag2,param2,flag3". See available keyword arguments by running `python generate_hdf5.py -h`. (Default: --ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac,--compression)
'

# Specify default values.
EXAMPLE_PICKLE="../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example.pkl"
SEQ_FEATURE_HDF5="../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example_seq_feature.h5"
OUTPUT_HDF5="../data/datasets/training_example/training_minOverlap200_maxUnion600_example.h5"
GENERATE_HDF5_EXTRA_ARGS="--ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac,--compression"


while [ "$1" != "" ]; do
    PARAM=$1
    shift
    VALUE=$1
    case $PARAM in
        -h | --help)
            echo "$USAGE"
            exit
            ;;
        --example_pickle)
            EXAMPLE_PICKLE=$VALUE
            ;;
        --seq_feature_hdf5)
            SEQ_FEATURE_HDF5=$VALUE
            ;;
        --output_hdf5)
            OUTPUT_HDF5=$VALUE
            ;;
        --generate_hdf5_extra_args)
            GENERATE_HDF5_EXTRA_ARGS=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            echo "$USAGE"
            exit 1
            ;;
    esac
    shift
done

echo "========================="
echo "EXAMPLE_PICKLE: $EXAMPLE_PICKLE"
echo "SEQ_FEATURE_HDF5: $SEQ_FEATURE_HDF5"
echo "OUTPUT_HDF5: $OUTPUT_HDF5"
echo "GENERATE_HDF5_EXTRA_ARGS: $GENERATE_HDF5_EXTRA_ARGS"
echo "========================="

echo "[+] Initialize output h5."
cp $SEQ_FEATURE_HDF5 $OUTPUT_HDF5

echo "[+] Generating hdf5 file."
IFS=',' read -r -a generate_hdf5_args_array <<< "$GENERATE_HDF5_EXTRA_ARGS"
python generate_hdf5.py $EXAMPLE_PICKLE $OUTPUT_HDF5 \
"${generate_hdf5_args_array[@]}"

if [[ ! " ${generate_hdf5_args_array[@]} " =~ "--skip_target" ]]; then
    echo "[+] Computing class weight in target labels."
    count_file=${OUTPUT_HDF5/.h5/_weight.npy}
    python compute_class_weight.py $OUTPUT_HDF5 $count_file
fi