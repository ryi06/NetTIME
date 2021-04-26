#!/bin/bash

USAGE=$'
Usage:

./preprocess_v1-1_generate-example_sequences.sh [--input_bed INPUT_BED] \
	[--genome_fasta GENOME_FASTA] [--embedding_index EMBEDDING_INDEX] \
	[--sequence_length SEQUENCE_LENGTH] [--output_dir OUTPUT_DIR] \
	[--tmp_dir TMP_DIR] [--set_path_extra_args SET_PATH_EXTRA_ARGS]

Flags:
--input_bed Path to input bed file from which to generate example sequences.
--condition_metadata Path to a text file specifying the conditions for which to generate data. (Default: ../data/metadata/prediction_example/conditions.txt)
--genome_fasta Path to decompressed genome fasta file. (Default: ../data/annotations/hg19.fa)
--embedding_index Path to embedding index file specifying the index for TF and cell type labels. (Default: ../data/embeddings/pretrained.pkl)
--sequence_length Example sequence length. (Default: 1000)
--output_dir Dataset output directory. (Default: ../data/datasets/<CURRENT_DATE>_seqLength<SEQUENCE_LENGTH>)
--tmp_dir Temporary directory to save intermediate result. (Default: /tmp)
--set_path_extra_args Flags for set_path, specified as "--flag1,param1,--flag2,param2,--flag3". See available optional arguments by running `python set_path.py -h`. (Default: --ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac,--tf_feature,hocomoco)
'

# Specify default values.
INPUT_BED="" # All entries in this file must be the same length and same as <SEQUENCE_LENGTH>
CONDITION_METADATA="../data/metadata/prediction_example/conditions.txt"
GENOME_FASTA="../data/annotations/hg19.fa"
EMBEDDING_INDEX="../data/embeddings/pretrained.pkl" # Set if target label need to be generated
SEQUENCE_LENGTH=1000
OUTPUT_DIR="../data/datasets/$(date +%Y%m%d)_seqLength${SEQUENCE_LENGTH}"
TMP_DIR="/tmp"
SET_PATH_EXTRA_ARGS="--ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac,--tf_feature,hocomoco"

while [ "$1" != "" ]; do
	PARAM=$1
	shift
	VALUE=$1
	case $PARAM in
		-h | --help)
			echo "$USAGE"
			exit
			;;
		--input_bed)
			INPUT_BED=$VALUE
			;;
		--condition_metadata)
			CONDITION_METADATA=$VALUE
			;;
		--genome_fasta)
			GENOME_FASTA=$VALUE
			;;
		--embedding_index)
			EMBEDDING_INDEX=$VALUE
			;;
		--sequence_length)
			SEQUENCE_LENGTH=$VALUE
			;;
		--output_dir)
			OUTPUT_DIR=$VALUE
			;;
		--tmp_dir)
			TMP_DIR=$VALUE
			;;
		--set_path_extra_args)
			SET_PATH_EXTRA_ARGS=$VALUE
			;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            echo "$USAGE"
            exit 1
            ;;
    esac
    shift
done

# Make sure INPUT_BED exists.
if [ ! -f "$INPUT_BED" ]; then
	echo "ERROR: INPUT_BED unspecified or does not exist."
	echo "$USAGE"
	exit 1
fi


echo "========================="
echo "INPUT_BED: $INPUT_BED"
echo "CONDITION_METADATA: $CONDITION_METADATA"
echo "GENOME_FASTA: $GENOME_FASTA"
echo "EMBEDDING_INDEX: $EMBEDDING_INDEX"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TMP_DIR: $TMP_DIR"
echo "SET_PATH_EXTRA_ARGS: $SET_PATH_EXTRA_ARGS"
echo "========================="


echo "[+] Generating example sequences."
# Generate examples .bed and .pkl
filename="$(basename $INPUT_BED)"
example_prefix=${TMP_DIR}/${filename/.bed/_example}
python interval2sequence.py $INPUT_BED $example_prefix \
$GENOME_FASTA --sequence_length $SEQUENCE_LENGTH

# Generate examples .fa
bedtools getfasta -fo ${example_prefix}.fa -fi $GENOME_FASTA -bed \
${example_prefix}.bed

# Create output directory
[[ ! -d ${OUTPUT_DIR} ]] && mkdir -p $OUTPUT_DIR

echo "[+] Setting path"
IFS=',' read -r -a set_path_args_array <<< "$SET_PATH_EXTRA_ARGS"
python set_path.py ${example_prefix}.pkl ${example_prefix}.pkl $OUTPUT_DIR \
$CONDITION_METADATA $EMBEDDING_INDEX "${set_path_args_array[@]}"

echo "[+] Transferring result to disk."
mv ${example_prefix}* ${OUTPUT_DIR}/.
ls -lh $OUTPUT_DIR

echo "Done!"
