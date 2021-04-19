#!/bin/bash

USAGE=$'
Usage:

./preprocess_v1-2_retrieve_signal.sh [--metadata METADATA] \
	[--example_pickle EXAMPLE_PICKLE] [--region_bed REGION_BED] \
	[--sequence_length SEQUENCE_LENGTH] [--target_type TARGET_TYPE] \
	[--zscore_dir ZSCORE_DIR] [--motif_threshold MOTIF_THRESHOLD] \
	[--tmp_dir TMP_DIR]

Flags:
--metadata Path to feature or target metadata file. (Default: ../data/metadata/training_example/out_ENCODE_TF_ChIP.txt)
--example_pickle Path to example.pkl file. (Default: ../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example.pkl)
--region_bed Path to bed file specifying genomic regions from which to generate examples. Optional but specifying REGION_BED file can speed up the processing time when the valid region is small compared to the genome.
--sequence_length Example sequence length. (Default: 1000)
--target_type target label type. Choose between "output_conserved" or "output_relaxed", referring to ENCODE conserved and relaxed peak set respectively. Specify this param when retrieving target labels. (Default: output_conserved)
--zscore_dir Path to directory for saving feature zscore normalization parameters. Specify this param when retrieving feature signals. (Default: ../data/datasets/training_example/training/zscore_params)
--motif_threshold FIMO p-value threshold. Specify this param when retrieving TF motif features. (Default: 1e-2)
--tmp_dir Temporary directory to save intermediate result. (Default: /tmp)
--job_id When running this program sequentially on a set of metadata entries, specify this param correponding to the <job_id>th entry in the metadata file from which to retrieve data.
'

# Specify default values.
METADATA="../data/metadata/training_example/out_ENCODE_TF_ChIP.txt"
EXAMPLE_PICKLE="../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example_path.pkl"
REGION_BED=""
SEQUENCE_LENGTH=1000
TARGET_TYPE="output_conserved"
ZSCORE_DIR="../data/datasets/training_example/training/zscore_params"
MOTIF_THRESHOLD="1e-2"
TMP_DIR='/tmp'

while [ "$1" != "" ]; do
    PARAM=$1
    shift
    VALUE=$1
    case $PARAM in
        -h | --help)
            echo "$USAGE"
            exit
            ;;
        --metadata)
            METADATA=$VALUE
            ;;
        --example_pickle)
            EXAMPLE_PICKLE=$VALUE
            ;;
        --region_bed)
            REGION_BED=$VALUE
            ;;
        --job_id)
            SLURM_ARRAY_TASK_ID=$VALUE
            ;;
        --sequence_length)
            SEQUENCE_LENGTH=$VALUE
            ;;
        --target_type)
            TARGET_TYPE=$VALUE
            ;;
        --zscore_dir)
            ZSCORE_DIR=$VALUE
            ;;
        --motif_threshold)
            MOTIF_THRESHOLD=$VALUE
            ;;
        --tmp_dir)
            TMP_DIR=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            echo "$USAGE"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then 
	echo "ERROR: JOB_ID is unset or set to empty."
	echo "$USAGE"
	exit
fi

echo "========================="
echo "METADATA: $METADATA"
echo "EXAMPLE_PICKLE: $EXAMPLE_PICKLE"
echo "REGION_BED: $REGION_BED"
echo "JOB_ID: $SLURM_ARRAY_TASK_ID"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "TARGET_TYPE: $TARGET_TYPE"
echo "ZSCORE_DIR: $ZSCORE_DIR"
echo "MOTIF_THRESHOLD: $MOTIF_THRESHOLD"
echo "TMP_DIR: $TMP_DIR"
echo "========================="


function download_bed () {
	prefix=${TMP_DIR}/${1}.${2}
	if [ -f $3 ]; then
		cp $3 ${prefix}.bed.gz
	else
		wget -q -O ${prefix}.bed.gz $3
	fi
	gzip -dvc ${prefix}.bed.gz > ${prefix}.bed
	# Intersect with the valid region if REGION_BED is provided
	if [ -f "$4" ]; then
		bedtools intersect -a ${prefix}.bed -b $4 > ${prefix}.intersect.bed
		sort -k1,1 -k2,2n -k3,3n ${prefix}.intersect.bed > ${prefix}.sorted.bed
	else
		sort -k1,1 -k2,2n -k3,3n ${prefix}.bed > ${prefix}.sorted.bed
	fi
	echo ${prefix}.sorted.bed
}


echo "[+] Getting example signals."
entry=`awk -v arrayInd=${SLURM_ARRAY_TASK_ID} \
'BEGIN {FS="\t";OFS=";"};NR==arrayInd {print $1, $2, $3, $5, $6, $7}' $METADATA`
IFS=";" read tf cell_type sample_name file1 file2 assay_type  <<< "$entry"

echo $entry

if [ "$assay_type" == "ChIP" ]; then
	# Retrieve target labels
	echo "[+] Retrieving target labels."
	if [ "$TARGET_TYPE" == "output_conserved" ]; then
		bed_file="$(download_bed "$sample_name" "conserved" "$file1" "$REGION_BED")"
	elif [ "$TARGET_TYPE" == "output_relaxed" ]; then
		bed_file="$(download_bed "$sample_name" "relaxed" "$file2" "$REGION_BED")"
	fi
	python retrieve_signal_target.py $EXAMPLE_PICKLE $bed_file $assay_type \
	$tf $cell_type $TARGET_TYPE --sequence_length $SEQUENCE_LENGTH
elif [ -z "${tf}" ]; then
	# Retrieve cell type specific features
	echo "[+] Retrieving cell type features."
	[[ ! -d ${ZSCORE_DIR} ]] && mkdir -p ${ZSCORE_DIR}
	peak_bed="$(download_bed "$sample_name" "$assay_type" "$file1" "$REGION_BED")"
	bw_file=${TMP_DIR}/${sample_name}.${assay_type}.bigWig
	if [ -f $file2 ]; then
		cp $file2 ${prefix}.bed.gz
	else
		wget -q -O $bw_file $file2
	fi
	python retrieve_signal_ct_feature.py $EXAMPLE_PICKLE $peak_bed $bw_file \
	$assay_type $cell_type $ZSCORE_DIR --sequence_length $SEQUENCE_LENGTH \
	--tmp_dir $TMP_DIR
elif [ -z ${cell_type} ]; then
	# Retrieve TF specific features
	echo "[+] Retrieving TF features."
	[[ ! -d ${ZSCORE_DIR} ]] && mkdir -p ${ZSCORE_DIR}
	motif_file="${TMP_DIR}/$(basename "$file2")"
	cp $file2 $motif_file
	fasta_file=${EXAMPLE_PICKLE/.pkl/.fa}
	python retrieve_signal_tf_feature.py $EXAMPLE_PICKLE $fasta_file \
	$motif_file $assay_type $tf $ZSCORE_DIR --tmp_dir $TMP_DIR \
	--sequence_length $SEQUENCE_LENGTH --threshold $MOTIF_THRESHOLD
else
	echo "Error: invalid assay_type ${assay_type}"
	exit 1
fi