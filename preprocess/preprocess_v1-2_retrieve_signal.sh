#!/bin/bash

USAGE=$'
Usage:

./preprocess_v1-2_retrieve_signal.sh [--metadata METADATA] \
    [--example_pickle EXAMPLE_PICKLE] [--region_bed REGION_BED] \
    [--sequence_length SEQUENCE_LENGTH] [--target_type TARGET_TYPE] \
    [--zscore_dir ZSCORE_DIR] [--motif_threshold MOTIF_THRESHOLD] \
    [--tmp_dir TMP_DIR]

Flags:
--metadata Path to feature or target metadata file. (Default: ../data/metadata/training_example/target.txt)
--example_pickle Path to example.pkl file. (Default: ../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example.pkl)
--region_bed Path to bed file specifying genomic regions from which to generate examples. Optional but specifying REGION_BED file can speed up the processing time when the valid region is small compared to the genome.
--sequence_length Example sequence length. (Default: 1000)
--chrom_sizes Path to the chromosome sizes file. (Default: ../data/annotations/hg19.chrom.sizes)
--genome_fasta Path to decompressed genome fasta file. (Default: ../data/annotations/hg19.fa)
--target_type target label type. Choose between "output_conserved" or "output_relaxed", referring to ENCODE conserved and relaxed peak set respectively. Specify this param when retrieving target labels. (Default: output_conserved)
--zscore_dir Path to directory for saving feature zscore normalization parameters. Specify this param when retrieving feature signals. (Default: ../data/datasets/training_example/training/zscore_params)
--motif_threshold FIMO p-value threshold. Specify this param when retrieving TF motif features. (Default: 1e-2)
--tmp_dir Temporary directory to save intermediate result. (Default: /tmp)
--job_id When running this program sequentially on a set of metadata entries, specify this param correponding to the <job_id>th entry in the metadata file from which to retrieve data.
'

# Specify default values.
METADATA="../data/metadata/training_example/target.txt"
EXAMPLE_PICKLE="../data/datasets/training_example/training/training_minOverlap200_maxUnion600_example_path.pkl"
REGION_BED=""
CHROM_SIZES='../data/annotations/hg19.chrom.sizes'
GENOME_FASTA="../data/annotations/hg19.fa"
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
        --chrom_sizes)
            CHROM_SIZES=$VALUE
            ;;
        --genome_fasta)
            GENOME_FASTA=$VALUE
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
echo "JOB_ID: $SLURM_ARRAY_TASK_ID"

echo "METADATA: $METADATA"
echo "EXAMPLE_PICKLE: $EXAMPLE_PICKLE"
echo "REGION_BED: $REGION_BED"
echo "GENOME_FASTA: $GENOME_FASTA"
echo "CHROM_SIZES: $CHROM_SIZES"
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

function bed_to_bigbed () {
    # Subset the first 6 columns of the bed file and convert to bigbed. 
    # Changing field 5 (score) to 100 because some narrowPeak files have 
    # score that are not in [0, 1000] which causes bedToBigBed to fail.
    # Changing the score value does not matter because bed file is only used
    # to extract location of the peaks.
    awk 'BEGIN {FS="\t";OFS="\t"}; {print $1, $2, $3, $4, 100, $6}' $1 \
    >> ${1/.bed/.std.bed}
    # Sort bed file.
    sort -k1,1 -k2,2n -k3,3n ${1/.bed/.std.bed} > $1
    bigbed_file=${1/.bed/.bigbed}
    bedToBigBed $1 $CHROM_SIZES $bigbed_file
    echo $bigbed_file
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
    bigbed_file="$(bed_to_bigbed $bed_file)"
    python retrieve_signal_target.py $EXAMPLE_PICKLE $bigbed_file $assay_type \
    $tf $cell_type $TARGET_TYPE --sequence_length $SEQUENCE_LENGTH
elif [ -z "${tf}" ]; then
    # Retrieve cell type specific features
    echo "[+] Retrieving cell type features."
    [[ ! -d ${ZSCORE_DIR} ]] && mkdir -p ${ZSCORE_DIR}
    peak_bed="$(download_bed "$sample_name" "$assay_type" "$file1" "$REGION_BED")"
    bw_file=${TMP_DIR}/${sample_name}.${assay_type}.bigWig
    if [ -f $file2 ]; then
        cp $file2 $bw_file
    else
        wget -q -O $bw_file $file2
    fi
    peak_bigbed="$(bed_to_bigbed $peak_bed)"
    python retrieve_signal_ct_feature.py $EXAMPLE_PICKLE $peak_bigbed \
    $bw_file $assay_type $cell_type $ZSCORE_DIR --sequence_length $SEQUENCE_LENGTH
elif [ -z ${cell_type} ]; then
    # Retrieve TF specific features
    echo "[+] Retrieving TF features."
    [[ ! -d ${ZSCORE_DIR} ]] && mkdir -p ${ZSCORE_DIR}
    motif_file="${TMP_DIR}/$(basename "$file2")"
    example_bed="${EXAMPLE_PICKLE/_metadata.pkl/.bed}"
    example_fasta="${TMP_DIR}/tmp.fasta"
    example_fimo="${TMP_DIR}/fimo.tsv"
    cp $file2 $motif_file
    # Get example fasta.
    bedtools getfasta -fi $GENOME_FASTA -bed $example_bed > $example_fasta
    # Run FIMO.
    fimo --parse-genomic-coord --skip-matched-sequence --text --verbosity 1 \
    --thresh $MOTIF_THRESHOLD $motif_file $example_fasta > $example_fimo
    # Convert FIMO result to bigbed.
    awk 'BEGIN {FS="\t";OFS="\t"}; NR>1 {print $3, $4, $5, $8, 100, $6}' \
    $example_fimo > ${example_fimo/.tsv/.bed}
    fimo_bigbed="$(bed_to_bigbed ${example_fimo/.tsv/.bed})"
    # Retrieve fimo result.
    python retrieve_signal_tf_feature.py $EXAMPLE_PICKLE $fimo_bigbed \
    $assay_type $tf $ZSCORE_DIR --sequence_length $SEQUENCE_LENGTH \
    --threshold $MOTIF_THRESHOLD
else
    echo "Error: invalid assay_type ${assay_type}"
    exit 1
fi
