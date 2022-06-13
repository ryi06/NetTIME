#!/bin/bash

USAGE=$'
Example usage:

./preprocess_v1-1_generate_example_sequences.sh [--region_bed REGION_BED] \
	[--output_prefix OUTPUT_PREFIX] [--chip_metadata ChIP_METADATA] \
	[--genome_fasta GENOME_FASTA] [--chrom_sizes CHROM_SIZES] \
	[--embedding_index EMBEDDING_INDEX] [--sequence_length SEQUENCE_LENGTH] \
	[--output_dir OUTPUT_DIR] [--tmp_dir TMP_DIR] [--min_overlap MIN_OVERLAP] \
	[--max_union MAX_UNION] [--set_path_extra_args SET_PATH_EXTRA_ARGS]

Flags:
--region_bed Paths to bed files specifying genomic regions from which to generate examples, specified as "file1,file2". (Default: ../data/metadata/training_example/training_regions.bed)
--output_prefix Output file name prefices, specified as "prefix1,prefix2". One prefix per region_bed file. (Default: training)
--chip_metadata Path to ChIP-seq metadata file. (Default: ../data/metadata/training_example/target.txt)
--genome_fasta Path to decompressed genome fasta file. (Default: ../data/annotations/hg19.fa)
--chrom_sizes Path to chromosome size file. (Default: ../data/annotations/hg19.chrom.sizes)
--embedding_index Path to embedding index file specifying the index for TF and cell type labels. (Default: ../data/embeddings/example.pkl)
--sequence_length Example sequence length. (Default: 1000)
--output_dir Dataset output directory. (Default: ../data/datasets/<CURRENT_DATE>_seqLength<SEQUENCE_LENGTH>)
--tmp_dir Temporary directory to save intermediate result. (Default: /tmp)
--min_overlap The minimum number of overlapping base pairs for two peaks to be merged.
--max_union Two peaks will be merged if their union is smaller than or equal to <max_union> base pairs.
--set_path_extra_args Flags for set_path, specified as "flag1,param1,flag2,param2,flag3". See available optional arguments by running `python set_path.py -h`. (Default: --ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac)
'

# Specify default values.
REGION_BED="../data/metadata/training_example/training_regions.bed"
OUTPUT_PREFIX="training"
ChIP_METADATA="../data/metadata/training_example/target.txt"
GENOME_FASTA="../data/annotations/hg19.fa"
CHROM_SIZES="../data/annotations/hg19.chrom.sizes"
EMBEDDING_INDEX="../data/embeddings/example.pkl"
SEQUENCE_LENGTH=1000
OUTPUT_DIR="../data/datasets/$(date +%Y%m%d)_seqLength${SEQUENCE_LENGTH}"
TMP_DIR="/tmp"
MIN_OVERLAP=200
MAX_UNION=600
SET_PATH_EXTRA_ARGS="--ct_feature,DNase,H3K4me1,H3K4me3,H3K27ac"

while [ "$1" != "" ]; do
	PARAM=$1
	shift
	VALUE=$1
	case $PARAM in
		-h | --help)
			echo "$USAGE"
			exit
			;;
		--region_bed)
			REGION_BED=$VALUE
			;;
		--output_prefix)
			OUTPUT_PREFIX=$VALUE
			;;
		--chip_metadata)
			ChIP_METADATA=$VALUE
			;;
		--genome_fasta)
			GENOME_FASTA=$VALUE
			;;
		--chrom_sizes)
            CHROM_SIZES=$VALUE
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
		--min_overlap)
			MIN_OVERLAP=$VALUE
			;;
		--max_union)
			MAX_UNION=$VALUE
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

echo "========================="
echo "REGION_BED: $REGION_BED"
echo "OUTPUT_PREFIX: $OUTPUT_PREFIX"
echo "ChIP_METADATA: $ChIP_METADATA"
echo "GENOME_FASTA: $GENOME_FASTA"
echo "CHROM_SIZES: $CHROM_SIZES"
echo "EMBEDDING_INDEX: $EMBEDDING_INDEX"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TMP_DIR: $TMP_DIR"
echo "MIN_OVERLAP: $MIN_OVERLAP"
echo "MAX_UNION: $MAX_UNION"
echo "SET_PATH_EXTRA_ARGS: $SET_PATH_EXTRA_ARGS"
echo "========================="

function download_bed () {
	prefix=${TMP_DIR}/${1}.${2}
	if [ -f $3 ]; then
		cp $3 ${prefix}.bed.gz
	else
		wget -q -O ${prefix}.bed.gz $3
	fi
	gzip -dvc ${prefix}.bed.gz > ${prefix}.bed
	awk 'BEGIN {FS="\t";OFS="\t"} {print $1, $2, $3}' ${prefix}.bed \
	>> $4
}

echo "[+] Combining peaks."
combined_prefix=${TMP_DIR}/combined
>${combined_prefix}.bed

# Adding ChIP-seq peaks from multiple conditions into one bed file.
sed 1d $ChIP_METADATA | while read p
do
	entry=`echo ${p} | awk 'BEGIN {FS=" "} {print $3, $5, $6}'`
	read sample_name conserved_loc relaxed_loc <<< $entry
	echo ${sample_name}

	sample_prefix=${TMP_DIR}/${sample_name}
	>${sample_prefix}.bed

	download_bed $sample_name "conserved" $conserved_loc ${sample_prefix}.bed
	download_bed $sample_name "relaxed" $relaxed_loc ${sample_prefix}.bed

	sort -k1,1 -k2,2n ${sample_prefix}.bed > ${sample_prefix}.sorted.bed
	# Remove duplicated peaks
	awk '!seen[$1,$2,$3]++' ${sample_prefix}.sorted.bed > \
	${sample_prefix}.sorted.dedup.bed 
	cat ${sample_prefix}.sorted.dedup.bed  >> ${combined_prefix}.bed
done

echo "[+] Merging peaks."
sort -k1,1 -k2,2n -k3,3n ${combined_prefix}.bed > ${combined_prefix}.sorted.bed

# Remove peaks longer than sequence_length
awk 'BEGIN {FS="\t";OFS="\t"} {l=$3 - $2; if( l < 1000 ) {print $0}}' \
${combined_prefix}.sorted.bed > ${combined_prefix}.max${SEQUENCE_LENGTH}.bed

# Merge peaks with <MIN_OVERLAP> min overlap and <MAX_UNION> max union
python merge_peaks.py ${combined_prefix}.max${SEQUENCE_LENGTH}.bed \
${TMP_DIR}/merged.bed $MIN_OVERLAP $MAX_UNION

# Generating example sequences
IFS=',' read -r -a region_bed_array <<< "$REGION_BED"
IFS=',' read -r -a output_prefix_array <<< "$OUTPUT_PREFIX"
# IFS=',' read -r -a set_path_args_array <<< "$SET_PATH_EXTRA_ARGS"
suffix=minOverlap${MIN_OVERLAP}_maxUnion${MAX_UNION}
for i in "${!region_bed_array[@]}"; do
	region="${region_bed_array[i]}"
	prefix="${output_prefix_array[i]}"
	merged_prefix="${TMP_DIR}/${prefix}_${suffix}"
	example_prefix=${merged_prefix}_example

	echo "[+] Generating example sequences for ${prefix}."
	# Create output directory
	output_dir=${OUTPUT_DIR}/${prefix}

	# Intersect merged peak bed file with region bed file.
	bedtools intersect -wa -a ${TMP_DIR}/merged.bed -b $region -f 1.0 | \
	sort -k1,1 -k2,2n -k3,3n > ${merged_prefix}.bed
	wc -l ${merged_prefix}.bed

	bash preprocess_v1-1-2_generate_custom_examples.sh \
	--condition_metadata $ChIP_METADATA --input_bed ${merged_prefix}.bed \
	--genome_fasta $GENOME_FASTA --chrom_sizes $CHROM_SIZES \
	--embedding_index $EMBEDDING_INDEX --sequence_length $SEQUENCE_LENGTH \
	--output_dir $output_dir --tmp_dir $TMP_DIR \
	--set_path_extra_args $SET_PATH_EXTRA_ARGS

done

echo "Done!"
