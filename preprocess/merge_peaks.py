import argparse
import time

import utils


############## FUNCTION ##############
def generate_intervals(input_bed):
    intervals = {}
    with open(input_bed, "r") as bed:
        for line in bed:
            chrom, start, end = line.strip().split("\t")[:3]
            if chrom not in intervals.keys():
                intervals[chrom] = [[int(start), int(end)]]
            else:
                intervals[chrom].append([int(start), int(end)])
    return intervals


def merge_intervals(intervals, minimum_overlap, maximum_union):
    merged_intervals = {}
    for chrom in intervals.keys():
        merged_intervals[chrom] = __merge(
            intervals[chrom], minimum_overlap, maximum_union
        )
    return merged_intervals


# Given a list of intervals where intervals[i] = [start_i, end_i], merge all
# intervals that overlap at least (>=) minimum_overlap positions. All merged
# intervals should have lengths <= maximum_union
# Example:
# intervals: [[1,3], [2,8], [4,15], [11,18], [15,20], [17,25], [30, 42], [31, 38]]
# min_overlap: 3
# max_union: 10
# result: [[1,3], [2,8], [4,15], [11,20], [17,25], [30,42]]
def __merge(intervals, minimum_overlap, maximum_union):
    merged = []
    for interval in intervals:
        if not merged:
            merged.append(interval)
        elif merged[-1][1] >= interval[1]:
            continue
        # If the list of merged intervals is empty or if the currect
        # interval does not overlap with previous for more than
        # minimum_overlap, append it.
        elif merged[-1][1] - minimum_overlap < interval[0]:
            merged.append(interval)
        else:
            # If the merged interval is smaller than
            # maximum_union, merge, else, append.
            if merged[-1][1] < interval[1]:
                if interval[1] - merged[-1][0] <= maximum_union:
                    merged[-1][1] = interval[1]
                else:
                    merged.append(interval)
    return merged


def generate_bed(merged, output_bed):
    with open(output_bed, "w") as bed:
        for chrom in merged.keys():
            for interval in merged[chrom]:
                bed.write("\t".join([chrom] + [str(x) for x in interval]))
                bed.write("\n")


############## MAIN ##############
def main(input_bed, min_overlap, max_union, output_bed):
    start_time = time.time()
    intervals = generate_intervals(input_bed)
    num_intervals = sum([len(intervals[x]) for x in intervals.keys()])
    utils.print_time(
        "{} peaks before merging".format(num_intervals), start_time
    )
    merged = merge_intervals(intervals, min_overlap, max_union)
    num_merged = sum([len(merged[x]) for x in merged.keys()])
    utils.print_time("{} peaks after merging.".format(num_merged), start_time)
    generate_bed(merged, output_bed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Merge overlapping peaks in a bed file.")

    parser.add_argument("input_bed", type=str, help="Path to input bed file.")
    parser.add_argument("output_bed", type=str, help="Path to output bed file.")
    parser.add_argument(
        "min_overlap",
        type=int,
        help="Minimum overlap between intervals allowed for intervals to be "
        "merged.",
    )
    parser.add_argument(
        "max_union",
        type=int,
        help="Maximum interval size allowed for merged intervals.",
    )

    args = parser.parse_args()
    utils.display_args(args, __file__)
    main(args.input_bed, args.min_overlap, args.max_union, args.output_bed)
