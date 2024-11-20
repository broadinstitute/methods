import argparse
import logging

import hail as hl

from gnomad.utils.filtering import subset_samples_and_variants


logging.basicConfig(
    format="%(asctime)s (%(name)s %(lineno)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("subset")
logger.setLevel(logging.INFO)


def main(args):

    hl.init(
        default_reference=args.import_build,
        log="/import_vcf.log",
        tmp_dir="hdfs:///import_vcf.tmp/",
    )

    logger.info("Importing filters info...")
    filters_ht = hl.import_vcf(
        args.part_two_path,
        reference_genome=args.import_build,
        force_bgz=True,
        find_replace=("nul", "."),
        array_elements_required=False, # NOTE: should probably make a PR out of this !!!
    ).rows()
    logger.info("Importing VCF...")
    # NOTE: always assumes file is bgzipped
    mt = hl.import_vcf(
        args.part_one_path,
        force_bgz=True,
        reference_genome=args.import_build,
        find_replace=("nul", "."),
        array_elements_required=False,
    )
    mt = mt.annotate_rows(filters=filters_ht[mt.row_key].filters)
    logger.info(f"MT count: {mt.count()}")
    hl.export_vcf(mt, args.vcf_out, parallel="header_per_shard")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "This script imports/merges callset filter and genotype information into one MatrixTable and exports to VCF. Currently required only for internal exomes callsets."
    )
    parser.add_argument(
        "--part_one_path",
        help="Path to part one outputs from callset VCFs. This bucket has a nested structure and contains one bucket per chromosome. E.g., gs://fc-2b089793-51e4-46f9-9ed0-c317f8d547eb/RDG_Broad_WES_Internal_v3/part_one_outputs/chr*/RDG_Broad_WES_Internal_chr*.*.hard_filtered_with_genotypes.vcf.gz.",
    )
    parser.add_argument(
        "--part_two_path",
        help="Path to part two outputs from callset (VCF shards containing filter information). E.g., gs://fc-2b089793-51e4-46f9-9ed0-c317f8d547eb/RDG_Broad_WES_Internal_v3/part_two_outputs/RDG_Broad_WES_Internal.filtered.*.vcf.gz.",
    )
    parser.add_argument(
        "--import_build",
        help="Reference build to use when importing VCF",
        default="GRCh38",
    )
    parser.add_argument(
        "--vcf_out",
        help="Output path for VCF. E.g., gs://seqr-datasets/v02/GRCh38/RDG_WES_Broad_Internal/v12/RDG_WES_Broad_Internal.vcf.bgz",
    )
    args = parser.parse_args()

    main(args)
