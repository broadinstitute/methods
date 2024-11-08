import argparse
import logging
from typing import List
import matplotlib.pyplot as plt

import hail as hl

from gnomad.utils.reference_genome import get_reference_genome


logging.basicConfig(
    format="%(asctime)s (%(name)s %(lineno)s): %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("sex_check")
logger.setLevel(logging.INFO)


def get_chr_cov(
    mt: hl.MatrixTable,
    build: str,
    chr_name: str,
    call_rate_threshold: float = 0.25,
    af_threshold: float = 0.01,
    af_field: str = "AF",
) -> hl.Table:
    """
    Calculate mean chromosome coverage.

    :param mt: MatrixTable containing samples with chrY variants
    :param build: Reference used, either GRCh37 or GRCh38
    :param chr_name: Chosen chromosome. Must be either autosome (number only) or sex chromosome (X, Y)
    :param call_rate_threshold: Minimum call rate threshold. Default is 0.25
    :param af_threshold: Minimum allele frequency threshold. Default is 0.01
    :param af_field: Name of field containing allele frequency information. Default is "AF"
    :return: Table annotated with mean coverage of specified chromosome
    """

    logger.warning(
        "This function expects chrY to be at index 23 and chrX to be at index 22 in the reference genome contigs list!"
    )

    if chr_name == "Y":
        chr_place = 23
    elif chr_name == "X":
        chr_place = 22
    else:
        try:
            # Chromosome index in '.contigs' list should be one less than the chromosome number
            chr_place = int(chr_name) - 1
        except ValueError:
            logger.error("chr_name cannot be converted to an integer")
            return -99

    chr_name = hl.get_reference(build).contigs[chr_place]

    logger.info(
        "Filtering to chromosome (and filtering to non-par regions if chromosome is X or Y)..."
    )
    sex_mt = hl.filter_intervals(
        mt,
        [hl.parse_locus_interval(chr_name, reference_genome=build)],
    )

    if chr_place == 22:
        sex_mt = sex_mt.filter_rows(sex_mt.locus.in_x_nonpar())
    if chr_place == 23:
        sex_mt = sex_mt.filter_rows(sex_mt.locus.in_y_nonpar())

    # Filter to common SNVs above defined callrate (should only have one index in the array because the MT only contains biallelic variants)
    # TODO: Make callrate filtering optional before adding code to gnomad_methods
    sex_mt = sex_mt.filter_rows(sex_mt[af_field] > af_threshold)
    sex_mt = hl.variant_qc(sex_mt)
    sex_mt = sex_mt.filter_rows(sex_mt.variant_qc.call_rate > call_rate_threshold)

    logger.info("Returning mean coverage on chromosome %s...", chr_name)
    sex_mt = sex_mt.annotate_cols(**{f"{chr_name}_mean_dp": hl.agg.mean(sex_mt.DP)})
    return sex_mt.cols()


def run_hails_impute_sex(
    mt: hl.MatrixTable,
    build: str,
    out_bucket: str,
    xy_fstat_threshold: float = 0.75,
    xx_fstat_threshold: float = 0.5,
    aaf_threshold: float = 0.05,
) -> hl.Table:
    """
    Impute sex, annotate MatrixTable with results, and output a histogram of fstat values.

    :param MatrixTable mt: MatrixTable containing samples to be ascertained for sex
    :param build: Reference used, either GRCh37 or GRCh38
    :param out_bucket: Bucket name for f-stat histogram
    :param xy_fstat_threshold: F-stat threshold above which a sample will be called XY. Default is 0.75
    :param xx_fstat_threshold: F-stat threshold below which a sample will be called XX. Default is 0.5
    :param aaf_threshold: Alternate allele frequency threshold for `hl.impute_sex`. Default is 0.05
    :return: Table with imputed sex annotations
    """

    logger.warning(
        "User needs to decide on fstat thresholds for XY/XX by looking at fstat plots!"
    )

    # Filter to the X chromosome and impute sex
    mt = hl.filter_intervals(
        mt,
        [
            hl.parse_locus_interval(
                get_reference_genome(mt.locus).x_contigs[0], reference_genome=build
            )
        ],
    )

    logger.info('CUSTOM FILTER TO DIPLOID ONLY')
    mt = mt.filter_cols(
        hl.agg.all(mt.GT.is_diploid() | hl.is_missing(mt.GT))
    )

    sex_ht = hl.impute_sex(
        mt.GT,
        aaf_threshold=aaf_threshold,
        male_threshold=xy_fstat_threshold,
        female_threshold=xx_fstat_threshold,
    )
    mt = mt.annotate_cols(**sex_ht[mt.col_key])
    sex_ht = mt.cols()

    # Plot histogram of fstat values
    df = sex_ht.to_pandas()
    plt.clf()
    plt.hist(df["f_stat"])
    plt.xlabel("Fstat")
    plt.ylabel("Frequency")
    plt.axvline(xy_fstat_threshold, color="blue", linestyle="dashed", linewidth=1)
    plt.axvline(xx_fstat_threshold, color="red", linestyle="dashed", linewidth=1)

    out_path = f"{out_bucket}/fstat_histogram.png"
    with hl.hadoop_open(out_path, "wb") as out:
        plt.savefig(out)

    return sex_ht


def call_sex(
    callset: str,
    temp_path: str,
    out_bucket: str,
    use_y_cov: bool = False,
    add_x_cov: bool = False,
    y_cov_threshold: float = 0.1,
    normalization_contig: str = "20",
    xy_fstat_threshold: float = 0.75,
    xx_fstat_threshold: float = 0.5,
    aaf_threshold: float = 0.05,
    call_rate_threshold: float = 0.25,
    final_annotations: List[str] = [
        "is_female",
        "f_stat",
        "n_called",
        "expected_homs",
        "observed_homs",
        "sex",
    ],
) -> None:
    """
    Call sex for the samples in a given callset and export results file to the desired path.

    :param str callset: String of full MatrixTable path for the callset
    :param temp_path: Path to bucket for temporary data
    :param out_bucket: Bucket name for text file of final table and for f-stat histogram
    :param use_y_cov: Set to True to calculate and use chrY coverage for sex inference. Default is False
    :param add_x_cov: Set to True to calculate chrX coverage. Must be specified with use_y_cov. Default is False
    :param y_cov_threshold: Y coverage threshold used to infer sex aneuploidies.
        XY samples below and XX samples above this threshold will be inferred as having aneuploidies.
        Default is 0.1
    :param normalization_contig: Chosen chromosome for calculating normalized coverage. Default is "20"
    :param xy_fstat_threshold: F-stat threshold above which a sample will be called XY. Default is 0.75
    :param xx_fstat_threshold: F-stat threshold below which a sample will be called XX. Default is 0.5
    :param aaf_threshold: Alternate allele frequency threshold for `hl.impute_sex`. Default is 0.05
    :param call_rate_threshold: Minimum required call rate. Default is 0.25
    :param final_annotations: List of fields to keep in final output text file.
        Default is ["is_female", "f_stat", "n_called", "expected_homs", "observed_homs", "sex"].
    :return: None; function writes sex information to text file
    """

    # Read in matrix table and define output file name prefix
    mt_name = callset.split("/")[-1].strip("\.mt")
    logger.info("Reading matrix table for callset: %s", callset)
    logger.info("Using chromosome Y coverage? %s", use_y_cov)
    mt = hl.read_matrix_table(callset)

    # Filter to SNVs and biallelics
    mt = mt.filter_rows(
        (hl.len(mt.alleles) == 2) & hl.is_snp(mt.alleles[0], mt.alleles[1])
    )

    # Filter to PASS variants only (variants with empty or missing filter set)
    # TODO: Make this an optional argument before moving to gnomad_methods
    # mt = mt.filter_rows(hl.is_missing(mt.filters) | (mt.filters.length() == 0), keep=True)

    logger.info('custom: no filters or just OUTSIDE_OF_TARGETS ')
    mt = mt.filter_rows((hl.len(mt.filters)==0 )| (mt.filters=={'OUTSIDE_OF_TARGETS'}))

    # Infer build:
    build = get_reference_genome(mt.locus).name
    logger.info("Build inferred as %s", build)

    logger.info("Inferring sex...")
    sex_ht = run_hails_impute_sex(
        mt,
        build,
        out_bucket,
        xy_fstat_threshold,
        xx_fstat_threshold,
        aaf_threshold,
    )
    sex_ht = sex_ht.checkpoint(f"{temp_path}/sex_{mt_name}.ht", overwrite=True)

    if use_y_cov:
        final_annotations.extend(
            [
                f"chr{normalization_contig}_mean_dp",
                "chrY_mean_dp",
                "normalized_y_coverage",
            ]
        )
        norm_ht = get_chr_cov(mt, "GRCh38", normalization_contig, call_rate_threshold)
        sex_ht = sex_ht.annotate(**norm_ht[sex_ht.s])
        chry_ht = get_chr_cov(mt, "GRCh38", "Y", call_rate_threshold)
        sex_ht = sex_ht.annotate(**chry_ht[sex_ht.s])
        sex_ht = sex_ht.annotate(
            normalized_y_coverage=hl.or_missing(
                sex_ht[f"chr{normalization_contig}_mean_dp"] > 0,
                sex_ht.chrY_mean_dp / sex_ht[f"chr{normalization_contig}_mean_dp"],
            )
        )
        if add_x_cov:
            final_annotations.extend(["chrX_mean_dp", "normalized_x_coverage"])
            chrx_ht = get_chr_cov(mt, "GRCh38", "X", call_rate_threshold)
            sex_ht = sex_ht.annotate(**chrx_ht[sex_ht.s])
            sex_ht = sex_ht.annotate(
                normalized_x_coverage=hl.or_missing(
                    sex_ht[f"chr{normalization_contig}_mean_dp"] > 0,
                    sex_ht.chrX_mean_dp / sex_ht[f"chr{normalization_contig}_mean_dp"],
                )
            )
        sex_ht = sex_ht.annotate(
            ambiguous_sex=hl.is_missing(sex_ht.is_female),
            sex_aneuploidy=(sex_ht.is_female)
            & hl.is_defined(sex_ht.normalized_y_coverage)
            & (sex_ht.normalized_y_coverage > y_cov_threshold)
            | (~sex_ht.is_female)
            & hl.is_defined(sex_ht.normalized_y_coverage)
            & (sex_ht.normalized_y_coverage < y_cov_threshold),
        )

        sex_expr = (
            hl.case()
            .when(sex_ht.ambiguous_sex, "ambiguous_sex")
            .when(sex_ht.sex_aneuploidy, "sex_aneuploidy")
            .when(sex_ht.is_female, "XX")
            .default("XY")
        )

    else:
        sex_ht = sex_ht.annotate(ambiguous_sex=hl.is_missing(sex_ht.is_female))
        sex_expr = hl.if_else(
            sex_ht.ambiguous_sex,
            "ambiguous_sex",
            hl.if_else(sex_ht.is_female, "XX", "XY"),
        )
    sex_ht = sex_ht.annotate(sex=sex_expr)
    sex_ht = sex_ht.select(*final_annotations)

    out_path = f"{out_bucket}/{mt_name}_sex.txt"
    sex_ht.export(out_path)


def main(args):
    """
    Call `call_sex` to infer sex of samples in input MatrixTable.

    :param args: User's command line inputs
    """
    call_sex(**vars(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script infers the sex of samples"
    )
    parser.add_argument(
        "-i", "--callset", required=True, help="Path to Callset MatrixTable"
    )
    parser.add_argument(
        "-t",
        "--temp-path",
        required=True,
        help="Path to bucket (where to store temporary data)",
    )
    parser.add_argument(
        "-o",
        "--out-bucket",
        required=True,
        help="Bucket name (where to store text file of final table and f-stat histogram)",
    )
    parser.add_argument(
        "-u",
        "--use-y-cov",
        help="Whether to use chromosome Y coverage when inferring sex. Note that Y coverage is required to infer sex aneuploidies",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--add-x-cov",
        help="Whether to also calculate chromosome X mean coverage. Must be specified with use-y-cov",
        action="store_true",
    )
    parser.add_argument(
        "-y",
        "--y-cov-threshold",
        help="Y coverage threshold used to infer sex aneuploidies (XY samples below and XX samples above this threshold will be inferred as having aneuploidies)",
        default=0.1,
    )
    parser.add_argument(
        "-m",
        "--xy-fstat-threshold",
        help="F-stat threshold above which a sample will be called XY. Default is 0.75",
        default=0.75,
    )
    parser.add_argument(
        "-f",
        "--xx-fstat-threshold",
        help="F-stat threshold below which a sample will be called XX. Default is 0.5",
        default=0.50,
    )
    parser.add_argument(
        "-a",
        "--aaf-threshold",
        help="Alternate allele frequency threshold for `hl.impute_sex`. Default is 0.05",
        default=0.05,
    )
    parser.add_argument(
        "-c",
        "--call-rate-threshold",
        help="Minimum variant call rate threshold. Default is 0.25",
        default=0.25,
    )
    # NOTE: this is an integer here because get_chr_cov expects chromosomes to be specified using their numbers
    parser.add_argument(
        "-n",
        "--normalization-contig",
        help="Autosome to use to normalize sex chromosome coverage. Default is chromosome 20",
        default="20",
    )
    parser.add_argument(
        "-l",
        "--final_annotations",
        help="List of columns to keep in final table.",
        default=[
            "is_female",
            "f_stat",
            "n_called",
            "expected_homs",
            "observed_homs",
            "sex",
        ],
    )

    args = parser.parse_args()
    main(args)
