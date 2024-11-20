import logging
import argparse
import pickle

import hail as hl
from gnomad.sample_qc.platform import (
    compute_callrate_mt,
    run_platform_pca,
    assign_platform_from_pcs,
)
from gnomad.sample_qc.filtering import compute_stratified_metrics_filter
from gnomad.utils.filtering import filter_to_autosomes
from gnomad.sample_qc.pipeline import filter_rows_for_qc
from gnomad.sample_qc.ancestry import pc_project, assign_population_pcs
from gnomad.utils import slack

from resources.resources_seqr_qc import (
    mt_path,
    missing_metrics_path,
    rdg_gnomad_pop_pca_loadings_path,
    rdg_gnomad_rf_model_path,
    remap_path,
    sample_qc_ht_path,
    sample_qc_tsv_path,
    seq_metrics_path,
    val_coding_ht_path,
    val_noncoding_ht_path,
    VCFDataTypeError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("seqr_sample_qc")
logger.setLevel(logging.INFO)


def validate_mt(mt: hl.MatrixTable, build: int, data_type: str, threshold=0.3):
    """
    Validate the mt by checking against a list of common coding and non-coding variants given its
    genome version. This validates genome_version, variants, and the reported sample type.

    :param mt: mt to validate
    :param build: reference build
    :param data_type: WGS or WES
    :param threshold: Threshold percentage of variants matching validation tables
    :return: True or Exception
    """

    def data_type_stats(mt, build, threshold):
        """
        Calculate stats for data type by checking against a list of common coding and non-coding variants.
        If the match for each respective type is over the threshold, we return a match.

        :param mt: Matrix Table to check
        :param build: reference build
        """
        stats = {}
        types_to_ht_path = {
            "noncoding": val_noncoding_ht_path(build),
            "coding": val_coding_ht_path(build),
        }
        for variant_type, ht_path in types_to_ht_path.items():
            ht = hl.read_table(ht_path)
            stats[variant_type] = ht_stats = {
                "matched_count": mt.semi_join_rows(ht).count_rows(),
                "total_count": ht.count(),
            }

            ht_stats["match"] = (
                ht_stats["matched_count"] / ht_stats["total_count"]
            ) >= threshold
        return stats

    data_type_stats = data_type_stats(mt, build, threshold)

    for name, stat in data_type_stats.items():
        logger.info(
            "Table contains %i out of %i common %s variants.",
            stat["matched_count"],
            stat["total_count"],
            name,
        )

    has_coding = data_type_stats["coding"]["match"]
    has_noncoding = data_type_stats["noncoding"]["match"]

    if not has_coding and not has_noncoding:
        raise VCFDataTypeError(
            f"Genome version validation error: dataset specified as GRCh{build} but doesn't contain "
            f"the expected number of common GRCh{build} variants"
        )
    elif has_noncoding and not has_coding:
        raise VCFDataTypeError(
            "Sample type validation error: Dataset contains noncoding variants but is missing common coding "
            f"variants for GRCh{build}. Please verify that the dataset contains coding variants."
        )
    elif has_coding and not has_noncoding:
        if data_type != "WES":
            raise VCFDataTypeError(
                f"Sample type validation error: dataset sample-type is specified as {data_type} but appears to be "
                "WES because it contains many common coding variants and lacks many non-coding variants"
            )
    elif has_noncoding and has_coding:
        if data_type != "WGS":
            raise VCFDataTypeError(
                f"Sample type validation error: dataset sample-type is specified as {data_type} but appears to be "
                "WGS because it contains many common non-coding variants"
            )


def apply_filter_flags_expr(
    mt: hl.MatrixTable, data_type: str, metric_thresholds: dict
) -> hl.expr.SetExpression:
    """
    Annotates table with flags for elevated contamination and chimera as well as low coverage and call rate
    :param Table mt: input MatrixTable
    :param str data_type: 'WES' or 'WGS' for selecting coverage threshold
    :param dict metric_thresholds: dictionary where key is metric and value is threshold value
    :return: Set of sequencing metric flags
    :rtype: SetExpression
    """
    flags = {
        "callrate": mt.filtered_callrate < metric_thresholds["callrate_thres"],
        "contamination": mt.PCT_CONTAMINATION
        > metric_thresholds[
            "contam_thres"
        ],  # TODO revisit current thresholds and rename once have to kristen's script output
        "chimera": mt.AL_PCT_CHIMERAS > metric_thresholds["chimera_thres"],
    }
    if data_type == "WES":
        flags.update(
            {
                "coverage": mt.HS_PCT_TARGET_BASES_20X
                < metric_thresholds["wes_cov_thres"]
            }
        )
    else:
        flags.update(
            {"coverage": mt.WGS_MEAN_COVERAGE < metric_thresholds["wgs_cov_thres"]}
        )

    return hl.set(
        hl.filter(
            lambda x: hl.is_defined(x),
            [hl.or_missing(filter_expr, name) for name, filter_expr in flags.items()],
        )
    )


def get_all_sample_metadata(
    mt: hl.MatrixTable, build: int, data_type: str, data_source: str, version: int
) -> hl.Table:
    """
    Annotate MatrixTable with all current metadata: sample sequencing metrics, sample ID mapping,
    and callrate for bi-allelic, high-callrate common SNPs.
    :param MatrixTable mt: VCF converted to a MatrixTable
    :param int build: build for write, 37 or 38
    :param str data_type: WGS or WES for write path and flagging metrics
    :param str data_source: internal or external for write path
    :param int version: Int for write path
    :return: Table with seq metrics and mapping
    :rtype: Table
    """
    logger.info("Importing and annotating with sequencing metrics...")
    meta_ht = hl.import_table(
        seq_metrics_path(build, data_type, data_source, version), impute=True
    ).key_by("SAMPLE")

    logger.info("Importing and annotating seqr ID names...")
    remap_ht = hl.import_table(
        remap_path(build, data_type, data_source, version), impute=True
    ).key_by("s")
    meta_ht = meta_ht.annotate(**remap_ht[meta_ht.key])
    meta_ht = meta_ht.annotate(
        seqr_id=hl.if_else(
            hl.is_missing(meta_ht.seqr_id), meta_ht.SAMPLE, meta_ht.seqr_id
        )
    )

    logger.info(
        "Filtering to bi-allelic, high-callrate, common SNPs to calculate callrate..."
    )
    mt = filter_rows_for_qc(
        mt,
        min_af=0.001,
        min_callrate=0.99,
        bi_allelic_only=True,
        snv_only=True,
        apply_hard_filters=False,
        min_inbreeding_coeff_threshold=None,
        min_hardy_weinberg_threshold=None,
    )
    callrate_ht = mt.select_cols(
        filtered_callrate=hl.agg.fraction(hl.is_defined(mt.GT))
    ).cols()
    meta_ht = meta_ht.annotate(**callrate_ht[meta_ht.key])
    return meta_ht


def run_platform_imputation(
    mt: hl.MatrixTable,
    plat_min_cluster_size: int,
    plat_min_sample_size: int,
    plat_assignment_pcs: int,
) -> hl.Table:
    """
    Run PCA using sample callrate across Broad's evaluation intervals and create Hail Table
    with platform PCs and assigned platform.
    :param MatrixTable mt: QC MatrixTable
    :param plat_min_cluster_size: min cluster size for HBDscan clustering
    :param plat_min_sample_size: min sample size for HBDscan clustering
    :param plat_assignment_pcs: Number PCs used for HBDscan clustering
    :return: Table with platform PCs and assigned platform
    :rtype: Table
    """
    intervals = hl.import_locus_intervals(
        "gs://gcp-public-data--broad-references/hg38/v0/exome_evaluation_regions.v1.interval_list"
    )
    callrate_mt = compute_callrate_mt(mt, intervals)
    eigenvalues, scores_ht, ignore = run_platform_pca(callrate_mt)
    plat_ht = assign_platform_from_pcs(
        scores_ht,
        hdbscan_min_cluster_size=plat_min_cluster_size,
        hdbscan_min_samples=plat_min_sample_size,
    )
    plat_pcs = {
        f"plat_PC{i+1}": scores_ht.scores[i]
        for i in list(range(0, plat_assignment_pcs))
    }
    scores_ht = scores_ht.annotate(**plat_pcs).drop("scores")
    plat_ht = plat_ht.annotate(**scores_ht[plat_ht.key])
    return plat_ht


def run_population_pca(mt: hl.MatrixTable, build: int, num_pcs=6) -> hl.Table:
    """
    Projects samples onto pre-computed gnomAD and rare disease sample principal components using PCA loadings.  A
    random forest classifier assigns gnomAD and rare disease sample population labels
    :param mt: QC MatrixTable
    :param build: 37 or 38 for write path
    :param pop_fit_path: fit from a previously trained random forest model (i.e., the output from a previous RandomForestClassifier() call)
    :param num_pcs: Number of PCs to use in model
    :return: Table annotated with assigned RDG and gnomAD population and PCs
    :rtype: Table
    """
    loadings = hl.read_table(rdg_gnomad_pop_pca_loadings_path(build))
    model_path = rdg_gnomad_rf_model_path()
    mt = mt.select_entries("GT")
    scores = pc_project(mt, loadings)
    scores = scores.annotate(known_pop="Unknown").key_by("s")

    logger.info("Unpacking RF model")
    fit = None
    with hl.hadoop_open(model_path, "rb") as f:
        fit = pickle.load(f)

    pop_pca_ht, ignore = assign_population_pcs(
        scores, pc_cols=[i + 1 for i in range(num_pcs)], output_col="qc_pop", fit=fit
    )
    pop_pca_ht = pop_pca_ht.key_by("s")
    pop_pcs = {f"pop_PC{i+1}": scores.scores[i] for i in range(num_pcs)}
    scores = scores.annotate(**pop_pcs).drop("scores", "known_pop")
    pop_pca_ht = pop_pca_ht.annotate(**scores[pop_pca_ht.key])
    return pop_pca_ht


def run_hail_sample_qc(mt: hl.MatrixTable, data_type: str) -> hl.MatrixTable:
    """
    Runs Hail's built in sample qc function on the MatrixTable. Splits the MatrixTable in order to calculate inbreeding
    coefficient and annotates the result back onto original MatrixTable. Applies flags by population and platform groups.
    :param MatrixTable mt: QC MatrixTable
    :param str data_type: WGS or WES for write path
    :return: MatrixTable annotated with hails sample qc metrics as well as pop and platform outliers
    :rtype: MatrixTable
    """
    mt = mt.select_entries(mt.GT)
    mt = filter_to_autosomes(mt)
    mt = hl.split_multi_hts(mt)
    mt = hl.sample_qc(mt)
    mt = mt.annotate_cols(
        sample_qc=mt.sample_qc.annotate(
            f_inbreeding=hl.agg.inbreeding(mt.GT, mt.info.AF[0])
        )
    )
    mt = mt.annotate_cols(idx=mt.qc_pop + "_" + hl.str(mt.qc_platform))

    sample_qc = [
        "n_snp",
        "r_ti_tv",
        "r_insertion_deletion",
        "n_insertion",
        "n_deletion",
        "r_het_hom_var",
    ]
    if data_type == "WGS":
        sample_qc = sample_qc + ["call_rate"]

    strat_ht = mt.cols()
    qc_metrics = {metric: strat_ht.sample_qc[metric] for metric in sample_qc}
    strata = {"qc_pop": strat_ht.qc_pop, "qc_platform": strat_ht.qc_platform}

    metric_ht = compute_stratified_metrics_filter(strat_ht, qc_metrics, strata)
    checkpoint_pass = metric_ht.aggregate(
        hl.agg.count_where(hl.len(metric_ht.qc_metrics_filters) == 0)
    )
    logger.info(
        "%i samples found passing pop/platform-specific filtering", checkpoint_pass
    )
    checkpoint_fail = metric_ht.aggregate(
        hl.agg.count_where(hl.len(metric_ht.qc_metrics_filters) != 0)
    )
    logger.info(
        "%i samples found failing pop/platform-specific filtering", checkpoint_fail
    )
    metric_ht = metric_ht.annotate(sample_qc=mt.cols()[metric_ht.key].sample_qc)
    return metric_ht


def main(args):

    hl.init(log="/seqr_sample_qc.log")
    hl._set_flags(
        no_whole_stage_codegen="1"
    )  # Flag needed for hail 0.2.93, may be able to remove in future release.
    logger.info("Beginning seqr sample QC pipeline...")

    data_type = args.data_type
    build = args.build
    data_source = args.data_source
    version = args.callset_version
    is_test = args.is_test
    overwrite = args.overwrite

    logger.info("Importing callset...")
    if not args.skip_write_mt:
        logger.info("Converting vcf to MatrixTable...")
        mt = hl.import_vcf(
            args.vcf_path,
            force_bgz=True,
            reference_genome=f"GRCh{build}",
            min_partitions=4,
        ).write(
            mt_path(build, data_type, data_source, version, is_test), overwrite=True
        )
    mt = hl.read_matrix_table(mt_path(build, data_type, data_source, version, is_test))
    mt = mt.annotate_entries(
        GT=hl.case()
        .when(mt.GT.is_diploid(), hl.call(mt.GT[0], mt.GT[1], phased=False))
        .when(mt.GT.is_haploid(), hl.call(mt.GT[0], phased=False))
        .default(hl.missing(hl.tcall))
    )
    if not args.skip_validate_mt:
        logger.info("Validating data type...")
        validate_mt(mt, build, data_type)

    if is_test:
        logger.info("Creating test mt...")
        mt = hl.filter_intervals(
            mt,
            [
                hl.parse_locus_interval(
                    hl.if_else(build == "37", "20", "chr20"),
                    reference_genome=f"GRCh{build}",
                )
            ],
        ).persist()

    logger.info("Annotating with sequencing metrics and filtered callrate...")
    meta_ht = get_all_sample_metadata(mt, build, data_type, data_source, version)
    mt = mt.annotate_cols(**meta_ht[mt.col_key], data_type=data_type)

    logger.info("Annotating with sample metric filter flags...")
    metric_thresholds = {
        "callrate_thres": args.callrate_low_threshold,
        "contam_thres": args.contam_up_threshold,
        "chimera_thres": args.chimera_up_threshold,
        "wes_cov_thres": args.wes_coverage_low_threshold,
        "wgs_cov_thres": args.wgs_coverage_low_threshold,
    }
    mt = mt.annotate_cols(
        filter_flags=apply_filter_flags_expr(mt, data_type, metric_thresholds)
    )

    logger.info("Assign platform or product")
    if data_type == "WES" and data_source == "External":
        logger.info("Running platform imputation...")
        plat_ht = run_platform_imputation(
            mt,
            args.plat_min_cluster_size,
            args.plat_min_sample_size,
            args.plat_assignment_pcs,
        )
        mt = mt.annotate_cols(**plat_ht[mt.col_key])
    elif data_source == "Internal":
        logger.info("Assigning platform from product in metadata...")
        mt = mt.annotate_cols(
            qc_platform=hl.if_else(hl.is_defined(mt.PRODUCT), mt.PRODUCT, "Unknown")
        )

        missing_metrics = mt.filter_cols(hl.is_defined(mt.PRODUCT), keep=False)
        missing_metrics.cols().select().export(
            missing_metrics_path(build, data_type, data_source, version)
        )  #  TODO Add logging step that prints unexpected missing samples
    else:
        mt = mt.annotate_cols(qc_platform="Unknown")

    logger.info("Projecting gnomAD population PCs...")
    pop_ht = run_population_pca(mt, build)
    mt = mt.annotate_cols(**pop_ht[mt.col_key])

    logger.info("Running Hail's sample qc...")
    hail_metric_ht = run_hail_sample_qc(mt, data_type)
    mt = mt.annotate_cols(**hail_metric_ht[mt.col_key])

    logger.info("Exporting sample QC tables...")
    ht = mt.cols()
    ht = ht.checkpoint(
        sample_qc_ht_path(build, data_type, data_source, version, is_test), overwrite
    )
    ht.flatten().export(sample_qc_tsv_path(build, data_type, data_source, version))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vcf-path", help="Path to VCF", required=True,
    )
    parser.add_argument(
        "--data-type",
        help="Sequencing data type (WES or WGS)",
        choices=["WES", "WGS"],
        required=True,
    )
    parser.add_argument(
        "-b",
        "--build",
        help="Reference build, 37 or 38",
        type=int,
        choices=[37, 38],
        required=True,
        default=38,
    )
    parser.add_argument(
        "-v",
        "--callset-version",
        help="Version of callset vcf",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--data-source",
        help="Data source (Internal or External)",
        choices=["Internal", "External"],
        required=True,
        default="Internal",
    )
    parser.add_argument(
        "--is-test",
        help="To run a test of the pipeline using test files and directories",
        action="store_true",
    )
    parser.add_argument(
        "--callrate-low-threshold",
        help="Lower threshold at which to flag samples for low callrate",
        default=0.85,
    )
    parser.add_argument(
        "--contam-up-threshold",
        help="Upper threshold at which to flag samples for elevated contamination",
        default=5,
    )
    parser.add_argument(
        "--chimera-up-threshold",
        help="Upper threshold at which to flag samples for elevated chimera",
        default=5,
    )
    parser.add_argument(
        "--wes-coverage-low-threshold",
        help="Lower threshold at which to flag exome samples for low coverage",
        default=85,
    )
    parser.add_argument(
        "--wgs-coverage-low-threshold",
        help="Lower threshold at which to flag genome samples for low coverage",
        default=30,
    )
    parser.add_argument(
        "--plat-min-cluster-size",
        help="Minimum cluster size for platform pca labeling",
        default=40,
    )
    parser.add_argument(
        "--plat-min-sample-size",
        help="Minimum sample size for platform pca labeling",
        default=40,
    )
    parser.add_argument(
        "--plat-assignment-pcs",
        help="Number of principal components to use in platform assignment clustering",
        default=6,
    )
    parser.add_argument(
        "--skip-write-mt", help="Skip writing out qc mt", action="store_true"
    )
    parser.add_argument(
        "--skip-validate-mt",
        help="Skip validating the mt against common coding and noncoding variants",
        action="store_true",
    )
    parser.add_argument(
        "--project-list", help="List of seqr projects that are in the callset"
    )
    parser.add_argument(
        "--slack-channel", help="Slack channel to post results and notifications to."
    )
    parser.add_argument(
        "--overwrite", help="Overwrite previous paths", action="store_true"
    )

    args = parser.parse_args()

    if args.slack_channel:
        from slack_creds import slack_token

        with slack.slack_notifications(slack_token, args.slack_channel):

            main(args)
    else:
        main(args)
