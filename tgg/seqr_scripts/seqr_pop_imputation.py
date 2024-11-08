# Seqr Code for JUST Pop Imputation
import argparse
import json
import logging
import pickle

import hail as hl
from gnomad.sample_qc.ancestry import assign_population_pcs, pc_project
from gnomad.sample_qc.filtering import compute_stratified_metrics_filter
from gnomad.sample_qc.pipeline import filter_rows_for_qc
from gnomad.sample_qc.platform import (
    assign_platform_from_pcs,
    compute_callrate_mt,
    run_platform_pca,
)
from gnomad.utils import slack
from gnomad.utils.filtering import filter_to_autosomes
from hail.utils.misc import new_temp_file

# import tgg.resources as tgg_path 

# print(dir(tgg_path))

from tgg.resources import (  # missing_metrics_path,; mt_path,; remap_path,; sample_qc_ht_path,; sample_qc_tsv_path,; seq_metrics_path,
    VCFDataTypeError,
    rdg_gnomad_pop_pca_loadings_path,
    rdg_gnomad_rf_model_path,
    rdg_gnomad_v4_pop_pca_loadings_path,
    rdg_gnomad_v4_rf_model_path,
    val_coding_ht_path,
    val_noncoding_ht_path,
)

# from gnomad_qc.v4.resources.sample_qc import per_pop_min_rf_probs_json_path
from gnomad_qc.v4.sample_qc.assign_ancestry import assign_pop_with_per_pop_probs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger("seqr_sample_qc")
logger.setLevel(logging.INFO)

def get_all_sample_metadata(
    mt: hl.MatrixTable,
    build: int,
    data_type: str,
    data_source: str,
    version: int,
    remap_path: str,
    sample_metadata_path: str,
    dragen: bool,
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
    meta_ht = hl.import_table(sample_metadata_path).key_by("SAMPLE").repartition(1000)

    logger.info("Importing and annotating seqr ID names...")
    remap_ht = hl.import_table(remap_path).key_by("s")
    meta_ht = meta_ht.annotate(**remap_ht[meta_ht.key])
    meta_ht = meta_ht.annotate(
        seqr_id=hl.if_else(
            hl.is_missing(meta_ht.seqr_id), meta_ht.SAMPLE, meta_ht.seqr_id
        )
    )

    float_metrics = [
        "PCT_CONTAMINATION",
        "AL_PCT_CHIMERAS",
        "WGS_MEAN_COVERAGE",
        "WGS_MEDIAN_COVERAGE",
        "HS_MEAN_TARGET_COVERAGE",
        "HS_PCT_TARGET_BASES_20X",
    ]

    if dragen:
        logger.info("DRAGEN float metrics:")  # do this with other float metrics
        float_metrics = [
            "contamination_rate",
            "custom_chimer_pct",
            "mean_coverage",
            "percent_bases_at_20x",
        ]

    hl_floats = {f_i: hl.float(meta_ht[f_i]) for f_i in float_metrics}
    meta_ht = meta_ht.annotate(**hl_floats)

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

def run_population_pca(
    mt: hl.MatrixTable,
    build: int,
    num_pcs: int = 20,
    v2_cmg_model: bool = False,
    v4_custom_mid_model_151: bool = False,
    v4_custom_mid_model_179: bool = False,
    v4_custom_mid_model_cmgmid: bool = False,
) -> hl.Table:
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
    if v2_cmg_model and any([v4_custom_mid_model_151, v4_custom_mid_model_179]):
        raise ValueError("Cannot request two models, u dolt>:(")
    if all([v4_custom_mid_model_151, v4_custom_mid_model_179, v4_custom_mid_model_cmgmid]):
        raise ValueError("Cannot request two models, u dolt>:(")

    if v4_custom_mid_model_151:
        logger.info(
            "Reading in custom gnomAD v4 RF model with 151 spiked in v2 MID training samples..."
        )
        logger.info("Not doing this now")
        # loadings = hl.read_table(rdg_gnomad_v4_pop_pca_loadings_path())
        # model_path = "gs://marten-seqr-sandbox-storage/ancestry/gnomad.joint.v4.0.pop.RF_fit.pickle"
        # # rdg_gnomad_v4_rf_model_path()
    elif v4_custom_mid_model_179:
        logger.info(
            "Reading in custom gnomAD v4 RF model with 179 spiked in v2 MID training samples..."
        )
        logger.info("Not doing this now...")
        # loadings = hl.read_table(rdg_gnomad_v4_pop_pca_loadings_path())
        # model_path = "gs://marten-seqr-sandbox-storage/ancestry/gnomad.joint.v4.0_v2_179samples.pop.RF_fit.pickle"
        # # rdg_gnomad_v4_rf_model_path()
    elif v4_custom_mid_model_cmgmid:
        logger.info('80 percent of CMG Gleeson MID samples, plus Persian and Arab for MID spike')
        loadings = hl.read_table(rdg_gnomad_v4_pop_pca_loadings_path())
        # model_path = 'gs://marten-seqr-sandbox-storage/ancestry/pop_ht_custom_probs_cmgmidsamples_rfmodel_20240909.pickle'
        model_path = 'gs://marten-seqr-sandbox-storage/ancestry/rf_10162024/pop_ht_custom_probs_cmgmidsamples_rfmodel_20240917.pickle'
    elif not v2_cmg_model:
        logger.info("Reading in standard gnomAD v4 loadings and model...")
        loadings = hl.read_table(rdg_gnomad_v4_pop_pca_loadings_path())
        model_path = rdg_gnomad_v4_rf_model_path()
    else:
        logger.info("Reading in gnomAD v2 + CMG loadings and model...")
        loadings = hl.read_table(rdg_gnomad_pop_pca_loadings_path(build=38))
        model_path = rdg_gnomad_rf_model_path()

    logger.info(f"With num_pcs: {num_pcs}")
    mt = mt.select_entries("GT")
    scores = pc_project(mt, loadings)
    scores = (
        scores.annotate(scores=scores.scores[:num_pcs], known_pop="Unknown")
        .key_by("s")
        .checkpoint(new_temp_file("scores_temp", extension="ht").replace(
            "/tmp/", "gs://seqr-scratch-temp/")
        )
    )

    logger.info("Unpacking RF model")
    fit = None
    with hl.hadoop_open(model_path, "rb") as f:
        fit = pickle.load(f)

    logger.info("Running assign_population_pcs...")
    pop_pca_ht, ignore = assign_population_pcs(
        scores,
        pc_cols=scores.scores,
        output_col="qc_pop",
        fit=fit,
    )
    pop_pca_ht = pop_pca_ht.key_by("s")
    pop_pcs = {f"pop_PC{i+1}": scores.scores[i] for i in range(num_pcs)}
    scores = scores.annotate(**pop_pcs).drop("scores", "known_pop")
    pop_pca_ht = pop_pca_ht.annotate(**scores[pop_pca_ht.key])
    return pop_pca_ht

def main(args):

    hl.init(log="/seqr_sample_qc.log", tmp_dir='gs://seqr-scratch-temp/sqc_tmp/')
    hl._set_flags(
        use_new_shuffle="1"
    )  # Flag needed for hail 0.2.93, may be able to remove in future release.
    logger.info("Beginning seqr sample QC pipeline...")

    data_type = args.data_type
    build = args.build
    data_source = args.data_source
    version = args.callset_version
    is_test = args.is_test
    overwrite = args.overwrite
    callset_path = args.callset_path
    remap_path = args.remap_path
    sample_metadata_path = args.sample_metadata_path
    bucket_path = args.bucket_path
    v2_cmg_model = args.v2_cmg_model
    dragen = args.is_dragen
    v4_custom_mid_model_151 = args.v4_custom_mid_model_151
    v4_custom_mid_model_179 = args.v4_custom_mid_model_179
    v4_custom_mid_model_cmgmid = args.v4_custom_mid_model_cmgmid
    skip_platform_imputation = args.skip_platform_imputation
    output_suffix = args.output_suffix

    logger.info("Importing callset as mt...")
    mt = hl.read_matrix_table(callset_path).repartition(1000)

    mt = mt.annotate_entries(
        GT=hl.case()
        .when(mt.GT.is_diploid(), hl.call(mt.GT[0], mt.GT[1], phased=False))
        .when(mt.GT.is_haploid(), hl.call(mt.GT[0], phased=False))
        .default(hl.missing(hl.tcall))
    )
    # if not args.skip_validate_mt:
    #     logger.info("Validating data type...")
    #     validate_mt(mt, build, data_type)
    # else:
    #     
    logger.info("Skipping validation...")

    if is_test:
        logger.info("Creating test mt...")
        mt = hl.filter_intervals(
            mt,
            [
                hl.parse_locus_interval(
                    hl.if_else(build == "37", "22", "chr22"),
                    reference_genome=f"GRCh{build}",
                )
            ],
        ).sample_cols(
            0.01
        )  # .persist()
        mt = mt.checkpoint(
            new_temp_file("test_mt", extension="mt").replace(
                "/tmp/", "gs://seqr-scratch-temp/"
            )
        )

    logger.info("Annotating with sequencing metrics and filtered callrate...")
    logger.info("Skipping Metadata information...")
    # meta_ht = get_all_sample_metadata(
    #     mt,
    #     build,
    #     data_type,
    #     data_source,
    #     version,
    #     remap_path,
    #     sample_metadata_path,
    #     dragen,
    # )
    # meta_ht = meta_ht.checkpoint(
    #     new_temp_file(
    #         "metadata_ht_imported",
    #         extension="ht".replace("/tmp/", "gs://seqr-scratch-temp/"),
    #     )
    # )

    def _get_root_sqc_output(
        bucket_path,
        data_type,
        version,
        data_source,
        v4_custom_mid_model_151,
        v4_custom_mid_model_179,
        v2_cmg_model,
        output_suffix=None,
    ) -> None:
        model_text = "gnomAD_v4"
        if v4_custom_mid_model_151:
            model_text = "v4_custom_mid_model_151"
        if v4_custom_mid_model_179:
            model_text = "v4_custom_mid_model_179"
        if v4_custom_mid_model_cmgmid:
            model_text = "v4_custom_mid_model_cmgmid"

        elif v2_cmg_model:
            model_text = "v2_cmg_model"

        return f'{bucket_path}/{data_type}_v{version}_{data_source}_{model_text}{output_suffix if output_suffix else ""}'

    # mt = mt.annotate_cols(**meta_ht[mt.col_key], data_type=data_type)

    logger.info('Skipping annotating with sample metric filter flags...')
    # logger.info("Annotating with sample metric filter flags...")
    # metric_thresholds = {
    #     "callrate_thres": args.callrate_low_threshold,
    #     "contam_thres": args.contam_up_threshold,
    #     "chimera_thres": args.chimera_up_threshold,
    #     "wes_cov_thres": args.wes_coverage_low_threshold,
    #     "wgs_cov_thres": args.wgs_coverage_low_threshold,
    # }
    # mt = mt.annotate_cols(
    #     filter_flags=apply_filter_flags_expr(mt, data_type, metric_thresholds, dragen)
    # )

    # mt = mt.checkpoint(
    #     new_temp_file("annotation_mt", extension="mt").replace(
    #         "/tmp/", "gs://seqr-scratch-temp/"
    #     )
    # )

    logger.info("Assign platform or product...")
    logger.info("Skipping platform imputation...")
    # if skip_platform_imputation:
    #     logger.info("Skipping platform impuation...")
    #     mt = mt.annotate_cols(qc_platform="Skipped")
    # elif data_type == "WES" and data_source == "External":
    #     logger.info("Running platform imputation...")
    #     plat_ht = run_platform_imputation(
    #         mt,
    #         args.plat_min_cluster_size,
    #         args.plat_min_sample_size,
    #         args.plat_assignment_pcs,
    #     )
    #     mt = mt.annotate_cols(**plat_ht[mt.col_key])
    # elif data_source == "Internal":
    #     logger.info("Assigning platform from product in metadata...")
    #     mt = mt.annotate_cols(
    #         qc_platform=hl.if_else(hl.is_defined(mt.PRODUCT), mt.PRODUCT, "Unknown")
    #     )

    #     missing_metrics = mt.filter_cols(hl.is_defined(mt.PRODUCT), keep=False)
    #     missing_metrics.cols().select().export(
    #         "gs://seqr-scratch-temp/missing_metrics_new_new_v2cmg_test.tsv"
    #     )  #  TODO Add logging step that prints unexpected missing samples
    # else:
    #     mt = mt.annotate_cols(qc_platform="Unknown")
    logger.info("Assigning platform or product finished...")

    # mt = mt.checkpoint(
    #     new_temp_file("sexcheck_mt", extension="mt").replace(
    #         "/tmp/", "gs://seqr-scratch-temp/"
    #     )
    # )

    # mt = hl.read_matrix_table('gs://seqr-scratch-temp/sexcheck_mt-rSSTqdZRVtMTqC9lzKc7am.mt')

    num_pcs = 20
    if v2_cmg_model:
        num_pcs = 6

    logger.info("Projecting gnomAD population PCs...")
    pop_ht = run_population_pca(
        mt,
        build,
        num_pcs=num_pcs,
        v2_cmg_model=v2_cmg_model,
        v4_custom_mid_model_151=v4_custom_mid_model_151,
        v4_custom_mid_model_179=v4_custom_mid_model_179,
        v4_custom_mid_model_cmgmid = v4_custom_mid_model_cmgmid,
    )

    logger.info("Checkpointing pop_ht...")
    pop_ht = pop_ht.checkpoint(
        new_temp_file("genetic_ancestry", extension="ht").replace(
            "/tmp/", "gs://seqr-scratch-temp/"
        )
    )

    if args.v4_custom_per_pop_probs:
        logger.info("Assigning pops with per-pop probabilities...")
        custom_probs = (
            "gs://marten-seqr-sandbox-storage/ancestry/rf_10162024/default_v4_pop_min_probs.json"  
        )
        if v4_custom_mid_model_151:
            logger.info(
                "From v4 custom model with 151 mid training samples spiked in..."
            )
            custom_probs = (
                "gs://marten-seqr-sandbox-storage/ancestry/per_pop_min_probs.json"
            )
        elif v4_custom_mid_model_179:
            logger.info(
                "From v4 custom model with 179 mid training samples spiked in..."
            )
            custom_probs = "gs://marten-seqr-sandbox-storage/ancestry/per_pop_min_probs_179samples.json"
        elif v4_custom_mid_model_cmgmid:
            logger.info(
                "From v4 custom model with all 700+ cmg mid samples spiked in, via 20240917..."
            )
            #custom_probs = "gs://marten-seqr-sandbox-storage/ancestry/per_pop_min_probs_cmgmid_20240909.json"
            custom_probs = 'gs://marten-seqr-sandbox-storage/ancestry/rf_10162024/per_pop_min_probs_cmgmid_08nq_90cutoff_20241016.json'

        with hl.hadoop_open(custom_probs, "r") as d:
            min_probs = json.load(d)
        pop_ht = assign_pop_with_per_pop_probs(pop_ht, min_probs, missing_label="oth")
        pop_ht = pop_ht.checkpoint(
            new_temp_file("genetic_ancestry_per_pop_probs", extension="ht").replace(
                "/tmp/", "gs://seqr-scratch-temp/"
            )
        )

    logger.info("Annotating genetic ancestry inference information back on...")
    mt = mt.annotate_cols(**pop_ht[mt.col_key])

    logger.info('Skipping Hails sample qc...')
    # logger.info("Running Hail's sample qc...")
    # hail_metric_ht = run_hail_sample_qc(mt, data_type)
    # mt = mt.annotate_cols(**hail_metric_ht[mt.col_key])

    logger.info("Exporting sample QC tables...")
    ht = mt.cols()

    output_root = _get_root_sqc_output(
        bucket_path=bucket_path,
        data_type=data_type,
        version=version,
        data_source=data_source,
        v2_cmg_model=v2_cmg_model,
        v4_custom_mid_model_151=v4_custom_mid_model_151,
        v4_custom_mid_model_179=v4_custom_mid_model_179,
        output_suffix=output_suffix,
    )

    ht = ht.checkpoint(
        f"{output_root}.ht",
        overwrite=True,
    )
    ht.flatten().export(f"{output_root}_flattened_tsv.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--callset-path",
        help="Path to callset as mt",
        required=True,
    )
    parser.add_argument(
        "--remap-path",
        help="Path to remapping tsv",
        required=True,
    )
    parser.add_argument(
        "--sample-metadata-path",
        help="Path to sample metadata tsv",
        required=False,
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
    parser.add_argument(
        "--bucket-path",
        help="Path to bucket to output results to",
        default="gs://seqr-loading-temp/v03/GRCh38/SNV_INDEL/sample_qc",
    )
    parser.add_argument(
        "--v2-cmg-model",
        help="To run population PCA with gnomAD v2 + CMG Model & Loadings",
        action="store_true",
    )
    parser.add_argument(
        "--v4-custom-per-pop-probs",
        help="Assign imputed genetic ancestry using gnomAD v4.0 custom per pop probabilities",
        action="store_true",
    )
    parser.add_argument(
        "--is-dragen",
        help="Run with a number of different flags, to fit DRAGEN/GSV returns",
        action="store_true",
    )
    parser.add_argument(
        "--v4-custom-mid-model-151",
        help="Use gnomAD v4 RF model with 151 spiked in v2 MID training samples...",
        action="store_true",
    )
    parser.add_argument(
        "--v4-custom-mid-model-179",
        help="Use gnomAD v4 RF model with 179 spiked in v2 MID training samples...",
        action="store_true",
    )
    parser.add_argument(
        "--v4-custom-mid-model-cmgmid",
        help="Use gnomAD v4 RF model with 700+ CMG MID training samples...",
        action="store_true",
    )
    parser.add_argument(
        "--skip-platform-imputation",
        help="Pass to skip platform imputation, label as skip",
        action="store_true",
    )
    parser.add_argument(
        "--output-suffix",
        help="Optional suffix to add to file names for outputs",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.slack_channel:
        from slack_creds import slack_token

        with slack.slack_notifications(slack_token, args.slack_channel):

            main(args)
    else:
        main(args)
