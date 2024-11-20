import collections

import pymysql


# SQL db schema @ http://useast.ensembl.org/info/docs/api/core/core_schema.html
# To get list of available dbs, connect and run SHOW DATABASES


CURRENT_ENSEMBL_DATABASE = "homo_sapiens_core_107_38"

"""
homo_sapiens_cdna_102_38
homo_sapiens_cdna_103_38
homo_sapiens_core_102_38
homo_sapiens_core_103_38
homo_sapiens_funcgen_102_38
homo_sapiens_funcgen_103_38
homo_sapiens_otherfeatures_102_38
homo_sapiens_otherfeatures_103_38
homo_sapiens_rnaseq_102_38
homo_sapiens_rnaseq_103_38
homo_sapiens_variation_102_38
homo_sapiens_variation_103_38
"""


def get_gene_id_to_transcript_metadata(
        database=CURRENT_ENSEMBL_DATABASE,
        only_protein_coding=False,
        only_canonical_transcripts=False):
    """Retrieves a dictionary containing gene_id => a list of dictionaries each of which
    contains information about one transcript that belongs to that gene.

    Args:
        database (str): The Ensembl database name (eg. "homo_sapiens_core_107_38")
        only_protein_coding (bool): If True, only return protein-coding genes and protein-coding transcripts
        only_canonical_transcripts (bool): If True, only return canonical transcripts

    Return:
        dict: mapping ENSG id string to a list of dictionaries where each dictionary contains metadata fields
    """

    gene_id_to_transcript_id = collections.defaultdict(list)
    with pymysql.connect(host="useastdb.ensembl.org", user="anonymous", database=database) as conn:
        with conn.cursor() as cursor:
            if only_canonical_transcripts:
                join_clause = "canonical_transcript_id = transcript_id"
            else:
                join_clause = "transcript.gene_id = gene.gene_id"

            columns = [
                # Gene fields
                "gene.stable_id",
                "gene.biotype",
                "gene.created_date",
                "gene.modified_date",
                # Transcript fields
                "transcript.stable_id",
                "transcript.biotype",
                "transcript.created_date",
                "transcript.modified_date",
            ]

            columns_str = ", ".join(columns)
            query_string = f"SELECT {columns_str} FROM gene LEFT JOIN transcript ON {join_clause}"
            if only_protein_coding:
                query_string += " WHERE gene.biotype = 'protein_coding' AND transcript.biotype = 'protein_coding'"

            cursor.execute(query_string)

            for row in cursor:
                gene_and_transcript_info = dict(zip(columns, row))
                gene_id = gene_and_transcript_info['gene.stable_id']
                gene_id_to_transcript_id[gene_id].append(gene_and_transcript_info)

    return gene_id_to_transcript_id


def get_gene_id_to_transcript_ids(
        database=CURRENT_ENSEMBL_DATABASE,
        only_protein_coding=False,
        only_canonical_transcripts=False):
    """Returns a dictionary mapping each Ensembl gene_id => a list of transcript ids for that gene.

    Args:
        database (str): The Ensembl database name (eg. "homo_sapiens_core_107_38")
        only_protein_coding (bool): If True, only return protein coding genes
        only_canonical_transcripts (bool): If True, only return canonical transcripts
    Return:
        dict: mapping ENSG id string to a list of ENST id strings
    """

    gene_id_to_transcript_metadata_list = get_gene_id_to_transcript_metadata(
        database=database,
        only_canonical_transcripts=only_canonical_transcripts,
        only_protein_coding=only_protein_coding)

    return {
        gene_id: [
            transcript_metadata["transcript.stable_id"] for transcript_metadata in transcript_metadata_list
        ]
        for gene_id, transcript_metadata_list in gene_id_to_transcript_metadata_list.items()
    }


def get_gene_id_to_canonical_transcript_id(database=CURRENT_ENSEMBL_DATABASE, only_protein_coding=False):
    """Returns a dictionary mapping each Ensembl gene_id => canonical transcript id

    Args:
        database (str): The Ensembl database name (eg. "homo_sapiens_core_107_38")
        only_protein_coding (bool): Only include protein-coding genes

    Return:
        dict: mapping ENSG id string to the canonical ENST id string
    """

    gene_id_to_transcript_metadata_list = get_gene_id_to_transcript_metadata(
        database=database,
        only_canonical_transcripts=True,
        only_protein_coding=only_protein_coding)

    gene_id_to_canonical_transcript_id = {}
    for gene_id, transcript_metadata_list in gene_id_to_transcript_metadata_list.items():
        if len(transcript_metadata_list) > 1:
            raise Exception(f"{gene_id} has more than 1 canonical transcript")
        if len(transcript_metadata_list) == 0:
            raise Exception(f"{gene_id} has 0 canonical transcripts")

        gene_id_to_canonical_transcript_id[gene_id] = transcript_metadata_list[0]["transcript.stable_id"]

    return gene_id_to_canonical_transcript_id


def get_gene_created_modified_dates(
        database=CURRENT_ENSEMBL_DATABASE,
        only_protein_coding=False,
        only_canonical_transcripts=False):
    """Returns a dictionary mapping each Ensembl gene_id => a 2-tuple containing the created date and the modified date
    for that gene.

    Args:
        database (str): The Ensembl database name (eg. "homo_sapiens_core_107_38")
        only_protein_coding (bool): If True, only return protein coding genes
        only_canonical_transcripts (bool): If True, only return canonical transcripts
    Return:
        dict: mapping ENSG id string to a 2-tuple containing the created date and the modified date for that gene.
    """

    gene_id_to_transcript_metadata_list = get_gene_id_to_transcript_metadata(
        database=database,
        only_canonical_transcripts=only_canonical_transcripts,
        only_protein_coding=only_protein_coding)

    return {
        gene_id: (transcript_metadata_list[0]["gene.created_date"], transcript_metadata_list[0]["gene.modified_date"])
        for gene_id, transcript_metadata_list in gene_id_to_transcript_metadata_list.items()
    }


def get_transcript_created_modified_dates(
        database=CURRENT_ENSEMBL_DATABASE,
        only_protein_coding=False,
        only_canonical_transcripts=False):
    """Returns a dictionary mapping each Ensembl gene_id => a list of 3-tuples each containing the
    transcript id, created date, and modified date of a transcript for that gene.

    Args:
        database (str): The Ensembl database name (eg. "homo_sapiens_core_107_38")
        only_protein_coding (bool): If True, only return protein coding genes
        only_canonical_transcripts (bool): If True, only return canonical transcripts
    Return:
        dict: a dictionary mapping each Ensembl gene_id => a list of 3-tuples each containing the transcript id,
        created date, and modified date of a transcript for that gene.
    """

    gene_id_to_transcript_metadata_list = get_gene_id_to_transcript_metadata(
        database=database,
        only_canonical_transcripts=only_canonical_transcripts,
        only_protein_coding=only_protein_coding)

    return {
        gene_id: [
            (
                transcript_metadata['transcript.stable_id'],
                transcript_metadata['transcript.created_date'],
                transcript_metadata['transcript.modified_date']
            )
            for transcript_metadata in transcript_metadata_list
        ]
        for gene_id, transcript_metadata_list in gene_id_to_transcript_metadata_list.items()
    }
