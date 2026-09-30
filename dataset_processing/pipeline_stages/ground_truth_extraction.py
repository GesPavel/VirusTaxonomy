import gc
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset_processing.pipeline_stages.pairwise_feature_generation import _build_hit_matrix


# Query genomes per similarity block. The block holds one dense
# (block x candidate) score matrix, so this bounds it to a few tens of MB.
SIMILARITY_BLOCK = 512


def _add_feature_sets(df):
    df = df.copy()

    if "hmms_hits" in df.columns:
        df["hmm_set"] = df["hmms_hits"].apply(set)

    if "pc_hits" in df.columns:
        df["pc_set"] = df["pc_hits"].apply(set)

    return df


def _similarity_matrices(genome_dict, representation):
    """Hit matrices whose row products are the similarity scores.

    Similarity is a shared-hit count, so a sparse row product gives the whole
    all-vs-all grid at once instead of one set intersection per genome pair.
    Hybrid sums the two blocks, which is what the per-pair version did too.
    """
    if representation in {"hmm", "pc"}:
        matrix, row_of_genome = _build_hit_matrix(genome_dict)
        return [matrix], row_of_genome

    if representation == "hybrid":
        hmm_matrix, row_of_genome = _build_hit_matrix(genome_dict, key="hmm")
        pc_matrix, _ = _build_hit_matrix(genome_dict, key="pc")
        return [hmm_matrix, pc_matrix], row_of_genome

    raise ValueError(f"Unsupported representation: {representation}")


def _similarity_block(matrices, candidate_matrices, query_rows):
    """Dense (len(query_rows) x n_candidates) similarity scores."""
    block = None
    for matrix, candidate_matrix in zip(matrices, candidate_matrices):
        scores = (matrix[query_rows] @ candidate_matrix).toarray()
        block = scores if block is None else block + scores

    return block


def _build_genome_lookup(df, representation):
    if representation == "hmm":
        return dict(zip(df["Accession"], df["hmm_set"]))

    elif representation == "pc":
        return dict(zip(df["Accession"], df["pc_set"]))

    elif representation == "hybrid":
        return {
            acc: {
                "hmm": hmm_set,
                "pc": pc_set,
            }
            for acc, hmm_set, pc_set in zip(
                df["Accession"],
                df["hmm_set"],
                df["pc_set"]
            )
        }

    else:
        raise ValueError(f"Unsupported representation: {representation}")


def _generate_pairs_for_genome(
    genome_i,
    same_taxon_candidates,
    candidate_genomes,
    candidate_taxa,
    taxon_i,
    similarities,
    k_pos
):
    """
    Partners for one query genome, as (accession, same_taxon) tuples:

      * k_pos same-taxon partners, capped by how many actually exist,
      * ceil(k_pos / 2) hard negatives -- the most similar wrong-taxon genomes,
      * floor(k_pos / 2) random negatives drawn from the wrong-taxon rest.

    The label is known from the bucket a partner was drawn from, so no taxon
    lookup is needed downstream. A genome with no same-taxon partner returns
    nothing: it would otherwise contribute negatives only.

    `similarities` holds this genome's score against every candidate, computed
    in bulk by the caller.
    """
    positive_pool = [g for g in same_taxon_candidates if g != genome_i]
    k_pos = min(k_pos, len(positive_pool))

    if k_pos == 0:
        return []

    n_hard = -(-k_pos // 2)  # ceil
    n_random = k_pos // 2

    # A candidate is a valid negative exactly when its taxon differs, which also
    # drops genome_i itself. A stable descending sort keeps candidate order
    # among equal scores, as the sorted list of pairs did.
    negatives = np.flatnonzero(candidate_taxa != taxon_i)
    ranked = negatives[np.argsort(-similarities[negatives], kind="stable")]

    hard_negatives = [candidate_genomes[j] for j in ranked[:n_hard]]

    # Sampling positions out of range(n) draws the same values from the random
    # module as sampling the genomes themselves would, without building the
    # list of every remaining candidate for every query genome.
    n_remaining = len(ranked) - n_hard
    sampled = random.sample(range(n_remaining), min(n_random, n_remaining))
    random_negatives = [candidate_genomes[ranked[n_hard + j]] for j in sampled]

    return (
        [(g, 1) for g in random.sample(positive_pool, k_pos)]
        + [(g, 0) for g in hard_negatives + random_negatives]
    )


def build_pairs_dataset(
    df,
    rank,
    representation,
    partner_df=None,
    k_pos=10,
    random_seed=42
):
    """
    Build labelled genome pairs: for every genome of `df`, up to k_pos same-taxon
    partners and an equal number of wrong-taxon ones, half of them the most
    similar (hard) negatives and half random. See _generate_pairs_for_genome.

    With `partner_df` left as None the partners are drawn from `df` itself
    (self-pairing). Passing a `partner_df` draws them from that frame instead --
    e.g. calibration genomes paired against the training set, so calibration sees
    the same cross-set setting as prediction rather than a self-paired one.

    Column order follows prediction time, where genome1 is the train genome and
    genome2 the queried one (see probability_prediction.predict_probabilities):
    in cross-set mode the partner goes to genome1 and the `df` genome to genome2,
    so len1/len2 keep the same meaning at every stage of the pipeline.
    """
    random.seed(random_seed)

    cross_set = partner_df is not None

    df = _add_feature_sets(df)
    partner_df = _add_feature_sets(partner_df) if cross_set else df

    genome_dict = _build_genome_lookup(df, representation)

    if cross_set:
        genome_dict.update(_build_genome_lookup(partner_df, representation))

    # The only taxonomy structure kept: candidates grouped by taxon. It gives
    # the positive pool directly and, by exclusion, the negative one.
    taxon_to_candidates = partner_df.groupby(rank)["Accession"].apply(list).to_dict()

    query_genomes = list(zip(df["Accession"], df[rank]))
    candidate_genomes = partner_df["Accession"].tolist()
    candidate_taxa = partner_df[rank].to_numpy()

    matrices, row_of_genome = _similarity_matrices(genome_dict, representation)
    query_rows = np.array([row_of_genome[g] for g, _ in query_genomes], dtype=np.int32)
    candidate_rows = np.array([row_of_genome[g] for g in candidate_genomes], dtype=np.int32)
    candidate_matrices = [matrix[candidate_rows].T.tocsr() for matrix in matrices]


    rows = []
    skipped = 0

    progress = tqdm(total=len(query_genomes), desc="Building genome pairs")

    # Query genomes are handled in the original order, one block of
    # similarity scores at a time, so the random draws are unchanged.
    for start in range(0, len(query_genomes), SIMILARITY_BLOCK):
        block = query_genomes[start:start + SIMILARITY_BLOCK]
        similarity_block = _similarity_block(
            matrices, candidate_matrices, query_rows[start:start + len(block)]
        )

        for offset, (genome_i, taxon_i) in enumerate(block):
            progress.update(1)

            partners = _generate_pairs_for_genome(
                genome_i=genome_i,
                same_taxon_candidates=taxon_to_candidates.get(taxon_i, ()),
                candidate_genomes=candidate_genomes,
                candidate_taxa=candidate_taxa,
                taxon_i=taxon_i,
                similarities=similarity_block[offset],
                k_pos=k_pos,
            )

            if not partners:
                skipped += 1
                continue

            for genome_j, same_taxon in partners:
                genome1, genome2 = (genome_j, genome_i) if cross_set else (genome_i, genome_j)

                rows.append({
                    "genome1": genome1,
                    "genome2": genome2,
                    "same_taxon": same_taxon,
                })

    progress.close()

    # Feature sets and lookups are dead from here on; the frame built below is
    # the memory peak, so drop them before allocating it.
    del df, partner_df, genome_dict, taxon_to_candidates, query_genomes, candidate_genomes
    del matrices, candidate_matrices, query_rows, candidate_rows, candidate_taxa
    gc.collect()

    pairs_df = pd.DataFrame(rows)

    print(f"Built {len(pairs_df)} total pairs")
    print("Positive pairs:", pairs_df["same_taxon"].sum())
    print("Negative pairs:", len(pairs_df) - pairs_df["same_taxon"].sum())
    print(f"Genomes skipped (no same-taxon partner available): {skipped}")

    return pairs_df