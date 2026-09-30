import pandas as pd
import time
from dataset_processing.pipeline_stages.pairwise_feature_generation import compute_features
from dataset_processing.pipeline_stages.pairwise_feature_generation import build_name_to_feature_dict


def predict_probabilities(
    df_train,
    df_test,
    model,
    taxon_rank,
    feature_config,
    representation
):
    print("    Building all test × train pairs...")
    start = time.time()

    # -----------------------------
    # 1. Build all test x train pairs
    # -----------------------------
    df_pairs = df_train[["Accession"]].merge(df_test[["Accession"]], how="cross")
    df_pairs = df_pairs.rename(columns={
        "Accession_x": "genome1",
        "Accession_y": "genome2"
    })

    print(f"    Pair dataframe shape: {df_pairs.shape}")
    print(f"    Pair generation done in {time.time() - start:.2f}s")

    # -----------------------------
    # 2. Build genome lookup
    # -----------------------------
    print("    Building genome lookup...")
    start = time.time()

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    genome_dict = build_name_to_feature_dict(df_all, representation)

    print(f"    Genome lookup built for {len(genome_dict):,} genomes in {time.time() - start:.2f}s")

    # -----------------------------
    # 3. Compute pairwise features
    # -----------------------------
    print("    Computing pairwise features...")
    start = time.time()

    X, _, genome_list_train, genome_list_test = compute_features(
        pairs_df=df_pairs,
        genome_dict=genome_dict,
        feature_config=feature_config,
        representation=representation,
        process_target_var=False
    )

    print(f"    Feature matrix shape: {X.shape}")
    print(f"    Feature computation done in {time.time() - start:.2f}s")

    # -----------------------------
    # 4. Predict pairwise probabilities
    # -----------------------------
    print("    Predicting pairwise probabilities...")
    start = time.time()

    probabilities = model.predict_proba(X)[:, 1]

    print(f"    Probability prediction done in {time.time() - start:.2f}s")

    print("    Building probability dataframe...")
    start = time.time()

    train_taxon_lookup = df_train.set_index("Accession")[taxon_rank].to_dict()
    test_taxon_lookup = df_test.set_index("Accession")[taxon_rank].to_dict()

    taxon_train_list = [train_taxon_lookup[g] for g in genome_list_train]
    taxon_test_list = [test_taxon_lookup[g] for g in genome_list_test]

    df_probabilities = pd.DataFrame({
        "genome_test": genome_list_test,
        "genome_train": genome_list_train,
        "taxon_train": taxon_train_list,
        "true_taxon": taxon_test_list,
        "probability": probabilities
    })

    print(f"    Probability dataframe shape: {df_probabilities.shape}")
    print(f"    Probability dataframe built in {time.time() - start:.2f}s")

    return df_probabilities

