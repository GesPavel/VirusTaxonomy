import yaml
from pathlib import Path
from datetime import datetime
import json
import argparse
import gc

import pandas as pd

from dataset_processing.pipeline_stages.ground_truth_extraction import build_pairs_dataset
from dataset_processing.pipeline_stages.novelty_fit import (
    DEFAULT_FPR_BUDGETS,
    PRIMARY_FPR_BUDGET,
    REPORTED_FPR_BUDGETS,
    fit_novelty_threshold,
)
from dataset_processing.pipeline_stages.probability_model_calibration import calibrate_model
from dataset_processing.pipeline_stages.probability_model_training import train_probability_model
from dataset_processing.pipeline_stages.probability_prediction import predict_probabilities
from dataset_processing.pipeline_stages.taxon_assignment import aggregate_and_summarize
from dataset_processing.pipeline_stages.taxon_assignment_evaluation import evaluate_assignment_results
from dataset_processing.train_test_split import load_dataset_split
from dataset_processing.util import (
    load_scenario_pickle,
    discover_test_variants,
)
from dataset_processing.paths import (
    ensure_dir,
    get_metrics_path,
    get_novelty_pr_curve_path,
    get_novelty_roc_curve_path,
    get_novelty_threshold_path,
    get_top_candidates_path,
)


# =========================
# Config loading
# =========================

def read_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =========================
# Experiment folder
# =========================

def get_experiments_root(config: dict) -> Path:
    return Path(config["experiment"].get("output_root") or "experiments")


def create_experiment_dir(config: dict) -> Path:
    """
    The experiment dir now only holds the final metrics (and the config used to
    produce them). Every intermediate artefact stays in memory.
    """
    exp_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = get_experiments_root(config) / f"{timestamp}_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return exp_dir


# =========================
# Stages
# =========================

def run_split_stage(config: dict, train_df):
    print("=== [1/8] SPLITTING DATASET ===")

    taxonomy_rank = config["data"]["taxon_rank"]
    split_config = config["split"]

    # The external test sets are not needed to build the split; they are loaded
    # later, at prediction time.
    empty_test_df = train_df.iloc[0:0]

    _, true_train, calibration, novelty_fit, known_taxa = load_dataset_split(
        train_df=train_df,
        test_df=empty_test_df,
        taxonomy_rank=taxonomy_rank,
        calibration_fraction=split_config["calibration_fraction"],
        novelty_fit_fraction=split_config["novelty_fit_fraction"],
        random_seed=config["experiment"]["random_seed"],
    )

    return true_train, calibration, novelty_fit, known_taxa


def run_pair_generation_stage(config: dict, true_train_df, calibration_df):
    print("=== [2/8] BUILDING GROUND TRUTH DATASET FOR PROBABILISTIC MODEL ===")

    taxon_rank = config["data"]["taxon_rank"]
    representation = config["data"]["representation"]

    print("--- Generating pairs for true_train ---")
    train_pairs_df = build_pairs_dataset(
        df=true_train_df,
        rank=taxon_rank,
        representation=representation,
        k_pos=config["pair_generation"]["k_pos"],
        random_seed=config["experiment"]["random_seed"],
    )

    # Calibration genomes are paired against the training set, not against each
    # other: that is the setting the model actually faces at prediction time.
    print("--- Generating pairs for calibration (against true_train) ---")
    val_pairs_df = build_pairs_dataset(
        df=calibration_df,
        rank=taxon_rank,
        representation=representation,
        partner_df=true_train_df,
        k_pos=config["pair_generation"]["k_pos"],
        random_seed=config["experiment"]["random_seed"],
    )

    return train_pairs_df, val_pairs_df


def run_training_stage(config: dict, true_train_df, train_pairs_df):
    print("=== [3/8] TRAINING MODEL ===")

    report_str, model = train_probability_model(
        df=true_train_df,
        pairs_df=train_pairs_df,
        taxon_rank=config["data"]["taxon_rank"],
        model_type=config["model"]["type"],
        model_config=config["model"],
        feature_config=config["features"],
        representation=config["data"]["representation"],
        random_seed=config["experiment"]["random_seed"],
    )

    print(report_str)

    return model


def run_calibration_stage(config: dict, calibration_df, true_train_df, val_pairs_df, model):
    print("=== [4/8] CALIBRATING MODEL ===")

    calibration_report, calibrated_model = calibrate_model(
        base_model=model,
        df=calibration_df,
        val_pairs_df=val_pairs_df,
        feature_config=config["features"],
        calibration_config=config["calibration"],
        representation=config["data"]["representation"],
        # The validation pairs span both sets, so both have to be resolvable.
        partner_df=true_train_df,
    )

    print(calibration_report)

    return calibrated_model


# Upper bound on the number of (test, train) pairs materialised at once. The
# pairwise feature matrix is the memory driver: at 10 float64 columns, 40M pairs
# is ~3.2 GB for X alone, and several times that once the pair frame, the python
# genome lists and the probability frame are counted. Aggregation then sorts the
# probability frame, which peaks at roughly 3x its size -- measured at 6.5 GB for
# a 20M chunk. Smaller chunks cost no extra total time, only more of them.
MAX_PAIRS_PER_CHUNK = 20_000_000

# The main test set (absent in scenario 1, which only has dark-matter variants).
MAIN_TEST_VARIANT = "test"

# How many candidate taxa per genome are kept for post-hoc analysis, main test
# set only.
N_TOP_CANDIDATES = 5


def _predict_and_assign_subset(
    config, model, true_train_df, subset_df, known_taxa, subset_name, n_top_candidates
):
    """
    Predict + aggregate one subset, in chunks of test genomes.

    Chunking is over test genomes only, so every (test genome, train genome)
    pair for a given test genome stays inside a single chunk. Aggregation groups
    by test genome, so the concatenated per-chunk results are identical to what a
    single-shot run would produce -- it only bounds peak memory.
    """
    n_train = len(true_train_df)
    chunk_size = max(1, MAX_PAIRS_PER_CHUNK // max(1, n_train))
    n_chunks = (len(subset_df) + chunk_size - 1) // chunk_size


    chunk_results = []

    for chunk_index in range(n_chunks):
        chunk_df = subset_df.iloc[chunk_index * chunk_size:(chunk_index + 1) * chunk_size]

        print(
            f"--- {subset_name}: chunk {chunk_index + 1}/{n_chunks} "
            f"({len(chunk_df):,} test genomes) ---"
        )

        df_probabilities = predict_probabilities(
            df_train=true_train_df,
            df_test=chunk_df,
            model=model,
            taxon_rank=config["data"]["taxon_rank"],
            feature_config=config["features"],
            representation=config["data"]["representation"],
        )

        chunk_results.append(
            aggregate_and_summarize(
                df_probabilities=df_probabilities,
                aggregation_config=config["aggregation"],
                known_taxa=known_taxa,
                n_top_candidates=n_top_candidates,
            )
        )

        del df_probabilities, chunk_df
        gc.collect()

    if len(chunk_results) == 1:
        return chunk_results[0]

    return pd.concat(chunk_results, ignore_index=True)


def run_prediction_and_assignment_stages(
    config: dict, scenario, model, true_train_df, novelty_fit_df, known_taxa
):
    """
    Stages 5 and 6, fused.

    The pairwise probability table is (n_test x n_train) rows -- hundreds of
    millions for the bigger dark-matter variants -- so it is never kept beyond
    the chunk it belongs to. Only the (small) per-genome assignment results
    survive the loop.
    """
    print("=== [5/8] PROBABILITY PREDICTION + [6/8] TAXON ASSIGNMENT ===")

    # The novelty-fit subset (used to fit the threshold) plus every available
    # test set: the main test and each dark-matter variant.
    subset_names = ["novelty_fit"] + discover_test_variants(config, scenario)

    assignments = {}

    for subset_name in subset_names:
        if subset_name == "novelty_fit":
            subset_df = novelty_fit_df
        else:
            subset_df = load_scenario_pickle(config, scenario, subset_name)

        assignments[subset_name] = _predict_and_assign_subset(
            config, model, true_train_df, subset_df, known_taxa, subset_name,
            # Only the main test set carries the candidate ranking.
            n_top_candidates=N_TOP_CANDIDATES if subset_name == MAIN_TEST_VARIANT else 0,
        )

        if subset_name != "novelty_fit":
            del subset_df
            gc.collect()

    return assignments


def save_top_candidates(work_dir: Path, assignments: dict):
    """
    Per-genome ranking of the N best candidate taxa with their raw aggregated
    scores, for the main test set only (scenario 1 has none). Purely an analysis
    artefact -- nothing in the pipeline reads it back.
    """
    if MAIN_TEST_VARIANT not in assignments:
        print("    No main test set in this scenario, skipping top candidates.")
        return

    columns = ["genome_test", "true_taxon", "aggregation_method", "aggregation_k"]
    for i in range(1, N_TOP_CANDIDATES + 1):
        columns += [f"top{i}_taxon", f"top{i}_score"]

    path = get_top_candidates_path(work_dir, MAIN_TEST_VARIANT)
    assignments[MAIN_TEST_VARIANT][columns].to_csv(path, index=False)

    print(f"    Top-{N_TOP_CANDIDATES} candidate taxa written to {path}")


def run_novelty_fit_stage(config: dict, work_dir: Path, novelty_fit_results):
    print("=== [7/8] NOVELTY FIT ===")

    use_normalized_probs = config["novelty"]["use_normalized_probabilities"]
    fpr_budgets = config["novelty"].get("max_fpr_budgets") or DEFAULT_FPR_BUDGETS
    reported_budgets = config["novelty"].get("reported_fpr_budgets") or REPORTED_FPR_BUDGETS
    primary_budget = config["novelty"].get("primary_fpr_budget") or PRIMARY_FPR_BUDGET

    result = fit_novelty_threshold(
        novelty_fit_results,
        use_normalized_probs=use_normalized_probs,
        fpr_budgets=fpr_budgets,
        reported_budgets=reported_budgets,
        primary_budget=primary_budget,
    )

    roc_path = get_novelty_roc_curve_path(work_dir)
    pr_path = get_novelty_pr_curve_path(work_dir)
    result.save_figures(roc_path, pr_path)

    threshold_path = get_novelty_threshold_path(work_dir)
    with open(threshold_path, "w") as f:
        json.dump(result.summary(), f, indent=2)

    print(
        f"    Novelty fit complete. Threshold: {result.threshold:.6g} "
        f"(TPR {result.tpr[result.chosen_index]:.3f} at FPR "
        f"{result.fpr[result.chosen_index]:.3f}, budget "
        f"{result.fpr_budget:.0%})"
    )
    print(f"    AUROC: {result.auroc:.4f}    AUPRC: {result.auprc:.4f}")
    print("    Operating point per FPR budget:")
    for budget, index in result.operating_points.items():
        if index is None:
            print(f"      FPR <= {budget:.0%}: no usable point")
            continue
        mark = "  <- used" if index == result.chosen_index else ""
        print(
            f"      FPR <= {budget:.0%}: TPR {result.tpr[index]:.4f} "
            f"at FPR {result.fpr[index]:.4f}, precision {result.precision[index]:.4f}, "
            f"threshold {result.thresholds[index]:.6g}{mark}"
        )
    print(f"    Threshold summary written to {threshold_path}")
    print(f"    Curves written to {roc_path} and {pr_path}")

    return result.threshold


def run_evaluation_stage(config: dict, scenario, work_dir: Path, assignments: dict, threshold):
    print("=== [8/8] EVALUATION ===")

    use_normalized_probs = config["novelty"]["use_normalized_probabilities"]

    # Produce a separate report for the main test and each dark-matter variant,
    # all evaluated against the shared novelty threshold.
    for variant in discover_test_variants(config, scenario):
        print(f"--- Evaluating test variant: {variant} ---")

        metrics = evaluate_assignment_results(
            assignments[variant], threshold, use_normalized_probs
        )

        metrics_path = get_metrics_path(work_dir, variant)
        with open(metrics_path, "w") as f:
            json.dump(
                {"novelty_threshold": threshold, "assignment": metrics},
                f,
                indent=2,
            )

        print(f"    Metrics written to {metrics_path}")


# =========================
# Main
# =========================

def resolve_scenarios(config: dict) -> list:
    """Turn data.chosen_scenario (1|2|3|4|all) into the list of scenarios to run."""
    chosen = config["data"]["chosen_scenario"]
    if str(chosen).lower() == "all":
        return [1, 2, 3, 4]
    return [int(chosen)]


def run_scenario_pipeline(config: dict, scenario, work_dir: Path):
    """
    Run the full pipeline for a single scenario. Everything except the final
    metrics is kept in memory and handed from stage to stage.
    """
    train_df = load_scenario_pickle(config, scenario, "train")

    true_train_df, calibration_df, novelty_fit_df, known_taxa = run_split_stage(
        config, train_df
    )

    # The full train pickle is no longer needed; drop the reference so the
    # subsets are the only thing held.
    del train_df
    gc.collect()

    train_pairs_df, val_pairs_df = run_pair_generation_stage(
        config, true_train_df, calibration_df
    )

    model = run_training_stage(config, true_train_df, train_pairs_df)

    model = run_calibration_stage(
        config, calibration_df, true_train_df, val_pairs_df, model
    )

    # Pair tables and the calibration subset are only inputs to train/calibrate.
    del train_pairs_df, val_pairs_df, calibration_df
    gc.collect()

    assignments = run_prediction_and_assignment_stages(
        config, scenario, model, true_train_df, novelty_fit_df, known_taxa
    )

    save_top_candidates(work_dir, assignments)

    threshold = run_novelty_fit_stage(config, work_dir, assignments["novelty_fit"])

    run_evaluation_stage(config, scenario, work_dir, assignments, threshold)


def run_experiment(config_path: Path):
    config = read_config(config_path)
    exp_dir = create_experiment_dir(config)

    scenarios = resolve_scenarios(config)
    multi_scenario = len(scenarios) > 1

    for scenario in scenarios:
        print(f"\n########## SCENARIO {scenario} ##########")

        # Flat layout for a single scenario; per-scenario subdirs when running all.
        if multi_scenario:
            work_dir = ensure_dir(exp_dir / f"scenario{scenario}")
        else:
            work_dir = exp_dir

        run_scenario_pipeline(config, scenario, work_dir)

    print(f"Experiment finished successfully. Metrics in: {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    run_experiment(Path(args.config))