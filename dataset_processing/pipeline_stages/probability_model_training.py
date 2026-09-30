import numpy as np
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from dataset_processing.pipeline_stages.pairwise_feature_generation import compute_features
from dataset_processing.pipeline_stages.pairwise_feature_generation import build_name_to_feature_dict


def _build_model(model_type, model_config, random_seed=42):
    if model_type == "logreg":
        cfg = model_config["logreg"]
        return LogisticRegression(
            max_iter=cfg["max_iter"],
            C=cfg["C"],
            solver=cfg["solver"],
            random_state=random_seed
        )

    elif model_type == "xgboost":
        cfg = model_config["xgboost"]
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            eval_metric=cfg["eval_metric"],
            random_state=random_seed,
            use_label_encoder=False
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _filter_pairs(pairs_df, allowed_genomes):
    allowed_genomes = set(allowed_genomes)
    return pairs_df[
        pairs_df["genome1"].isin(allowed_genomes) &
        pairs_df["genome2"].isin(allowed_genomes)
    ].copy()


def _describe_probs(name, probs):
    lines = []
    lines.append(f"{name} probability distribution:")
    lines.append(f"  min: {np.min(probs):.6f}")
    lines.append(f"  max: {np.max(probs):.6f}")
    lines.append(f"  mean: {np.mean(probs):.6f}")
    lines.append(f"  std: {np.std(probs):.6f}")

    percentiles = np.percentile(probs, [1, 5, 25, 50, 75, 95, 99])
    lines.append(f"  percentiles (1,5,25,50,75,95,99): {percentiles.tolist()}")
    lines.append(f"  % < 0.01: {np.mean(probs < 0.01):.6f}")
    lines.append(f"  % > 0.99: {np.mean(probs > 0.99):.6f}")
    lines.append(f"  % in [0.4, 0.6]: {np.mean((probs >= 0.4) & (probs <= 0.6)):.6f}")

    return "\n".join(lines)


def train_probability_model(
    df,
    pairs_df,
    taxon_rank,
    model_type,
    model_config,
    feature_config,
    representation,
    random_seed=42
):
    genome_dict = build_name_to_feature_dict(df, representation)

    X_train, y_train, _, _ = compute_features(
        pairs_df,
        genome_dict,
        feature_config,
        representation=representation
    )

    model = _build_model(model_type, model_config, random_seed=random_seed)

    model.fit(X_train, y_train)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PAIRWISE MODEL TRAINING REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Model: {model_type}")
    report_lines.append(f"Taxonomic rank: {taxon_rank}")
    report_lines.append("")
    report_lines.append(f"Trained on {len(pairs_df)} pairs.")
    report_lines.append(f"Feature matrix shape: {X_train.shape}")
    report_lines.append("")
    report_lines.append("Model trained on all available data. No internal validation performed.")

    report_str = "\n".join(report_lines)

    return report_str, model