import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, brier_score_loss

from dataset_processing.pipeline_stages.pairwise_feature_generation import compute_features
from dataset_processing.pipeline_stages.pairwise_feature_generation import build_name_to_feature_dict
from dataset_processing.pipeline_stages.probability_model_training import _describe_probs


def calibrate_model(
        base_model,
        df,
        val_pairs_df,
        feature_config,
        calibration_config,
        representation,
        partner_df=None
):
    """
    `partner_df` mirrors build_pairs_dataset: when the validation pairs were
    built across two sets, both sides have to be in the feature lookup.
    """
    method = calibration_config.get("method", "isotonic")

    genome_dict = build_name_to_feature_dict(df, representation)

    if partner_df is not None:
        genome_dict.update(build_name_to_feature_dict(partner_df, representation))

    X_val, y_val, _, _ = compute_features(
        val_pairs_df,
        genome_dict,
        feature_config,
        representation=representation
    )

    # Calculate pre-calibration stats
    base_probs = base_model.predict_proba(X_val)[:, 1]
    pre_log_loss = log_loss(y_val, base_probs)
    pre_brier = brier_score_loss(y_val, base_probs)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method=method
    )

    calibrated_model.fit(X_val, y_val)

    # Calculate post-calibration stats
    calibrated_probs = calibrated_model.predict_proba(X_val)[:, 1]
    post_log_loss = log_loss(y_val, calibrated_probs)
    post_brier = brier_score_loss(y_val, calibrated_probs)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CALIBRATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Calibration Method: {method}")
    report_lines.append(f"Validation Pairs Used: {len(val_pairs_df)}")
    report_lines.append("")
    report_lines.append("--- Pre-Calibration (Base Model) ---")
    report_lines.append(f"LogLoss: {pre_log_loss:.6f}")
    report_lines.append(f"Brier Score: {pre_brier:.6f}")
    report_lines.append("")
    report_lines.append(_describe_probs("PRE-CALIBRATION", base_probs))
    report_lines.append("")
    report_lines.append("--- Post-Calibration ---")
    report_lines.append(f"LogLoss: {post_log_loss:.6f}")
    report_lines.append(f"Brier Score: {post_brier:.6f}")
    report_lines.append("")
    report_lines.append(_describe_probs("POST-CALIBRATION", calibrated_probs))

    report_str = "\n".join(report_lines)

    return report_str, calibrated_model