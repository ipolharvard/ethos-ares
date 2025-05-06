from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .inference.constants import Reason

sns.set_theme(context="notebook")


def compute_basic_metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "prevalence": y_true.mean(),
        "auc": roc_auc_score(y_true, y_pred),
        "auprc": -np.trapezoid(*roc_curve(y_true, y_pred)[:2]),
    }


def objective_function(std, points, equal_variance=False):
    thresholds = np.linspace(-10, 11, num=10000)
    std2 = std[0] if equal_variance else std[1]
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=0, scale=std[0])
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=1, scale=std2)
    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    val = sum(np.min((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points)
    return val


def compute_fitted_metrics(
    y_true,
    y_pred,
    equal_variance: bool = False,
    operating_point: float | None = None,
    operating_point_type: str = "01",
    auc_interpolation_type="gaussian",  # 'quadratic','linear'
) -> dict:
    fpr_points, tpr_points, thresholds = roc_curve(y_true, y_pred)
    points = np.stack((tpr_points, fpr_points), axis=1).tolist()

    # number of samples in interpolation curves
    samples = 10000
    tpr_values = np.linspace(0, 1, samples)
    fpr_values = np.linspace(0, 1, samples)
    if auc_interpolation_type == "gaussian":
        delta = 1e-6
        upper_const = 10
        # Define the range constraints for x and y
        std_constraint = {
            "type": "ineq",
            "fun": lambda x: np.array([x[0] - delta, upper_const - x[0]]),
        }
        std2_constraint = {
            "type": "ineq",
            "fun": lambda x: np.array([x[1] - delta, upper_const - x[1]]),
        }
        if equal_variance:
            constraints = [std_constraint]
            x0 = np.array([1.0])
        else:
            constraints = [std_constraint, std2_constraint]
            x0 = np.array([0.5, 0.5])

        result = minimize(objective_function, x0, args=(points,), constraints=constraints)

        if equal_variance:
            optimal_x = result.x
        else:
            optimal_x, optimal_y = result.x

        # Parameters for hypothesis 1 (mean and standard deviation)
        mean_hypothesis_1 = 0.0
        std_dev_hypothesis_1 = optimal_x

        # Parameters for hypothesis 2 (mean and standard deviation)
        mean_hypothesis_2 = 1
        std_dev_hypothesis_2 = optimal_x
        if not equal_variance:
            std_dev_hypothesis_2 = optimal_y

        # Calculate the cumulative distribution functions (CDFs) for the two distributions
        min_v = -5
        max_v = 10
        lattice_points = np.linspace(min_v, max_v, num=samples)

        cdf_hypothesis_1 = norm.cdf(
            lattice_points, loc=mean_hypothesis_1, scale=std_dev_hypothesis_1
        )
        cdf_hypothesis_2 = norm.cdf(
            lattice_points, loc=mean_hypothesis_2, scale=std_dev_hypothesis_2
        )
        # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr_values = 1 - cdf_hypothesis_2
        fpr_values = 1 - cdf_hypothesis_1
    elif auc_interpolation_type == "linear" or auc_interpolation_type == "quadratic":
        unique_fpr_points = []
        unique_tpr_points = []

        fpr_dict = defaultdict(list)
        for x, y in zip(fpr_points, tpr_points):
            fpr_dict[x].append(y)

        for x, y_values in fpr_dict.items():
            unique_fpr_points.append(x)
            unique_tpr_points.append(np.mean(y_values))  # Take the mean of Y values for duplicates

        # Generate 10,000 equidistant points in the range of X
        # Note this is not ideal as interpolation point should be equdistanmt in 2D
        fpr_values = np.linspace(
            np.array(unique_fpr_points).max(), np.array(unique_fpr_points).min(), samples
        )
        # Create the interpolation function
        interpolation_function = interp1d(
            unique_fpr_points, unique_tpr_points, kind=auc_interpolation_type
        )
        # Calculate interpolated y values
        tpr_values = interpolation_function(fpr_values)

    lattice_idx = [
        np.argmin((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points
    ]

    if operating_point is None:
        match operating_point_type:
            case "01":
                # Find the best operating point defined as the closest point to (0,1)
                min_idx = np.argmin(fpr_values**2 + (tpr_values - 1) ** 2)
            case "Youden":
                # Find the best operating point defined as the point with the maximum Youden index
                min_idx = np.argmax(tpr_values - fpr_values)
            case "maxF1":
                # Find the best operating point defined as the point with the maximum F1 score
                min_idx = np.argmax(
                    2 * tpr_values * (1 - fpr_values) / (tpr_values + (1 - fpr_values))
                )
            case _:
                raise ValueError("operating_point_type must be one of: '01', 'Youden', 'maxF1'")

        # fit interpolation function between y and x points,
        # where y is thresholds and x is lattice_points
        f = np.poly1d(np.polyfit(lattice_idx[1:-1], thresholds[1:-1], 3))
        operating_point = f(min_idx)
    else:  # use provided operating point
        f = np.poly1d(np.polyfit(lattice_idx[1:-1], thresholds[1:-1], 3))
        min_idx_start = int((0 - min_v) / (max_v - min_v) * samples)
        max_idx_start = int((1 - min_v) / (max_v - min_v) * samples)
        # find the closest point to the operating point
        diff = f(np.arange(min_idx_start, max_idx_start - 1)) - operating_point
        min_idx = np.argmin(diff**2) + min_idx_start

    fpr = fpr_values[min_idx]
    tpr = tpr_values[min_idx]

    positives = np.sum(np.asarray(y_true) == 1)
    negatives = len(y_true) - positives

    denominator = tpr_values * positives + fpr_values * negatives
    recall_values = tpr_values
    more_than_zero = denominator > 0
    precision_values = np.divide(tpr_values * positives, denominator, where=more_than_zero)
    precision_values[~more_than_zero] = 1
    # =======================================================
    # Compute metrics now using the operating point
    # =======================================================

    tp = tpr * positives
    fn = (1 - tpr) * positives
    tn = (1 - fpr) * negatives
    fp = fpr * negatives

    # tier 1
    auprc = -np.trapezoid(precision_values, recall_values)
    auc = -np.trapezoid(tpr_values, fpr_values)
    accuracy = (tp + tn) / (positives + negatives)
    sensitivity = tpr
    specificity = 1 - fpr

    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    plr = sensitivity / (1 - specificity)
    nlr = (1 - sensitivity) / specificity
    f1 = tp / (tp + (fp + fn) / 2)

    precision_points, recall_points, _ = precision_recall_curve(y_true, y_pred)

    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "plr": plr,
        "nlr": nlr,
        "f1": f1,
        "recall": tpr,
        "precision_points": precision_points,
        "recall_points": recall_points,
        "precision_values": precision_values,
        "recall_values": recall_values,
        "tpr_points": tpr_points,
        "fpr_points": fpr_points,
        "tpr_values": tpr_values[::-1],
        "fpr_values": fpr_values[::-1],
        "operating_point": operating_point,
        "operating_point_index_in_values": samples - 1 - min_idx,
    }


def print_auc_roc_plot(res, fitted_res, title="AUC-ROC", lw=2, clinical=False):
    plt.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--", label="Random Guess")
    plt.plot(
        fitted_res["fpr_values"],
        fitted_res["tpr_values"],
        color="darkorange",
        lw=lw,
        label="AUC-ROC Fitted",
    )
    plt.scatter(fitted_res["fpr_points"], fitted_res["tpr_points"])
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)

    text = [
        f"N={res['n']:,}",
        f"prevalence={res['prevalence']:.2%}",
    ]
    if clinical:
        text.extend(
            [
                f"auc={fitted_res['auc']:.3f}",
            ]
        )
    else:
        text.extend(
            [
                f"auc={res['auc']:.3f}",
                f"fitted_auc={fitted_res['auc']:.3f}",
                f"fitted_f1-score={fitted_res['f1']:.3f}",
                f"fitted_precision={fitted_res['precision']:.3f}",
                f"fitted_recall={fitted_res['recall']:.3f}",
                f"fitted_accuracy={fitted_res['accuracy']:.3f}",
            ]
        )
    anc = AnchoredText(
        "\n".join(text),
        loc="lower right",
        frameon=True,
        pad=0.3,
        prop=dict(size=12),
    )
    anc.patch.set_boxstyle("round,pad=0.2")
    plt.gca().add_artist(anc)


def load_results(input_dir: str | Path) -> pl.DataFrame:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {input_dir}")

    parquet_dfs = []
    if parquet_fps := list(input_dir.rglob("*.parquet")):
        parquet_dfs = [
            pl.read_parquet(res_path, glob=False).with_columns(pl.col("^.*time$").cast(pl.Duration))
            for res_path in parquet_fps
        ]

    # To be removed in the future
    json_dfs = []
    if json_fps := list(input_dir.rglob("*.json")):
        json_dfs = [
            pl.read_json(res_path, infer_schema_length=None).with_columns(
                pl.col("^.*time$").cast(pl.Duration)
            )
            for res_path in json_fps
        ]

    if not parquet_dfs and not json_dfs:
        raise FileNotFoundError(f"No results found in {input_dir}")

    return pl.concat((*parquet_dfs, *json_dfs), how="diagonal")


def preprocess_inference_results(
    input_dir: str | Path,
    actual_expr: pl.Expr,
    expected_expr: pl.Expr = None,
    filter_ambiguous: pl.Expr = None,
    additional_columns: list[str] | None = None,
    max_rep_num: int | None = None,
    group_by_col: str | pl.Expr = "data_idx",
    warn_on_dropped: bool = True,
) -> pl.DataFrame:
    df = load_results(input_dir)
    prev_len = len(df)
    df = df.filter(pl.col("stop_reason").is_in([Reason.GOT_TOKEN, Reason.TIME_LIMIT]))
    if warn_on_dropped and (dropped := prev_len - len(df)):
        logger.warning(f"Dropped {dropped:,} results due to stop reason: {Reason.KEY_ERROR}.")

    if filter_ambiguous is not None:
        prev_len = len(df)
        df = df.filter(filter_ambiguous)
        if warn_on_dropped and (dropped := prev_len - len(df)):
            logger.warning(f"Dropped {dropped:,} ({dropped / prev_len:.2%}) ambiguous results.")

    optional_columns = [col for col in ("icu_stay_id", "hadm_id", "stay_id") if col in df.columns]
    aggregations = [
        ("expected", "first"),
        ("actual", "mean"),
        ("true_token_dist", "first"),
        ("token_dist", "mean"),
        ("true_token_time", "first"),
        ("token_time", "mean"),
        *[
            (col_name, "first")
            for col_name in ["patient_id", *optional_columns, *(additional_columns or [])]
        ],
    ]
    max_rep_num_expr = pl.min_horizontal(max_rep_num, pl.len())

    return (
        df.lazy()
        .with_columns(
            actual=actual_expr,
            expected=expected_expr if expected_expr is not None else "expected",
        )
        .group_by(group_by_col)
        .agg(
            *[
                (
                    getattr(pl.col(col_name), op)()
                    if max_rep_num is None
                    else getattr(pl.col(col_name).sample(max_rep_num_expr), op)()
                )
                for col_name, op in aggregations
            ],
            counts=max_rep_num_expr,
        )
        .collect()
    )


def compute_and_print_metrics(y_true, y_pred, figure_title: str, **kwargs) -> dict:
    basic_res = compute_basic_metrics(y_true, y_pred)
    fitted_res = compute_fitted_metrics(y_true, y_pred, **kwargs)
    print_auc_roc_plot(basic_res, fitted_res, figure_title)
    fitted_res = {f"fitted_{k}": v for k, v in fitted_res.items()}
    return basic_res | fitted_res


def plot_calibration_curve(y_true, y_pred, n_bins: int = 10):
    """Plots the calibration curve for given true labels and predicted probabilities.

    Parameters:
    y_true (array-like): True binary labels (0 or 1).
    y_pred (array-like): Predicted probabilities or scores.
    n_bins (int): Number of bins to use for calibration curve.

    Returns:
    None
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    # Plot calibration curve
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def get_auc_vs_fraction(
    df: pl.DataFrame,
    num_fractions: int = 10,
    frac_start: float = 0.01,
    frac_end: float = 1,
    num_fit_reps: int = 10,
) -> pl.DataFrame:
    """Computes the AUC for different fractions of the data."""
    res = []
    for frac in np.logspace(np.log10(frac_start), np.log10(frac_end), num=num_fractions):
        scores = [
            compute_fitted_metrics(*df.sample(fraction=frac, seed=i)["expected", "actual"])["auc"]
            for i in range(num_fit_reps)
        ]
        res.append(
            {
                "fraction": frac,
                "n_samples": round(len(df) * frac),
                "auc": np.median(scores),
            }
        )
    return pl.from_dicts(res)
