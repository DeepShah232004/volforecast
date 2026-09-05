"""Build paper-ready tables and figures from the two locked result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = PROJECT_ROOT / "evaluation/results/final_evaluation.json"
SHAP_PATH = PROJECT_ROOT / "interpretability/results/shap_summary.json"
TABLE_PATH = PROJECT_ROOT / "reporting/tables/results_tables.md"
FIGURE_DIR = PROJECT_ROOT / "reporting/figures"

EXPECTED_EVALUATION_SHA256 = (
    "49d4ff213ddf871678e378f0c048860edfa528e563ccdc24c310ce02fe179c0d"
)
EXPECTED_SHAP_SHA256 = (
    "89aa3ac576866de9f8ff97d9de1128355e8873ca1ba9910a384144032227e98e"
)

MODEL_ORDER = ["persistence", "ewma", "garch", "random_forest", "xgboost"]
ML_MODEL_ORDER = ["random_forest", "xgboost"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "ewma": "EWMA",
    "garch": "GARCH(1,1)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {
    "persistence": "#8A8F98",
    "ewma": "#52788F",
    "garch": "#16324F",
    "random_forest": "#2A9D8F",
    "xgboost": "#E76F51",
}
GROUP_ORDER = [
    "trailing_volatility",
    "vix_changes",
    "compounded_returns",
    "vix_level",
]
GROUP_LABELS = {
    "trailing_volatility": "Trailing volatility",
    "vix_changes": "VIX changes",
    "compounded_returns": "Compounded returns",
    "vix_level": "VIX level",
}
FEATURE_LABELS = {
    "return_std_5d": "Return stdev (5d)",
    "return_std_21d": "Return stdev (21d)",
    "return_std_63d": "Return stdev (63d)",
    "return_std_126d": "Return stdev (126d)",
    "return_std_252d": "Return stdev (252d)",
    "mean_abs_return_5d": "Mean |return| (5d)",
    "mean_abs_return_21d": "Mean |return| (21d)",
    "mean_abs_return_63d": "Mean |return| (63d)",
    "compounded_return_5d": "Compounded return (5d)",
    "compounded_return_21d": "Compounded return (21d)",
    "compounded_return_63d": "Compounded return (63d)",
    "ewma_volatility_094": "EWMA volatility",
    "vix_close": "VIX level",
    "vix_log_change_1d": "VIX change (1d)",
    "vix_log_change_5d": "VIX change (5d)",
    "vix_log_change_21d": "VIX change (21d)",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_locked_results() -> tuple[dict, dict]:
    expected = {
        EVALUATION_PATH: EXPECTED_EVALUATION_SHA256,
        SHAP_PATH: EXPECTED_SHAP_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"Locked result artifact is missing: {path}")
        observed_hash = file_sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"{path.name} hash is {observed_hash}; expected {expected_hash}."
            )
    evaluation = read_json(EVALUATION_PATH)
    shap = read_json(SHAP_PATH)
    if evaluation.get("schema_version") != 1 or shap.get("schema_version") != 1:
        raise ValueError("Unsupported locked-result schema version.")
    return evaluation, shap


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Markdown table row has the wrong number of cells.")
    escaped_headers = [header.replace("|", "\\|") for header in headers]
    escaped_rows = [
        [cell.replace("|", "\\|") for cell in row]
        for row in rows
    ]
    return [
        "| " + " | ".join(escaped_headers) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
        *("| " + " | ".join(row) + " |" for row in escaped_rows),
    ]


def interval(record: dict, digits: int = 6, scale: float = 1.0) -> str:
    estimate = scale * record["estimate"]
    lower = scale * record["ci_lower"]
    upper = scale * record["ci_upper"]
    return f"{estimate:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def build_markdown(evaluation: dict, shap: dict) -> str:
    overall = evaluation["metric_groups"]["overall"]["models"]
    lines = [
        "# Locked Results Tables",
        "",
        "Generated only from the committed final-evaluation and SHAP summary artifacts.",
        "Lower values are better for RMSE, MAE, and QLIKE.",
        "",
        "## Table 1. Final common test sample",
        "",
    ]
    sample = evaluation["sample"]
    sample_rows = [
        ["Security-date forecasts", f"{sample['rows']:,}"],
        ["Securities", f"{sample['permnos']:,}"],
        ["Forecast-origin dates", f"{sample['dates']:,}"],
        ["Origin range", f"{sample['origin_start']} to {sample['origin_end']}"],
        ["SHAP sample rows", f"{shap['sample']['rows']:,}"],
        ["SHAP sample securities", f"{shap['sample']['permnos']:,}"],
    ]
    lines.extend(markdown_table(["Quantity", "Value"], sample_rows))

    lines.extend(["", "## Table 2. Overall out-of-sample performance", ""])
    overall_rows: list[list[str]] = []
    garch = overall["garch"]
    for model in MODEL_ORDER:
        metrics = overall[model]
        rmse_change = 100 * (
            garch["rmse"]["estimate"] - metrics["rmse"]["estimate"]
        ) / garch["rmse"]["estimate"]
        qlike_change = 100 * (
            garch["qlike"]["estimate"] - metrics["qlike"]["estimate"]
        ) / garch["qlike"]["estimate"]
        overall_rows.append(
            [
                MODEL_LABELS[model],
                interval(metrics["rmse"]),
                interval(metrics["mae"]),
                interval(metrics["qlike"]),
                "—" if model == "garch" else f"{rmse_change:+.2f}%",
                "—" if model == "garch" else f"{qlike_change:+.2f}%",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Model",
                "RMSE [95% CI]",
                "MAE [95% CI]",
                "QLIKE [95% CI]",
                "RMSE vs GARCH",
                "QLIKE vs GARCH",
            ],
            overall_rows,
        )
    )
    lines.extend(
        [
            "",
            "Positive relative changes indicate improvement over GARCH. Overall changes are descriptive; no paired overall superiority test was preregistered.",
            "",
            "## Table 3. Pooled QLIKE by VIX regime",
            "",
        ]
    )
    regime_rows = []
    for model in MODEL_ORDER:
        regime_rows.append(
            [
                MODEL_LABELS[model],
                f"{evaluation['metric_groups']['vix_low']['models'][model]['qlike']['estimate']:.6f}",
                f"{evaluation['metric_groups']['vix_middle']['models'][model]['qlike']['estimate']:.6f}",
                f"{evaluation['metric_groups']['vix_high']['models'][model]['qlike']['estimate']:.6f}",
            ]
        )
    lines.extend(markdown_table(["Model", "Low VIX", "Middle VIX", "High VIX"], regime_rows))
    lines.extend(
        [
            "",
            "These pooled subgroup losses are descriptive. H1 and H2 decisions use the preregistered date-equal paired differences below.",
            "",
            "## Table 4. Regime hypothesis tests against GARCH",
            "",
        ]
    )
    h1 = evaluation["hypotheses"]["H1"]
    h2 = evaluation["hypotheses"]["H2"]
    regime_test_rows = []
    for model in ML_MODEL_ORDER:
        key = f"{model}_vs_garch"
        high = h1["comparisons"][key]
        low = h2["comparisons"][key]
        regime_test_rows.append(
            [
                MODEL_LABELS[model],
                f"{high['daily_qlike_improvement']:.5f} [{high['ci_lower']:.5f}, {high['ci_upper']:.5f}]",
                f"{high['newey_west']['two_sided_p_value']:.3f}",
                f"{100 * low['relative_ml_qlike_improvement']:.2f}% [{100 * low['ci_lower']:.2f}%, {100 * low['ci_upper']:.2f}%]",
                f"{low['newey_west_absolute_difference']['two_sided_p_value']:.3f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "ML model",
                "High-VIX QLIKE advantage [95% CI]",
                "NW p",
                "Low-VIX relative advantage [95% CI]",
                "NW p",
            ],
            regime_test_rows,
        )
    )
    lines.extend(
        [
            "",
            f"H1 decision: **{h1['status'].replace('_', ' ')}**. H2 decision: **{h2['status'].replace('_', ' ')} under the locked rule**.",
            "",
            "## Table 5. Earnings-window QLIKE penalty",
            "",
        ]
    )
    h3 = evaluation["hypotheses"]["H3"]
    earnings_rows = []
    for model in MODEL_ORDER:
        result = h3["models"][model]
        earnings_rows.append(
            [
                MODEL_LABELS[model],
                f"{result['earnings_minus_non_earnings_daily_qlike']:.5f}",
                f"[{result['ci_lower']:.5f}, {result['ci_upper']:.5f}]",
                f"{result['newey_west']['two_sided_p_value']:.4g}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Model", "Date-matched penalty", "95% block-bootstrap CI", "NW p"],
            earnings_rows,
        )
    )
    lines.extend(
        [
            "",
            f"H3 decision: **{h3['status'].replace('_', ' ')}**. Positive values mean higher QLIKE for earnings-window forecasts.",
            "",
            "## Table 6. Grouped SHAP importance",
            "",
        ]
    )
    group_maps = {
        model: {
            record["group"]: record
            for record in shap["models"][model]["segments"]["overall"]["groups"]
        }
        for model in ML_MODEL_ORDER
    }
    group_rows = []
    for group in GROUP_ORDER:
        group_rows.append(
            [
                GROUP_LABELS[group],
                f"{100 * group_maps['random_forest'][group]['normalized_share']:.2f}%",
                f"{100 * group_maps['xgboost'][group]['normalized_share']:.2f}%",
            ]
        )
    lines.extend(markdown_table(["Feature group", "Random Forest", "XGBoost"], group_rows))
    lines.extend(["", "## Table 7. Leading individual SHAP features", ""])
    feature_rows = []
    for model in ML_MODEL_ORDER:
        for record in shap["models"][model]["segments"]["overall"]["features"][:5]:
            feature_rows.append(
                [
                    MODEL_LABELS[model],
                    str(record["rank"]),
                    FEATURE_LABELS[record["feature"]],
                    f"{100 * record['normalized_share']:.2f}%",
                ]
            )
    lines.extend(markdown_table(["Model", "Rank", "Feature", "Importance share"], feature_rows))
    lines.extend(["", "## Table 8. SHAP stability across subperiods", ""])
    stability_rows = []
    for model in ML_MODEL_ORDER:
        stability = shap["models"][model]["stability"]
        stability_rows.append(
            [
                MODEL_LABELS[model],
                f"{stability['feature_rank_spearman']:.3f}",
                str(len(stability["feature_top_5_overlap"]["shared"])),
                f"{stability['group_rank_spearman']:.3f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Model", "Feature-rank Spearman", "Shared top-five features", "Group-rank Spearman"],
            stability_rows,
        )
    )
    lines.extend(
        [
            "",
            "SHAP values explain log-volatility predictions. Importance is descriptive, and individual rankings among correlated volatility measures should not be interpreted causally.",
            "",
            f"Source SHA-256 (evaluation): `{EXPECTED_EVALUATION_SHA256}`  ",
            f"Source SHA-256 (SHAP): `{EXPECTED_SHAP_SHA256}`",
            "",
        ]
    )
    return "\n".join(lines)


def configure_plotting():
    cache = Path(tempfile.gettempdir()) / "volforecast-matplotlib-cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#8A8F98",
            "axes.linewidth": 0.7,
            "xtick.color": "#343A40",
            "ytick.color": "#343A40",
            "text.color": "#1F2933",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def save_figure(fig, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "volforecast reporting pipeline"},
    )
    fig.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "volforecast reporting pipeline",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def figure_overall_metrics(evaluation: dict, plt) -> None:
    metrics = ["rmse", "mae", "qlike"]
    titles = ["RMSE", "MAE", "QLIKE (primary)"]
    overall = evaluation["metric_groups"]["overall"]["models"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    y = list(range(len(MODEL_ORDER)))
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        estimates = [overall[model][metric]["estimate"] for model in MODEL_ORDER]
        lower = [overall[model][metric]["ci_lower"] for model in MODEL_ORDER]
        upper = [overall[model][metric]["ci_upper"] for model in MODEL_ORDER]
        errors = [
            [estimate - bound for estimate, bound in zip(estimates, lower, strict=True)],
            [bound - estimate for estimate, bound in zip(estimates, upper, strict=True)],
        ]
        axis.barh(
            y,
            estimates,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            alpha=0.92,
            height=0.62,
        )
        axis.errorbar(
            estimates,
            y,
            xerr=errors,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.8,
            capsize=2.5,
        )
        axis.set_title(title)
        axis.set_xlim(left=0)
        axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axes[0].invert_yaxis()
    fig.suptitle("Overall out-of-sample forecast accuracy", fontsize=15, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Lower is better. Whiskers are 95% moving-block bootstrap intervals over forecast dates.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "figure_1_overall_metrics")
    plt.close(fig)


def figure_regime_tests(evaluation: dict, plt) -> None:
    h1 = evaluation["hypotheses"]["H1"]["comparisons"]
    h2 = evaluation["hypotheses"]["H2"]["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), sharey=True)
    y = [0, 1]
    labels = [MODEL_LABELS[model] for model in ML_MODEL_ORDER]

    high = [h1[f"{model}_vs_garch"] for model in ML_MODEL_ORDER]
    high_est = [record["daily_qlike_improvement"] for record in high]
    high_error = [
        [estimate - record["ci_lower"] for estimate, record in zip(high_est, high, strict=True)],
        [record["ci_upper"] - estimate for estimate, record in zip(high_est, high, strict=True)],
    ]
    axes[0].axvline(0, color="#343A40", linewidth=1)
    for position, model, estimate, low_error, upper_error in zip(
        y, ML_MODEL_ORDER, high_est, high_error[0], high_error[1], strict=True
    ):
        axes[0].errorbar(
            estimate,
            position,
            xerr=[[low_error], [upper_error]],
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=4,
            markersize=6,
        )
    axes[0].set_title("H1: High-VIX QLIKE advantage")
    axes[0].set_xlabel("GARCH loss − ML loss")

    low = [h2[f"{model}_vs_garch"] for model in ML_MODEL_ORDER]
    low_est = [100 * record["relative_ml_qlike_improvement"] for record in low]
    low_error = [
        [estimate - 100 * record["ci_lower"] for estimate, record in zip(low_est, low, strict=True)],
        [100 * record["ci_upper"] - estimate for estimate, record in zip(low_est, low, strict=True)],
    ]
    axes[1].axvline(0, color="#343A40", linewidth=1)
    axes[1].axvline(5, color="#8A8F98", linewidth=1, linestyle="--")
    for position, model, estimate, lower_error, upper_error in zip(
        y, ML_MODEL_ORDER, low_est, low_error[0], low_error[1], strict=True
    ):
        axes[1].errorbar(
            estimate,
            position,
            xerr=[[lower_error], [upper_error]],
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=4,
            markersize=6,
        )
    axes[1].set_title("H2: Low-VIX relative advantage")
    axes[1].set_xlabel("ML improvement over GARCH (%)")
    axes[1].text(
        5.25,
        0.5,
        "5% bound",
        transform=axes[1].get_xaxis_transform(),
        ha="left",
        va="center",
        rotation=90,
        fontsize=8,
        color="#68707A",
    )

    for axis in axes:
        axis.set_yticks(y, labels)
        axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Preregistered regime comparisons", fontsize=15, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.03,
        "Points are date-equal estimates; whiskers are 95% moving-block bootstrap intervals. Positive values favor ML.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "figure_2_regime_tests")
    plt.close(fig)


def figure_earnings_penalty(evaluation: dict, plt) -> None:
    h3 = evaluation["hypotheses"]["H3"]["models"]
    estimates = [h3[model]["earnings_minus_non_earnings_daily_qlike"] for model in MODEL_ORDER]
    errors = [
        [estimate - h3[model]["ci_lower"] for estimate, model in zip(estimates, MODEL_ORDER, strict=True)],
        [h3[model]["ci_upper"] - estimate for estimate, model in zip(estimates, MODEL_ORDER, strict=True)],
    ]
    y = list(range(len(MODEL_ORDER)))
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    axis.axvline(0, color="#343A40", linewidth=1)
    for position, model, estimate, low_error, high_error in zip(
        y, MODEL_ORDER, estimates, errors[0], errors[1], strict=True
    ):
        axis.errorbar(
            estimate,
            position,
            xerr=[[low_error], [high_error]],
            fmt="o",
            color=MODEL_COLORS[model],
            ecolor=MODEL_COLORS[model],
            capsize=4,
            markersize=7,
        )
    axis.set_yticks(y, [MODEL_LABELS[model] for model in MODEL_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Earnings-window QLIKE − non-earnings QLIKE")
    axis.set_title("Forecast accuracy degrades around earnings announcements", pad=14)
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        -0.01,
        "Within-date differences control for the market environment. Whiskers are 95% moving-block bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "figure_3_earnings_penalty")
    plt.close(fig)


def figure_shap_groups(shap: dict, plt) -> None:
    maps = {
        model: {
            record["group"]: 100 * record["normalized_share"]
            for record in shap["models"][model]["segments"]["overall"]["groups"]
        }
        for model in ML_MODEL_ORDER
    }
    y = list(range(len(GROUP_ORDER)))
    height = 0.34
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    random_forest_bars = axis.barh(
        [position - height / 2 for position in y],
        [maps["random_forest"][group] for group in GROUP_ORDER],
        height=height,
        color=MODEL_COLORS["random_forest"],
        label="Random Forest",
    )
    xgboost_bars = axis.barh(
        [position + height / 2 for position in y],
        [maps["xgboost"][group] for group in GROUP_ORDER],
        height=height,
        color=MODEL_COLORS["xgboost"],
        label="XGBoost",
    )
    axis.bar_label(random_forest_bars, fmt="%.1f%%", padding=3, fontsize=8)
    axis.bar_label(xgboost_bars, fmt="%.1f%%", padding=3, fontsize=8)
    axis.set_yticks(y, [GROUP_LABELS[group] for group in GROUP_ORDER])
    axis.invert_yaxis()
    axis.set_xlabel("Share of total mean absolute SHAP value (%)")
    axis.set_title("Both ML models are dominated by trailing volatility information", pad=14)
    axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)
    axis.legend(frameon=False, loc="lower right")
    fig.text(
        0.5,
        -0.01,
        "SHAP values explain log-volatility predictions; grouped shares are descriptive, not causal.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "figure_4_shap_groups")
    plt.close(fig)


def figure_shap_stability(shap: dict, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    period_colors = {"2015_2019": "#16324F", "2020_2024": "#E76F51"}
    period_labels = {"2015_2019": "2015–2019", "2020_2024": "2020–2024"}
    for axis, model in zip(axes, ML_MODEL_ORDER, strict=True):
        overall = shap["models"][model]["segments"]["overall"]["features"]
        selected = [record["feature"] for record in overall[:8]]
        period_maps = {
            period: {
                record["feature"]: 100 * record["normalized_share"]
                for record in shap["models"][model]["segments"][period]["features"]
            }
            for period in period_colors
        }
        y = list(range(len(selected)))
        for position, feature in zip(y, selected, strict=True):
            axis.plot(
                [period_maps[period][feature] for period in period_colors],
                [position, position],
                color="#C7CDD3",
                linewidth=1.2,
                zorder=1,
            )
        for period, color in period_colors.items():
            axis.scatter(
                [period_maps[period][feature] for feature in selected],
                y,
                color=color,
                s=32,
                label=period_labels[period],
                zorder=2,
            )
        rho = shap["models"][model]["stability"]["feature_rank_spearman"]
        axis.set_yticks(y, [FEATURE_LABELS[feature] for feature in selected])
        axis.invert_yaxis()
        axis.set_xlabel("Normalized mean absolute SHAP share (%)")
        axis.set_title(f"{MODEL_LABELS[model]}  |  rank ρ = {rho:.3f}")
        axis.grid(axis="x", color="#D9DEE3", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Feature importance is stable across test subperiods", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "figure_5_shap_stability")
    plt.close(fig)


def build_assets(evaluation: dict, shap: dict) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TABLE_PATH.with_suffix(".md.tmp")
    temporary.write_text(build_markdown(evaluation, shap), encoding="utf-8")
    temporary.replace(TABLE_PATH)

    plt = configure_plotting()
    figure_overall_metrics(evaluation, plt)
    figure_regime_tests(evaluation, plt)
    figure_earnings_penalty(evaluation, plt)
    figure_shap_groups(shap, plt)
    figure_shap_stability(shap, plt)


def main() -> None:
    evaluation, shap = load_locked_results()
    build_assets(evaluation, shap)
    print(f"Saved tables to {TABLE_PATH}")
    print(f"Saved five PNG and five PDF figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
