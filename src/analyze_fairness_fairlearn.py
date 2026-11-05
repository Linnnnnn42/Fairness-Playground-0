"""使用 Fairlearn 评估三种公平性，并生成图表。

运行：uv run python src/analyze_fairness_fairlearn.py
可选：--data PATH --output-dir PATH --tolerance 0.05
输出：分组指标、判定表、Separation 柱状图及 ROC 空间决策点图。
R 只有 0/1，因此无法恢复连续分数的完整 ROC 曲线。
参考：https://fairlearn.org/v0.15/quickstart.html
"""

import argparse
from functools import partial
from math import isclose, nan
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 保存图片，无需打开图形窗口。
import matplotlib.pyplot as plt
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, count, false_positive_rate, selection_rate, true_positive_rate,
)
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "a1": ("Sex", ["Female", "Male"]),
    "a2": ("Age", ["Not elderly", "Elderly"]),
    "a3": ("Ethnicity", ["Majority", "Minority"]),
}
COLORS = ["#2166ac", "#d6604d"]


def conditional_outcome(y_true, y_pred, decision):
    """Sufficiency：P(Y=1 | R=decision)，空分母返回 NaN。"""
    outcomes = [y for y, r in zip(y_true, y_pred) if r == decision]
    return sum(outcomes) / len(outcomes) if outcomes else nan


def safe_error_rate(y_true, y_pred, positive):
    """没有真实正例/负例时，相应 TPR/FPR 未定义，不能记为零。"""
    if not any(y == positive for y in y_true):
        return nan
    metric = true_positive_rate if positive else false_positive_rate
    return metric(y_true, y_pred)


def confusion_count(y_true, y_pred, cell):
    return int(confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()[cell])


METRICS = {
    "count": count,
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "TPR": partial(safe_error_rate, positive=1),
    "FPR": partial(safe_error_rate, positive=0),
    "P(Y=1|R=0)": partial(conditional_outcome, decision=0),
    "P(Y=1|R=1)": partial(conditional_outcome, decision=1),
    **{name: partial(confusion_count, cell=i)
       for i, name in enumerate(["TN", "FP", "FN", "TP"])},
}
DEFINITIONS = {
    "Independence": ["selection_rate"],
    "Sufficiency": ["P(Y=1|R=0)", "P(Y=1|R=1)"],
    "Separation": ["TPR", "FPR"],
}


def load_data(path):
    data = pd.read_csv(path, usecols=[*GROUPS, "y", "r"])
    if data.empty or not data.isin([0, 1]).all().all():
        raise ValueError("Expected non-empty, complete binary columns a1,a2,a3,y,r")
    return data.astype(int)


def assess(data, tolerance=0.05):
    """各敏感属性分别评估；Separation 等价于二元决策的 Equalized Odds。"""
    tables, results = {}, []
    for attribute in GROUPS:
        frame = MetricFrame(metrics=METRICS, y_true=data.y, y_pred=data.r,
                            sensitive_features=data[attribute])
        table = frame.by_group.reindex([0, 1])
        tables[attribute] = table
        # difference() 默认比较各组最大值与最小值。
        gaps = frame.difference(method="between_groups")
        for definition, metrics in DEFINITIONS.items():
            # 不让默认的跳过 NaN 行为将缺失条件误判为公平。
            valid = [m for m in metrics if table[m].notna().all()]
            failed = any(gaps[m] > tolerance and
                         not isclose(gaps[m], tolerance, rel_tol=0, abs_tol=1e-12)
                         for m in valid)
            status = "FAIL" if failed else "PASS" if len(valid) == len(metrics) else "UNDEFINED"
            results.append({
                "attribute": attribute.upper(), "definition": definition,
                "max_gap": max(gaps[m] for m in metrics) if len(valid) == len(metrics) else nan,
                "status": status,
            })
    return tables, pd.DataFrame(results)


def plot_reports(tables, output_dir):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, (attribute, (title, names)) in zip(axes, GROUPS.items()):
        table = tables[attribute]
        for group, (name, color) in enumerate(zip(names, COLORS)):
            values = table.loc[group, ["TPR", "FPR"]].astype(float)
            bars = ax.bar([i + (group - 0.5) * 0.36 for i in range(2)], values,
                          width=0.36, color=color, label=name)
            ax.bar_label(bars, labels=[f"{v:.1%}" if pd.notna(v) else "N/A" for v in values],
                         padding=3, fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["TPR: P(R=1 | Y=1)", "FPR: P(R=1 | Y=0)"],
               ylim=(0, 1.15), title=f"{attribute.upper()}: {title}", ylabel="Rate")
        ax.legend(loc="upper right", frameon=False)
    fig.suptitle("Separation: compare both TPR and FPR across groups")
    fig.savefig(output_dir / "separation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for ax, (attribute, (title, names)) in zip(axes, GROUPS.items()):
        ax.plot([0, 1], [0, 1], "--", color="0.75", linewidth=1)
        for group, (name, color) in enumerate(zip(names, COLORS)):
            row = tables[attribute].loc[group]
            if pd.notna(row.FPR) and pd.notna(row.TPR):
                # 只画实际决策点，不虚构不存在的预测分数或阈值曲线。
                ax.scatter(row.FPR, row.TPR, s=90, color=color,
                           label=f"{name} (n={int(row['count'])})")
                ax.annotate(f"({row.FPR:.2f}, {row.TPR:.2f})", (row.FPR, row.TPR),
                            xytext=(5, -18 if group == 0 else 10),
                            textcoords="offset points", fontsize=9)
            else:
                ax.plot([], [], "o", color=color, label=f"{name}: undefined")
        ax.set(xlim=(-0.03, 1.08), ylim=(-0.03, 1.08), xlabel="False positive rate",
               ylabel="True positive rate", title=f"{attribute.upper()}: {title}", aspect="equal")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.grid(alpha=0.15)
    fig.suptitle("ROC space: observed operating points\nBinary R only; no continuous-score ROC curve available")
    fig.savefig(output_dir / "roc_points.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/model_outcome.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fairlearn")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.tolerance <= 1:
        parser.error("--tolerance must be between 0 and 1")
    data = load_data(args.data)
    tables, summary = assess(data, args.tolerance)
    print(f"Rows: {len(data)}; tolerance: {args.tolerance:.1%} absolute probability difference")
    for attribute, table in tables.items():
        print(f"\n{attribute.upper()} (groups 0, 1: {', '.join(GROUPS[attribute][1])})")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nFairness summary (max_gap is a probability difference):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, names=["attribute", "group"]).to_csv(args.output_dir / "group_metrics.csv")
    summary.to_csv(args.output_dir / "fairness_summary.csv", index=False)
    plot_reports(tables, args.output_dir)
    print(f"\nReports saved to {args.output_dir.resolve()}")
    print("ROC plot shows only the observed binary-decision points, not a score-based ROC curve.")


if __name__ == "__main__":
    main()
