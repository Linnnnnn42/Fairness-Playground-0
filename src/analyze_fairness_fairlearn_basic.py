"""Fairlearn 基础版本，仅计算 Independence 和 Sufficiency。

运行：uv run python src/analyze_fairness_fairlearn_basic.py
使用内置 selection_rate、count 和 MetricFrame.difference，无自定义 metric。
"""

from math import isclose
from pathlib import Path

import pandas as pd
from fairlearn.metrics import MetricFrame, count, selection_rate


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "model_outcome.csv"
TOLERANCE = 0.05
ATTRIBUTES = {"a1": "Sex", "a2": "Age", "a3": "Ethnicity"}


def load_data(path):
    data = pd.read_csv(path, usecols=[*ATTRIBUTES, "y", "r"])
    if data.empty or not data.isin([0, 1]).all().all():
        raise ValueError("CSV must contain non-empty, complete binary data")
    return data.astype(int)


def compare_groups(data, attribute, decision=None):
    subset = data if decision is None else data.loc[data["r"] == decision]
    target = "r" if decision is None else "y"
    if subset.empty:
        print("    A= 0: 0/0 = undefined\n    A= 1: 0/0 = undefined")
        print("    Result: UNDEFINED (empty group)")
        return None

    # selection_rate 只统计 y_pred 中 1 的比例，忽略 y_true。
    # Independence：传 R，计算 P(R=1 | A)。
    # Sufficiency：先筛选 R=decision，再传 Y，计算 P(Y=1 | R=decision,A)。
    # 这里借用其比例计算能力，并非把真实 Y 当成模型的预测结果。
    frame = MetricFrame(
        metrics={"rate": selection_rate, "count": count},
        y_true=subset["y"], y_pred=subset[target],
        sensitive_features=subset[attribute],
    )
    groups = frame.by_group.reindex([0, 1])
    for group, row in groups.iterrows():
        if pd.isna(row["count"]):
            print(f"    A= {group}: 0/0 = undefined")
        else:
            total = int(row["count"])
            positives = round(row["rate"] * total)
            print(f"    A= {group}: {positives}/{total} = {row['rate']:.2%}")
    if groups["rate"].isna().any():
        print("    Result: UNDEFINED (empty group)")
        return None

    gap = float(frame.difference(method="between_groups")["rate"])
    # 只补偿浮点舍入误差，保证恰好相差 0.05 时通过。
    passed = gap <= TOLERANCE or isclose(gap, TOLERANCE, rel_tol=0, abs_tol=1e-12)
    print(f"    Gap: {gap * 100:.2f} percentage points; {'PASS' if passed else 'FAIL'}")
    return passed


def status(passed):
    return "UNDEFINED" if passed is None else "PASS" if passed else "FAIL"


def main():
    data = load_data(DATA_PATH)
    print(f"Data: {DATA_PATH}\nRows: {len(data)}")
    print(f"Tolerance: {TOLERANCE * 100:g} percentage points")
    results = []
    for attribute, label in ATTRIBUTES.items():
        print(f"\n{attribute.upper()} ({label})")
        print("  Independence: P(R=1 | A)")
        independence = compare_groups(data, attribute)
        conditional = []
        for decision in (0, 1):
            print(f"  Sufficiency: P(Y=1 | R={decision}, A)")
            conditional.append(compare_groups(data, attribute, decision))
        sufficiency = (False if False in conditional else
                       None if None in conditional else True)
        results.append((attribute.upper(), independence, sufficiency))
    print("\nSummary")
    print(f"{'Attribute':<12}{'Independence':<16}Sufficiency")
    for attribute, independence, sufficiency in results:
        print(f"{attribute:<12}{status(independence):<16}{status(sufficiency)}")


if __name__ == "__main__":
    main()
