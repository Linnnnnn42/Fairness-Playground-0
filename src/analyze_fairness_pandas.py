"""pandas 版本：uv run python src/analyze_fairness_pandas.py。

对二元变量求均值即得到取值为 1 的概率，容差为 5 个百分点。
"""

from fractions import Fraction
from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "model_outcome.csv"
TOLERANCE = Fraction(5, 100)
ATTRIBUTES = {"a1": "Sex", "a2": "Age", "a3": "Ethnicity"}


def load_data(path: Path) -> pd.DataFrame:
    columns = [*ATTRIBUTES, "y", "r"]
    data = pd.read_csv(path, usecols=columns)
    if data.empty or not data.isin([0, 1]).all().all():
        raise ValueError("CSV must contain non-empty, complete binary data")
    return data.astype(int)


def compare_groups(data, attribute, decision=None):
    """groupby 一次计算各组正例人数、总人数和概率。"""
    subset = data if decision is None else data.loc[data["r"] == decision]
    target = "r" if decision is None else "y"
    stats = subset.groupby(attribute)[target].agg(["sum", "size", "mean"])
    # 补齐缺失的组，避免缺少一个组时误判为公平。
    stats = stats.reindex([0, 1])
    for group, row in stats.iterrows():
        if pd.isna(row["size"]):
            print(f"    A= {group}: 0/0 = undefined")
        else:
            print(f"    A= {group}: {int(row['sum'])}/{int(row['size'])} "
                  f"= {row['mean']:.2%}")
    if stats["mean"].isna().any():
        print("    Result: UNDEFINED (empty group)")
        return None
    # 从计数构造精确分数，保证恰好相差 5 个百分点时通过。
    rates = [Fraction(int(row['sum']), int(row['size']))
             for _, row in stats.iterrows()]
    gap = abs(rates[0] - rates[1])
    passed = gap <= TOLERANCE
    print(f"    Gap: {float(gap) * 100:.2f} percentage points; "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def status(passed):
    return "UNDEFINED" if passed is None else "PASS" if passed else "FAIL"


def main():
    data = load_data(DATA_PATH)
    print(f"Data: {DATA_PATH}\nRows: {len(data)}")
    print(f"Tolerance: {float(TOLERANCE) * 100:g} percentage points")
    results = []
    for attribute, label in ATTRIBUTES.items():
        print(f"\n{attribute.upper()} ({label})")
        print("  Independence: P(R=1 | A)")
        independence = compare_groups(data, attribute)
        conditional = []
        for decision in (0, 1):
            print(f"  Sufficiency: P(Y=1 | R={decision}, A)")
            conditional.append(compare_groups(data, attribute, decision))
        # Sufficiency 要求 R=0 和 R=1 两种条件都通过。
        sufficiency = (False if False in conditional else
                       None if None in conditional else True)
        results.append((attribute.upper(), independence, sufficiency))
    print("\nSummary")
    print(f"{'Attribute':<12}{'Independence':<16}Sufficiency")
    for attribute, independence, sufficiency in results:
        print(f"{attribute:<12}{status(independence):<16}{status(sufficiency)}")


if __name__ == "__main__":
    main()
