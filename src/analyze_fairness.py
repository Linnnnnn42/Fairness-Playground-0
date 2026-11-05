"""计算 Independence 和 Sufficiency；仅使用 Python 标准库。

运行：uv run python src/analyze_fairness.py
容差为两组概率的绝对差 0.05，即 5 个百分点。
"""

import csv
from fractions import Fraction
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "model_outcome.csv"
TOLERANCE = Fraction(5, 100)
ATTRIBUTES = {"a1": "Sex", "a2": "Age", "a3": "Ethnicity"}


def load_data(path: Path) -> list[dict[str, int]]:
    """读取五个二元变量，忽略 CSV 中的行编号列。"""
    columns = (*ATTRIBUTES, "y", "r")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {columns}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if any(row[column] not in ("0", "1") for column in columns):
                raise ValueError(f"Line {line}: missing or non-binary value")
            rows.append({column: int(row[column]) for column in columns})
    if not rows:
        raise ValueError("CSV contains no data")
    return rows


def probability(rows, attribute, group, decision=None):
    """计算 P(R=1|A=group)，或指定 decision 后的 P(Y=1|R=decision,A=group)。"""
    subset = [
        row for row in rows
        if row[attribute] == group
        and (decision is None or row["r"] == decision)
    ]
    target = "r" if decision is None else "y"
    numerator = sum(row[target] for row in subset)
    denominator = len(subset)
    value = Fraction(numerator, denominator) if denominator else None
    return numerator, denominator, value


def compare_groups(rows, attribute, decision=None):
    """打印两组的计数和概率，并返回是否通过容差检查。"""
    probabilities = []
    for group in (0, 1):
        numerator, denominator, value = probability(rows, attribute, group, decision)
        probabilities.append(value)
        formatted = f"{float(value):.2%}" if value is not None else "undefined"
        print(f"    A= {group}: {numerator}/{denominator} = {formatted}")

    if any(value is None for value in probabilities):
        print("    Result: UNDEFINED (empty group)")
        return None
    gap = abs(probabilities[0] - probabilities[1])
    # 使用精确分数判断，避免 0.05 边界的浮点误差。
    passed = gap <= TOLERANCE
    print(f"    Gap: {float(gap) * 100:.2f} percentage points; "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def status(passed):
    return "UNDEFINED" if passed is None else "PASS" if passed else "FAIL"


def main():
    rows = load_data(DATA_PATH)
    print(f"Data: {DATA_PATH}\nRows: {len(rows)}")
    print(f"Tolerance: {float(TOLERANCE) * 100:g} percentage points")
    results = []
    for attribute, label in ATTRIBUTES.items():
        print(f"\n{attribute.upper()} ({label})")
        # Independence: R 与 A 独立。R 是二元变量，只需比较 R=1。
        print("  Independence: P(R=1 | A)")
        independence = compare_groups(rows, attribute)

        # Sufficiency: 给定 R 后 Y 与 A 独立，R=0 和 R=1 均需通过。
        conditional_results = []
        for decision in (0, 1):
            print(f"  Sufficiency: P(Y=1 | R={decision}, A)")
            conditional_results.append(compare_groups(rows, attribute, decision))
        sufficiency = (
            False if False in conditional_results
            else None if None in conditional_results
            else True
        )
        results.append((attribute.upper(), independence, sufficiency))

    print("\nSummary")
    print(f"{'Attribute':<12}{'Independence':<16}Sufficiency")
    for attribute, independence, sufficiency in results:
        print(f"{attribute:<12}{status(independence):<16}{status(sufficiency)}")


if __name__ == "__main__":
    main()
