#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build a common classification trace from several UCI datasets.

The output is intentionally a plain CSV so Rust examples and other tools can
replay the same observations without pulling a Python dependency into muxer.
Every fifth row trains three fixed policies; the remaining rows become trace
records for majority, best-single-feature, and categorical naive-Bayes arms.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    label: str
    features: tuple[str, ...]


def read_delimited(path: Path, delimiter: str, label_index: int = -1) -> list[Record]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter=delimiter):
            row = [field.strip() for field in row]
            if not row or len(row) < 2:
                continue
            label = row[label_index].rstrip(".")
            if label in {"y", "quality"} and row[0].lower() in {"age", "fixed acidity"}:
                continue
            features = tuple(row[:label_index] if label_index == -1 else row[:label_index] + row[label_index + 1 :])
            if label and all(features):
                records.append(Record(label, features))
    return records


def load_mushroom(root: Path) -> list[Record]:
    path = next(root.rglob("agaricus-lepiota.data"))
    return read_delimited(path, ",", 0)


def load_car(root: Path) -> list[Record]:
    path = next(root.rglob("car.data"))
    return read_delimited(path, ",")


def load_bank(root: Path) -> list[Record]:
    path = next(root.rglob("bank-full.csv"))
    return read_delimited(path, ";")


def load_wine(root: Path, variant: str) -> list[Record]:
    path = next(root.rglob(f"winequality-{variant}.csv"))
    return read_delimited(path, ";")


def load_adult(root: Path) -> list[Record]:
    records: list[Record] = []
    for path in sorted(root.rglob("adult.data")) + sorted(root.rglob("adult.test")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 15 or not fields[0] or fields[-1].startswith("|"):
                    continue
                label = fields[-1].rstrip(".")
                features = tuple(fields[:-1])
                if all(features):
                    records.append(Record(label, features))
    return records


def as_number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_features(train: list[Record], rows: list[Record]) -> list[Record]:
    if not train:
        return rows
    columns = len(train[0].features)
    thresholds: list[tuple[float, ...] | None] = []
    for column in range(columns):
        values = [as_number(row.features[column]) for row in train]
        if any(value is None for value in values) or len(set(values)) < 8:
            thresholds.append(None)
            continue
        numeric = sorted(value for value in values if value is not None)
        thresholds.append(tuple(numeric[len(numeric) * q // 4] for q in (1, 2, 3)))

    normalized: list[Record] = []
    for row in rows:
        features: list[str] = []
        for column, raw in enumerate(row.features):
            cuts = thresholds[column]
            number = as_number(raw) if cuts is not None else None
            if cuts is None or number is None:
                features.append(raw)
            elif number <= cuts[0]:
                features.append("bin0")
            elif number <= cuts[1]:
                features.append("bin1")
            elif number <= cuts[2]:
                features.append("bin2")
            else:
                features.append("bin3")
        normalized.append(Record(row.label, tuple(features)))
    return normalized


class Models:
    def __init__(self, train: list[Record]) -> None:
        labels = Counter(row.label for row in train)
        self.majority = labels.most_common(1)[0][0]
        self.classes = tuple(sorted(labels))
        self.feature_maps: list[dict[str, str]] = []
        self.feature_counts: list[dict[str, Counter[str]]] = []
        self.cardinalities: list[int] = []

        for column in range(len(train[0].features)):
            values = sorted({row.features[column] for row in train})
            self.cardinalities.append(max(1, len(values)))
            counts: dict[str, Counter[str]] = defaultdict(Counter)
            for row in train:
                counts[row.features[column]][row.label] += 1
            mapping = {
                value: max(self.classes, key=lambda label: (counts[value][label], label))
                for value in values
            }
            self.feature_maps.append(mapping)
            self.feature_counts.append(counts)

        self.best_feature = max(range(len(self.feature_maps)), key=lambda column: self.feature_accuracy(column, train))

    def feature_accuracy(self, column: int, rows: list[Record]) -> float:
        mapping = self.feature_maps[column]
        correct = sum(mapping.get(row.features[column], self.majority) == row.label for row in rows)
        return correct / max(1, len(rows))

    def predict_feature(self, row: Record) -> str:
        return self.feature_maps[self.best_feature].get(row.features[self.best_feature], self.majority)

    def predict_bayes(self, row: Record) -> str:
        total = sum(self.feature_counts[0][value][label] for value in self.feature_counts[0] for label in self.classes)
        scores: dict[str, float] = {}
        for label in self.classes:
            class_count = sum(counts[label] for counts in self.feature_counts[0].values())
            score = math.log((class_count + 1) / (total + len(self.classes)))
            for column, value in enumerate(row.features):
                counts = self.feature_counts[column].get(value, Counter())
                numerator = counts[label] + 1
                denominator = class_count + self.cardinalities[column]
                score += math.log(numerator / denominator)
            scores[label] = score
        return max(self.classes, key=lambda label: (scores[label], label))


def iter_datasets(root: Path) -> list[tuple[str, list[Record], set[str]]]:
    datasets: list[tuple[str, list[Record], set[str]]] = []
    loaders = [
        ("mushroom", load_mushroom, {"p"}),
        ("car", load_car, set()),
        ("bank", load_bank, set()),
        ("wine_red", lambda path: load_wine(path, "red"), set()),
        ("wine_white", lambda path: load_wine(path, "white"), set()),
        ("adult", load_adult, set()),
    ]
    for name, loader, hard_labels in loaders:
        try:
            rows = loader(root)
        except StopIteration:
            print(f"skip {name}: source file not found")
            continue
        if len(rows) < 20:
            print(f"skip {name}: only {len(rows)} valid rows")
            continue
        datasets.append((name, rows, hard_labels))
    if not datasets:
        raise SystemExit(f"no supported datasets found under {root}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/uci"))
    parser.add_argument("--output", type=Path, default=Path("data/traces/classification-traces.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "row_id",
                "arm",
                "label",
                "predicted",
                "ok",
                "junk",
                "hard_junk",
                "quality_score",
                "cost_units",
                "elapsed_ms",
            ],
        )
        writer.writeheader()
        for name, raw_rows, hard_labels in iter_datasets(args.input):
            train = [row for index, row in enumerate(raw_rows) if index % 5 == 0]
            eval_rows = [row for index, row in enumerate(raw_rows) if index % 5 != 0]
            rows = normalize_features(train, raw_rows)
            train = [row for index, row in enumerate(rows) if index % 5 == 0]
            eval_rows = [row for index, row in enumerate(rows) if index % 5 != 0]
            models = Models(train)
            policies = (
                ("majority", lambda row: models.majority, 1, 2),
                ("feature", models.predict_feature, 3, 8),
                ("naive_bayes", models.predict_bayes, 10, 24 + len(train[0].features) * 2),
            )
            for row_id, row in enumerate(eval_rows):
                for arm, predictor, cost, elapsed_ms in policies:
                    predicted = predictor(row)
                    ok = predicted == row.label
                    hard_junk = not ok and row.label in hard_labels
                    writer.writerow(
                        {
                            "dataset": name,
                            "row_id": row_id,
                            "arm": arm,
                            "label": row.label,
                            "predicted": predicted,
                            "ok": int(ok),
                            "junk": int(not ok),
                            "hard_junk": int(hard_junk),
                            "quality_score": int(ok),
                            "cost_units": cost,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
            print(f"built {name}: train={len(train)} eval={len(eval_rows)}")


if __name__ == "__main__":
    main()
