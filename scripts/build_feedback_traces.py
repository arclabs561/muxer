#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build separate native-schema traces for real-data muxer validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from fetch_feedback_datasets import Source, load_sources, sha256_path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUN_STATUSES = {"ok", "timeout", "memout", "not_applicable", "crash", "other"}


@dataclass(frozen=True)
class AslibScenario:
    name: str
    source_id: str
    measure: str
    objective: str


@dataclass(frozen=True)
class NabSeries:
    name: str
    source_id: str
    label_key: str


ASLIB_SCENARIOS = (
    AslibScenario(
        "CSP-Minizinc-Time-2016",
        "aslib_csp_runs",
        "PAR10",
        "minimize",
    ),
    AslibScenario(
        "OPENML-WEKA-2017",
        "aslib_openml_runs",
        "predictive_accuracy",
        "maximize",
    ),
)

NAB_SERIES = (
    NabSeries(
        "aws_cpu",
        "nab_aws_cpu",
        "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    ),
    NabSeries(
        "nyc_taxi",
        "nab_nyc_taxi",
        "realKnownCause/nyc_taxi.csv",
    ),
    NabSeries(
        "traffic_speed",
        "nab_traffic_speed",
        "realTraffic/speed_6005.csv",
    ),
    NabSeries(
        "twitter_aapl",
        "nab_twitter_aapl",
        "realTweets/Twitter_volume_AAPL.csv",
    ),
    NabSeries(
        "ad_exchange_cpc",
        "nab_ad_exchange_cpc",
        "realAdExchange/exchange-2_cpc_results.csv",
    ),
)


def source_path(root: Path, source: Source) -> Path:
    return root.joinpath(*source.path.parts)


def verify_sources(root: Path, sources: list[Source]) -> None:
    for source in sources:
        path = source_path(root, source)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing source {source.source_id}: {path}; run "
                "scripts/fetch_feedback_datasets.py"
            )
        actual_size = path.stat().st_size
        actual_digest = sha256_path(path)
        if actual_size != source.size or actual_digest != source.sha256:
            raise ValueError(
                f"source {source.source_id} failed verification: expected "
                f"{source.size} bytes and {source.sha256}, found {actual_size} "
                f"bytes and {actual_digest}"
            )


@contextmanager
def atomic_csv(path: Path, fieldnames: list[str]) -> Iterator[csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            yield writer
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def require_columns(actual: list[str] | None, required: set[str], source: str) -> None:
    missing = required - set(actual or [])
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{source} is missing required columns: {names}")


def build_open_bandit(
    raw_root: Path,
    output: Path,
    sources: dict[str, Source],
) -> dict[str, object]:
    fieldnames = [
        "behavior",
        "row_id",
        "timestamp",
        "item_id",
        "position",
        "reward",
        "logging_propensity",
        "target_propensity",
    ]
    statistics: dict[str, dict[str, object]] = {}
    total_rows = 0
    with atomic_csv(output, fieldnames) as writer:
        for behavior, source_id in (
            ("random", "obd_random_all"),
            ("bts", "obd_bts_all"),
        ):
            source = sources[source_id]
            clicks = 0
            propensity_min = math.inf
            propensity_max = 0.0
            rows = 0
            with source_path(raw_root, source).open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                require_columns(
                    reader.fieldnames,
                    {
                        "timestamp",
                        "item_id",
                        "position",
                        "click",
                        "propensity_score",
                    },
                    source_id,
                )
                for row_id, row in enumerate(reader):
                    item_id = int(row["item_id"])
                    position = int(row["position"])
                    reward = int(row["click"])
                    propensity = float(row["propensity_score"])
                    if not 0 <= item_id < 80:
                        raise ValueError(f"{source_id} row {row_id}: invalid item_id")
                    if position not in {1, 2, 3} or reward not in {0, 1}:
                        raise ValueError(f"{source_id} row {row_id}: invalid outcome")
                    if not math.isfinite(propensity) or not 0.0 < propensity <= 1.0:
                        raise ValueError(
                            f"{source_id} row {row_id}: invalid propensity"
                        )
                    target_propensity = 1.0 / 40.0 if item_id < 40 else 0.0
                    writer.writerow(
                        {
                            "behavior": behavior,
                            "row_id": row_id,
                            "timestamp": row["timestamp"],
                            "item_id": item_id,
                            "position": position,
                            "reward": reward,
                            "logging_propensity": format(propensity, ".12g"),
                            "target_propensity": format(target_propensity, ".12g"),
                        }
                    )
                    clicks += reward
                    propensity_min = min(propensity_min, propensity)
                    propensity_max = max(propensity_max, propensity)
                    rows += 1
            if rows == 0:
                raise ValueError(f"{source_id} contains no rows")
            statistics[behavior] = {
                "rows": rows,
                "clicks": clicks,
                "click_rate": clicks / rows,
                "logging_propensity_min": propensity_min,
                "logging_propensity_max": propensity_max,
                "target_policy": "uniform over item_id 0 through 39",
            }
            total_rows += rows
    return {"rows": total_rows, "by_behavior": statistics}


def iter_arff_rows(path: Path) -> Iterator[list[str]]:
    in_data = False
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if in_data and not line.startswith("@"):
                yield next(csv.reader([line]))


def build_aslib(
    raw_root: Path,
    output: Path,
    sources: dict[str, Source],
) -> dict[str, object]:
    fieldnames = [
        "scenario",
        "instance_id",
        "algorithm",
        "repetition",
        "value",
        "objective",
        "runstatus",
    ]
    total_rows = 0
    by_scenario: dict[str, dict[str, object]] = {}
    with atomic_csv(output, fieldnames) as writer:
        for scenario in ASLIB_SCENARIOS:
            source = sources[scenario.source_id]
            statuses: Counter[str] = Counter()
            algorithms: set[str] = set()
            instances: set[str] = set()
            keys: set[tuple[str, str, str]] = set()
            rows = 0
            for row_id, row in enumerate(iter_arff_rows(source_path(raw_root, source))):
                if len(row) != 5:
                    raise ValueError(
                        f"{scenario.source_id} row {row_id}: expected 5 fields"
                    )
                instance_id, repetition, algorithm, raw_value, runstatus = row
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError(
                        f"{scenario.source_id} row {row_id}: invalid {scenario.measure}"
                    )
                if runstatus not in RUN_STATUSES:
                    raise ValueError(
                        f"{scenario.source_id} row {row_id}: invalid runstatus"
                    )
                key = (instance_id, repetition, algorithm)
                if key in keys:
                    raise ValueError(f"{scenario.source_id}: duplicate run key {key}")
                keys.add(key)
                writer.writerow(
                    {
                        "scenario": scenario.name,
                        "instance_id": instance_id,
                        "algorithm": algorithm,
                        "repetition": repetition,
                        "value": format(value, ".12g"),
                        "objective": scenario.objective,
                        "runstatus": runstatus,
                    }
                )
                statuses[runstatus] += 1
                algorithms.add(algorithm)
                instances.add(instance_id)
                rows += 1
            if rows == 0:
                raise ValueError(f"{scenario.source_id} contains no data rows")
            by_scenario[scenario.name] = {
                "rows": rows,
                "instances": len(instances),
                "algorithms": len(algorithms),
                "measure": scenario.measure,
                "objective": scenario.objective,
                "runstatus": dict(sorted(statuses.items())),
            }
            total_rows += rows
    return {"rows": total_rows, "by_scenario": by_scenario}


def build_fuzzbench(
    raw_root: Path,
    output: Path,
    source: Source,
) -> dict[str, object]:
    fieldnames = [
        "experiment",
        "benchmark",
        "fuzzer",
        "trial_id",
        "time",
        "edges_covered",
    ]
    rows = 0
    benchmarks: set[str] = set()
    fuzzers: set[str] = set()
    trials: set[str] = set()
    keys: set[tuple[str, str, str, int]] = set()
    with gzip.open(source_path(raw_root, source), "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, set(fieldnames), source.source_id)
        with atomic_csv(output, fieldnames) as writer:
            for row_id, row in enumerate(reader):
                time = int(row["time"])
                edges = int(row["edges_covered"])
                if time < 0 or edges < 0:
                    raise ValueError(f"{source.source_id} row {row_id}: negative value")
                key = (row["benchmark"], row["fuzzer"], row["trial_id"], time)
                if key in keys:
                    raise ValueError(
                        f"{source.source_id}: duplicate trajectory key {key}"
                    )
                keys.add(key)
                writer.writerow(
                    {
                        "experiment": row["experiment"],
                        "benchmark": row["benchmark"],
                        "fuzzer": row["fuzzer"],
                        "trial_id": row["trial_id"],
                        "time": time,
                        "edges_covered": edges,
                    }
                )
                benchmarks.add(row["benchmark"])
                fuzzers.add(row["fuzzer"])
                trials.add(row["trial_id"])
                rows += 1
    if rows == 0:
        raise ValueError(f"{source.source_id} contains no rows")
    return {
        "rows": rows,
        "benchmarks": len(benchmarks),
        "fuzzers": len(fuzzers),
        "trials": len(trials),
    }


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def build_nab(
    raw_root: Path,
    output: Path,
    sources: dict[str, Source],
) -> dict[str, object]:
    labels_path = source_path(raw_root, sources["nab_labels"])
    with labels_path.open(encoding="utf-8") as handle:
        raw_labels = json.load(handle)

    fieldnames = [
        "series",
        "source_key",
        "row_id",
        "timestamp",
        "value",
        "annotated_window",
    ]
    total_rows = 0
    by_series: dict[str, dict[str, object]] = {}
    with atomic_csv(output, fieldnames) as writer:
        for series in NAB_SERIES:
            if series.label_key not in raw_labels:
                raise ValueError(f"NAB labels missing {series.label_key}")
            windows = [
                (parse_timestamp(start), parse_timestamp(end))
                for start, end in raw_labels[series.label_key]
            ]
            source = sources[series.source_id]
            timestamps: Counter[str] = Counter()
            annotated_rows = 0
            rows = 0
            with source_path(raw_root, source).open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                require_columns(
                    reader.fieldnames,
                    {"timestamp", "value"},
                    series.source_id,
                )
                for row_id, row in enumerate(reader):
                    timestamp = parse_timestamp(row["timestamp"])
                    value = float(row["value"])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"{series.source_id} row {row_id}: invalid value"
                        )
                    annotated = any(start <= timestamp <= end for start, end in windows)
                    writer.writerow(
                        {
                            "series": series.name,
                            "source_key": series.label_key,
                            "row_id": row_id,
                            "timestamp": row["timestamp"],
                            "value": format(value, ".15g"),
                            "annotated_window": int(annotated),
                        }
                    )
                    timestamps[row["timestamp"]] += 1
                    annotated_rows += int(annotated)
                    rows += 1
            if rows == 0:
                raise ValueError(f"{series.source_id} contains no rows")
            duplicate_timestamps = sum(count - 1 for count in timestamps.values())
            by_series[series.name] = {
                "rows": rows,
                "windows": len(windows),
                "annotated_rows": annotated_rows,
                "duplicate_timestamps_preserved": duplicate_timestamps,
            }
            total_rows += rows
    return {"rows": total_rows, "by_series": by_series}


def output_record(
    path: Path,
    schema: list[str],
    transform: str,
    statistics: dict[str, object],
) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "schema": schema,
        "transform": transform,
        "statistics": statistics,
    }


def write_provenance(
    path: Path,
    sources: list[Source],
    outputs: list[dict[str, object]],
) -> None:
    source_records = []
    for source in sources:
        record = asdict(source)
        record["bytes"] = record.pop("size")
        record["path"] = str(source.path)
        source_records.append(record)
    document = {
        "format_version": 1,
        "builder": "scripts/build_feedback_traces.py",
        "sources": source_records,
        "outputs": outputs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "feedback_sources.toml",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/feedback/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/feedback/derived",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_list = load_sources(args.manifest)
        verify_sources(args.input, source_list)
        sources = {source.source_id: source for source in source_list}

        open_bandit_path = args.output / "open_bandit.csv"
        open_bandit_stats = build_open_bandit(args.input, open_bandit_path, sources)
        aslib_path = args.output / "aslib.csv"
        aslib_stats = build_aslib(args.input, aslib_path, sources)
        fuzzbench_path = args.output / "fuzzbench.csv"
        fuzzbench_stats = build_fuzzbench(
            args.input,
            fuzzbench_path,
            sources["fuzzbench_sample"],
        )
        nab_path = args.output / "nab.csv"
        nab_stats = build_nab(args.input, nab_path, sources)

        outputs = [
            output_record(
                open_bandit_path,
                [
                    "behavior",
                    "row_id",
                    "timestamp",
                    "item_id",
                    "position",
                    "reward",
                    "logging_propensity",
                    "target_propensity",
                ],
                "Preserve logged rewards and propensities; add a context-free "
                "target policy uniform over item_id 0 through 39.",
                open_bandit_stats,
            ),
            output_record(
                aslib_path,
                [
                    "scenario",
                    "instance_id",
                    "algorithm",
                    "repetition",
                    "value",
                    "objective",
                    "runstatus",
                ],
                "Project two ASlib algorithm-run scenarios into one native "
                "algorithm-selection table without filtering rows.",
                aslib_stats,
            ),
            output_record(
                fuzzbench_path,
                [
                    "experiment",
                    "benchmark",
                    "fuzzer",
                    "trial_id",
                    "time",
                    "edges_covered",
                ],
                "Remove the source dataframe index and preserve every coverage "
                "trajectory measurement.",
                fuzzbench_stats,
            ),
            output_record(
                nab_path,
                [
                    "series",
                    "source_key",
                    "row_id",
                    "timestamp",
                    "value",
                    "annotated_window",
                ],
                "Join five NAB streams to the official inclusive anomaly "
                "windows; preserve duplicate timestamps as distinct rows.",
                nab_stats,
            ),
        ]
        write_provenance(args.output / "provenance.json", source_list, outputs)
    except (OSError, ValueError, KeyError, csv.Error, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for record in outputs:
        statistics = record["statistics"]
        print(
            f"built {record['path']:18} rows={statistics['rows']:>6} "
            f"sha256={record['sha256']}"
        )
    print(f"provenance written to {args.output / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
