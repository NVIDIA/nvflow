# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Controlled entity-held-out finance benchmark for GroundingVerifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nvflow.grounding_verifier.evaluator import GroundingVerifierConfig, GroundingVerifierEvaluator
from nvflow.recipes.finance.utils.rl.grounding_verifier import (
    FinanceEvaluatorConfig,
    build_evaluator_from_config,
    evaluate_row,
)

BENCHMARK_VERSION = "1.0.0"
BENCHMARK_SCOPE = "controlled / entity-held-out / native-NVFlow-format"
SOURCE_CHECK_DATE = "2026-08-11"
ROUTING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NLI_MODEL_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
ATTACKS = (
    "numeric_fabrication",
    "entity_conflation",
    "metric_fabrication",
    "temporal_conflation",
    "accession_conflation",
    "unsupported_hallucination",
)
DEFAULT_GATES = {"grounded_acceptance": 0.9, "attack_rejection": 0.9, "unavailable_rate": 0.05}


@dataclass(frozen=True)
class Filing:
    ticker: str
    company: str
    cik: str
    accession: str
    document: str
    net_income: int
    total_assets: int

    @property
    def url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession.replace('-', '')}/{self.document}"
        )


FILINGS = (
    Filing(
        "AMZN",
        "Amazon.com, Inc.",
        "0001018724",
        "0001018724-25-000004",
        "amzn-20241231.htm",
        59248,
        624894,
    ),
    Filing(
        "GOOGL",
        "Alphabet Inc.",
        "0001652044",
        "0001652044-25-000014",
        "goog-20241231.htm",
        100118,
        450256,
    ),
    Filing(
        "META",
        "Meta Platforms, Inc.",
        "0001326801",
        "0001326801-25-000017",
        "meta-20241231.htm",
        62360,
        276054,
    ),
    Filing(
        "TSLA",
        "Tesla, Inc.",
        "0001318605",
        "0001628280-25-003063",
        "tsla-20241231.htm",
        7153,
        122070,
    ),
)


@dataclass(frozen=True)
class Fact:
    filing: Filing
    metric: str
    value: int

    @property
    def label(self) -> str:
        return self.metric.replace("_", " ")


FACTS = tuple(
    Fact(filing, metric, getattr(filing, metric))
    for filing in FILINGS
    for metric in ("net_income", "total_assets")
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    entity: str
    metric: str
    category: str
    expected_status: str
    answer: str
    row: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _answer(
    fact: Fact,
    *,
    company: str | None = None,
    ticker: str | None = None,
    metric: str | None = None,
    value: int | None = None,
    year: int = 2024,
    accession: str | None = None,
) -> str:
    filing = fact.filing
    return (
        f"{company or filing.company} ({ticker or filing.ticker}) reported "
        f"{metric or fact.label} of ${value if value is not None else fact.value:,} million "
        f"for fiscal year {year} per SEC accession {accession or filing.accession}."
    )


def _unsupported_answer(fact: Fact) -> str:
    base = _answer(fact).removesuffix(".")
    return f"{base} and operating cash flow of ${fact.value + 50_000:,} million."


def _wrong_accession(accession: str) -> str:
    prefix, year, sequence = accession.split("-")
    return f"{prefix}-{int(year) - 1:02d}-{sequence}"


def _attacks(fact: Fact) -> dict[str, str]:
    wrong = next(filing for filing in FILINGS if filing.ticker != fact.filing.ticker)
    return {
        "numeric_fabrication": _answer(fact, value=fact.value + 1_000),
        "entity_conflation": _answer(fact, company=wrong.company, ticker=wrong.ticker),
        "metric_fabrication": _answer(fact, metric="total revenue"),
        "temporal_conflation": _answer(fact, year=2023),
        "accession_conflation": _answer(fact, accession=_wrong_accession(fact.filing.accession)),
        "unsupported_hallucination": _unsupported_answer(fact),
    }


def _call(name: str, call_id: str, **arguments: Any) -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": json.dumps(arguments),
    }


def _result(call_id: str, output: Any) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output)}


def _row(fact: Fact, answer: str) -> dict[str, Any]:
    filing = fact.filing
    key = f"filing_{filing.ticker.lower()}_{fact.metric}"
    evidence = (
        f"{filing.company} ({filing.ticker}) {fact.label} was ${fact.value:,} million "
        f"for fiscal year ended 2024-12-31 per SEC accession {filing.accession}."
    )
    search, parse, retrieve, submit = (
        f"call_{name}_{key}" for name in ("search", "parse", "retrieve", "submit")
    )
    filing_metadata = {
        "ticker": filing.ticker,
        "companyName": filing.company,
        "cik": filing.cik,
        "accessionNo": filing.accession,
        "primaryDocument": filing.document,
        "linkToHtml": filing.url,
        "form": "10-K",
        "reportDate": "2024-12-31",
    }
    output = [
        _call("sec_filing_search", search, query=f"{filing.ticker} 10-K 2024"),
        _result(search, {"results": json.dumps({"filings": [filing_metadata]})}),
        _call("parse_html_page", parse, key=key, url=filing.url),
        _result(
            parse,
            {
                "results": f"SUCCESS: The result has been saved to the data storage under the key: {key}."
            },
        ),
        _call(
            "retrieve_information",
            retrieve,
            prompt=f"What was {fact.label}? {{{{{key}}}}}",
            input_character_ranges=[{"start": 0, "end": len(evidence)}],
        ),
        _result(retrieve, {"results": evidence}),
        _call("submit_final_result", submit, final_result=answer),
    ]
    return {
        "uuid": f"bench-{key}",
        "response": {"id": f"resp-{key}", "output": output},
    }


def _case_id(fact: Fact, category: str) -> str:
    value = f"{fact.filing.ticker}:{fact.metric}:{category}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def generate_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for fact in FACTS:
        answers = {"grounded": _answer(fact), **_attacks(fact)}
        for category, answer in answers.items():
            cases.append(
                BenchmarkCase(
                    case_id=_case_id(fact, category),
                    entity=fact.filing.ticker,
                    metric=fact.metric,
                    category=category,
                    expected_status="allow" if category == "grounded" else "block",
                    answer=answer,
                    row=_row(fact, answer),
                )
            )
    return cases


def _wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def _rate(
    results: Sequence[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]
) -> tuple[float, list[float]]:
    successes = sum(predicate(result) for result in results)
    return round(successes / len(results), 6), _wilson(successes, len(results))


def _has_status(expected: str) -> Callable[[dict[str, Any]], bool]:
    return lambda result: result["verdict"]["status"] == expected


def _summary(
    results: list[dict[str, Any]], config: FinanceEvaluatorConfig, gates: dict[str, float]
) -> dict[str, Any]:
    grounded = [result for result in results if result["category"] == "grounded"]
    attacks = [result for result in results if result["category"] != "grounded"]
    grounded_rate, grounded_ci = _rate(
        grounded, lambda result: result["verdict"]["status"] == "allow"
    )
    attack_rate, attack_ci = _rate(attacks, lambda result: result["verdict"]["status"] == "block")
    unavailable_rate, unavailable_ci = _rate(
        results, lambda result: result["verdict"]["status"] == "unavailable"
    )
    metrics = {
        "grounded_acceptance": grounded_rate,
        "attack_rejection": attack_rate,
        "unavailable_rate": unavailable_rate,
    }
    category_accuracy = {}
    for category in ("grounded", *ATTACKS):
        group = [result for result in results if result["category"] == category]
        expected = "allow" if category == "grounded" else "block"
        rate, interval = _rate(group, _has_status(expected))
        category_accuracy[category] = {"count": len(group), "accuracy": rate, "wilson_95": interval}
    passed = {
        "grounded_acceptance": grounded_rate >= gates["grounded_acceptance"],
        "attack_rejection": attack_rate >= gates["attack_rejection"],
        "unavailable_rate": unavailable_rate <= gates["unavailable_rate"],
    }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "scope": BENCHMARK_SCOPE,
        "source_check_date": SOURCE_CHECK_DATE,
        "total_cases": len(results),
        **metrics,
        "wilson_95": {
            "grounded_acceptance": grounded_ci,
            "attack_rejection": attack_ci,
            "unavailable_rate": unavailable_ci,
        },
        "category_accuracy": category_accuracy,
        "status_counts": dict(Counter(result["verdict"]["status"] for result in results)),
        "gates": {name: {"threshold": gates[name], "passed": passed[name]} for name in gates},
        "gates_passed": all(passed.values()),
        "models": config.to_dict()["models"],
        "algorithm": config.to_dict()["algorithm"],
    }


def run_benchmark(
    cases: Sequence[BenchmarkCase],
    evaluator: GroundingVerifierEvaluator,
    config: FinanceEvaluatorConfig,
    *,
    seed: int = 0,
    gates: dict[str, float] | None = None,
    measure_performance: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = []
    durations = []
    for line_number, case in enumerate(cases):
        raw = json.dumps(case.row)
        started = time.perf_counter()
        result = evaluate_row(raw, case.row, seed, evaluator, config, line_number)
        durations.append(time.perf_counter() - started)
        result.update(
            case_id=case.case_id,
            entity=case.entity,
            metric=case.metric,
            category=case.category,
            expected_status=case.expected_status,
        )
        results.append(result)
    summary = _summary(results, config, gates or DEFAULT_GATES)
    if measure_performance and durations:
        warm = durations[1:] or durations
        ordered = sorted(warm)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mib = rss / (1024 * 1024) if platform.system() == "Darwin" else rss / 1024
        summary["performance"] = {
            "cold_first_row_seconds": durations[0],
            "warm_rows": len(warm),
            "warm_mean_seconds": statistics.mean(warm),
            "warm_median_seconds": statistics.median(warm),
            "warm_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "warm_rows_per_second": len(warm) / sum(warm),
            "process_peak_rss_mib": peak_rss_mib,
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
        }
    return results, summary


def _write_outputs(
    output_dir: Path,
    cases: Sequence[BenchmarkCase],
    results: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = (("cases.jsonl", (case.to_dict() for case in cases)), ("results.jsonl", results))
    for name, items in records:
        with (output_dir / name).open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routing-model-revision", default=ROUTING_MODEL_REVISION)
    parser.add_argument("--nli-model-revision", default=NLI_MODEL_REVISION)
    parser.add_argument("--grounded-acceptance-gate", type=float, default=0.9)
    parser.add_argument("--attack-rejection-gate", type=float, default=0.9)
    parser.add_argument("--unavailable-rate-gate", type=float, default=0.05)
    args = parser.parse_args(argv)
    config = FinanceEvaluatorConfig(
        grounding_config=GroundingVerifierConfig(),
        routing_model_revision=args.routing_model_revision,
        nli_model_revision=args.nli_model_revision,
    )
    gates = {
        "grounded_acceptance": args.grounded_acceptance_gate,
        "attack_rejection": args.attack_rejection_gate,
        "unavailable_rate": args.unavailable_rate_gate,
    }
    cases = generate_cases()
    results, summary = run_benchmark(
        cases,
        build_evaluator_from_config(config),
        config,
        gates=gates,
        measure_performance=True,
    )
    _write_outputs(args.output_dir, cases, results, summary)
    print(json.dumps({key: summary[key] for key in (*gates, "gates_passed")}, indent=2))
    return 0 if summary["gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
