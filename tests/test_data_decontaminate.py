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
"""data.decontaminate: layer verdicts, embedding cache, fail-closed inputs, and plans."""

from __future__ import annotations

import json
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.integrations.embeddings import encoders
from nvflow.stages.data.decontaminate import DecontaminateStage
from nvflow.stages.data.decontaminate_worker import DecontaminationSettings, main

REFERENCES = [
    {
        "id": "r1",
        "question": "How did museum ticket sales change between season 2023 and season 2024 at the agency?",
    },
    {
        "id": "r2",
        "question": "Summarize the river flood risk factors the agency disclosed for 2022.",
    },
]
POOL = [
    {"uuid": "p1", "q": "Summarize the river flood risk factors the agency disclosed for 2022."},
    {
        "uuid": "p2",
        "q": "Using the visitor report: How did museum ticket sales change between season 2023 "
        "and season 2024 at the agency? Answer briefly.",
    },
    {
        "uuid": "p3",
        "q": "How did museum ticket sales change between season 2021 and season 2022 at the agency?",
    },
    {"uuid": "p4", "q": "List three exhibits opened last year."},
]


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _settings(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    reference = _write(tmp_path / "bench.jsonl", REFERENCES)
    return {
        "text_fields": ["q"],
        "id_field": "uuid",
        "references": {
            "bench": {"path": str(reference), "text_fields": ["question"], "id_field": "id"}
        },
        "ngram": {"size": 4, "drop": 0.6, "review": 0.3},
        **overrides,
    }


def _run(source: Path, output: Path, settings: dict[str, Any]) -> int:
    return main(
        ["run", str(source), "--output_dir", str(output), "--settings", json.dumps(settings)]
    )


class _BagOfWords:
    """Deterministic stand-in encoder that records every text it embeds."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, settings: Any) -> encoders.Encoder:
        def encode(texts: list[str]) -> np.ndarray:
            self.texts.extend(texts)
            vectors = np.zeros((len(texts), 512), dtype=np.float32)
            for row, text in enumerate(texts):
                for word in text.lower().split():
                    vectors[row, zlib.crc32(word.strip("?.:,").encode()) % 512] += 1
            return vectors

        return encode


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"text_fields": ["q"]}, "references"),
        ({"text_fields": ["q"], "references": {"bench": {}}}, "path"),
        (
            {"text_fields": ["q"], "references": {"bench": "/b.jsonl"}, "ngram": {"window": 3}},
            "unknown",
        ),
        (
            {"text_fields": ["q"], "references": {"bench": "/b.jsonl"}, "ngram": {"size": 8}},
            "threshold",
        ),
        (
            {
                "text_fields": ["q"],
                "references": {"bench": "/b.jsonl"},
                "exact": False,
                "ngram": None,
            },
            "at least one",
        ),
        (
            {
                "text_fields": ["q"],
                "references": {"bench": "/b.jsonl"},
                "embedding": {"backend": "x"},
            },
            "backend",
        ),
        (
            {"text_fields": ["q"], "references": {"bench": "/b.jsonl"}, "ngram": {"top_k": 0}},
            "top_k",
        ),
        (
            {
                "text_fields": ["q"],
                "references": {"bench": "/b.jsonl"},
                "ngram": {"review": 0.6, "drop": 0.5},
            },
            "below",
        ),
        (
            {"text_fields": ["q"], "references": {"bench": "/b.jsonl"}, "ngram": {"top_k": True}},
            "top_k",
        ),
    ],
)
def test_settings_fail_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DecontaminationSettings.from_config(config)


def test_a_misspelled_stage_option_fails_closed() -> None:
    config = {"input_file": "/in.jsonl", "output_dir": "/out", "text_fields": ["q"]}

    with pytest.raises(ValueError, match="unknown data.decontaminate option.*ngrams"):
        DecontaminateStage().validate_config(
            {**config, "references": {"bench": "/b.jsonl"}, "ngrams": {"drop": 0.9}}
        )


def test_lexical_layers_drop_exact_and_contained_references(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", POOL)
    output = tmp_path / "decontaminated"

    assert _run(source, output, _settings(tmp_path)) == 0

    assert [row["uuid"] for row in _lines(output / "kept.jsonl")] == ["p3", "p4"]
    dropped = {row["uuid"]: row["_contamination"] for row in _lines(output / "dropped.jsonl")}
    assert {(hit["layer"], hit["reference_id"]) for hit in dropped["p1"]} == {
        ("exact", "r2"),
        ("ngram", "r2"),
    }
    assert dropped["p2"] == [
        {
            "layer": "ngram",
            "score": 1.0,
            "reference": "bench",
            "reference_id": "r1",
            "verdict": "drop",
        }
    ]
    [candidate] = _lines(output / "candidates.jsonl")
    assert candidate["id"] == "p3"
    assert candidate["reference_id"] == "r1"
    assert candidate["layer"] == "ngram"
    assert candidate["score"] == pytest.approx(5 / 12, abs=1e-4)
    assert candidate["reference_text"] == REFERENCES[0]["question"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert (report["kept_rows"], report["dropped_rows"], report["review_candidates"]) == (2, 2, 1)
    assert report["reference_rows"] == {"bench": 2}
    assert (output / "decontaminate.done").exists()


def test_reference_records_sharing_an_id_keep_their_own_text(tmp_path: Path) -> None:
    reference = _write(
        tmp_path / "bench.jsonl",
        [
            {"id": "same", "question": "alpha beta gamma delta epsilon zeta eta theta"},
            {"id": "same", "question": "one two three four five six seven eight"},
        ],
    )
    source = _write(
        tmp_path / "pool.jsonl", [{"uuid": "p1", "q": "alpha beta gamma delta and more"}]
    )
    settings = {
        "text_fields": ["q"],
        "id_field": "uuid",
        "references": {
            "bench": {"path": str(reference), "text_fields": ["question"], "id_field": "id"}
        },
        "ngram": {"size": 4, "review": 0.2},
    }
    output = tmp_path / "decontaminated"

    assert _run(source, output, settings) == 0

    [candidate] = _lines(output / "candidates.jsonl")
    assert candidate["reference_text"] == "alpha beta gamma delta epsilon zeta eta theta"


def test_a_reference_record_without_text_fails_the_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    partial = _write(tmp_path / "partial.jsonl", [*REFERENCES, {"id": "r3", "prompt": "no text"}])
    settings["references"]["bench"]["path"] = str(partial)

    with pytest.raises(ValueError, match="1 of 3 records have no text"):
        _run(_write(tmp_path / "pool.jsonl", POOL), tmp_path / "decontaminated", settings)


def test_ngram_keeps_each_rows_best_top_k_matches(tmp_path: Path) -> None:
    references = [
        {
            "id": f"r{index}",
            "question": "w1 w2 w3 w4 " + " ".join(f"t{index}x{n}" for n in range(index + 1)),
        }
        for index in range(6)
    ]
    settings = {
        "text_fields": ["q"],
        "id_field": "uuid",
        "references": {
            "bench": {
                "path": str(_write(tmp_path / "bench.jsonl", references)),
                "text_fields": ["question"],
                "id_field": "id",
            }
        },
        "ngram": {"size": 4, "review": 0.1, "top_k": 2},
    }
    source = _write(tmp_path / "pool.jsonl", [{"uuid": "p1", "q": "w1 w2 w3 w4 and more"}])
    output = tmp_path / "decontaminated"

    assert _run(source, output, settings) == 0

    assert [row["reference_id"] for row in _lines(output / "candidates.jsonl")] == ["r0", "r1"]


def test_a_reference_shorter_than_the_ngram_size_matches_a_record_containing_it(
    tmp_path: Path,
) -> None:
    reference = _write(
        tmp_path / "short.jsonl", [{"id": "s1", "question": "Lisbon's summer 2023 rainfall?"}]
    )
    source = _write(
        tmp_path / "pool.jsonl",
        [
            {
                "uuid": "p1",
                "q": "From the weather report, what was Lisbon's summer 2023 rainfall in total?",
            },
            {"uuid": "p2", "q": "What was Lisbon's summer 2022 rainfall in total?"},
        ],
    )
    settings = {
        "text_fields": ["q"],
        "id_field": "uuid",
        "references": {"short": {"path": str(reference), "text_fields": ["question"]}},
        "ngram": {"size": 8, "drop": 0.5},
    }
    output = tmp_path / "decontaminated"

    assert _run(source, output, settings) == 0

    [dropped] = _lines(output / "dropped.jsonl")
    assert (dropped["uuid"], dropped["_contamination"][0]["score"]) == ("p1", 1.0)
    assert [row["uuid"] for row in _lines(output / "kept.jsonl")] == ["p2"]


def test_chunked_runs_match_one_chunk(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", POOL)
    settings = _settings(tmp_path)
    outputs = []
    for num_chunks in (1, 3):
        output = tmp_path / f"chunks-{num_chunks}"
        assert (
            main(
                [
                    "run",
                    str(source),
                    "--output_dir",
                    str(output),
                    "--settings",
                    json.dumps(settings),
                    "--num_chunks",
                    str(num_chunks),
                ]
            )
            == 0
        )
        outputs.append(
            [
                (output / name).read_text(encoding="utf-8")
                for name in ("kept.jsonl", "dropped.jsonl")
            ]
        )

    assert outputs[0] == outputs[1]


def test_embedding_layer_reviews_paraphrases_and_reuses_cached_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _BagOfWords()
    monkeypatch.setitem(encoders._FACTORIES, "sentence_transformers", encoder)
    source = _write(tmp_path / "pool.jsonl", POOL)
    settings = _settings(
        tmp_path,
        ngram=None,
        exact=True,
        embedding={
            "backend": "sentence_transformers",
            "model": "bag-of-words",
            "prefix": "query: ",
            "review": 0.8,
            "drop": 0.99,
            "top_k": 1,
        },
        cache_dir=str(tmp_path / "cache"),
    )

    assert _run(source, tmp_path / "first", settings) == 0

    candidates = _lines(tmp_path / "first" / "candidates.jsonl")
    assert {(row["id"], row["layer"], row["reference_id"]) for row in candidates} >= {
        ("p3", "embedding", "r1")
    }
    assert "p4" not in {row["id"] for row in candidates}
    texts = [reference["question"] for reference in REFERENCES] + [row["q"] for row in POOL]
    assert sorted(encoder.texts) == sorted(f"query: {text}" for text in texts)
    embedded = len(encoder.texts)

    assert _run(source, tmp_path / "second", settings) == 0
    assert len(encoder.texts) == embedded
    assert (tmp_path / "second" / "candidates.jsonl").read_bytes() == (
        tmp_path / "first" / "candidates.jsonl"
    ).read_bytes()


def test_worker_with_vllm_loaded_exits_with_its_status_after_writing_outputs(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "pool.jsonl", POOL)
    output = tmp_path / "decontaminated"
    script = (
        "import runpy, sys, types; sys.modules['vllm'] = types.ModuleType('vllm'); "
        "runpy.run_module('nvflow.stages.data.decontaminate_worker', run_name='__main__')"
    )
    arguments = ["run", str(source), "--output_dir", str(output)]

    finished = subprocess.run(
        [sys.executable, "-c", script, *arguments, "--settings", json.dumps(_settings(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )

    assert finished.returncode == 0, finished.stderr
    assert (output / "decontaminate.done").exists()
    failed = subprocess.run(
        [sys.executable, "-c", script, *arguments, "--settings", "{}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode != 0


def test_changed_reference_file_fails_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", POOL)
    output = tmp_path / "decontaminated"
    settings = _settings(tmp_path)
    assert _run(source, output, settings) == 0

    _write(Path(settings["references"]["bench"]["path"]), REFERENCES[:1])
    with pytest.raises(ValueError, match="settings changed: reference_digests"):
        _run(source, output, settings)


def _context(config: dict[str, Any], *, present: set[str] | None = None) -> PlanningContext:
    class Filesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return {path: path in (present or set()) for path in paths}

    return PlanningContext(
        stage="decontaminate",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="decon",
        run_after=(),
        live=present is not None,
        filesystem=cast(Any, None if present is None else Filesystem()),
        engine=None if present is None else "slurm",
    )


CONFIG: dict[str, Any] = {
    "input_file": "/data/pool.jsonl",
    "output_dir": "/out/decon",
    "text_fields": ["q"],
    "references": {"bench": "/eval/bench.jsonl"},
    "embedding": {"backend": "vllm", "model": "/models/embed", "review": 0.85},
}


def _shape(config: dict[str, Any]) -> list[tuple[str, str, int | None]]:
    plan = DecontaminateStage().plan(_context(config))
    return [
        (task.name, task.commands[0].container, task.resources[0].gpus_per_node)
        for task in plan.tasks
    ]


def test_vllm_embeddings_plan_a_cpu_task_then_a_gpu_task() -> None:
    assert _shape(CONFIG) == [
        ("decon-lexical", "nemo-skills", 0),
        ("decon-embed-merge", "vllm", 1),
    ]


def test_installation_command_runs_only_in_the_stages_container() -> None:
    plan = DecontaminateStage().plan(_context({**CONFIG, "installation_command": "pip install x"}))

    assert ["pip install x" in task.commands[0].command for task in plan.tasks] == [True, False]


def test_cpu_embeddings_plan_one_task() -> None:
    embedding = {"backend": "sentence_transformers", "model": "/models/minilm", "review": 0.85}
    assert _shape({**CONFIG, "embedding": embedding}) == [
        ("decon-lexical-embed-merge", "nemo-skills", 0)
    ]


def test_chunked_plan_merges_after_every_chunk() -> None:
    embedding = {**CONFIG["embedding"], "container": "embedder", "num_gpus": 2}
    plan = DecontaminateStage().plan(_context({**CONFIG, "embedding": embedding, "num_chunks": 2}))

    assert [(task.name, task.commands[0].container) for task in plan.tasks] == [
        ("decon-chunk0-lexical", "nemo-skills"),
        ("decon-chunk0-embed", "embedder"),
        ("decon-chunk1-lexical", "nemo-skills"),
        ("decon-chunk1-embed", "embedder"),
        ("decon-merge", "nemo-skills"),
    ]
    assert plan.tasks[1].resources[0].gpus_per_node == 2
    assert [d.upstream for d in plan.tasks[-1].dependencies] == [
        plan.tasks[1].task_id,
        plan.tasks[3].task_id,
    ]


def test_live_plan_requires_reference_files() -> None:
    with pytest.raises(ValueError, match="/eval/bench.jsonl"):
        DecontaminateStage().plan(_context(CONFIG, present={"/data/pool.jsonl"}))
