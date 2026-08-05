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
"""Per-stage JSONL field allowlists for DG-SDG.

Each generic DG-SDG stage projects its output JSONL to ``STAGE_KEEP[stage] |
domain_keep_fields`` (set union) at the stage boundary, before the next stage
reads it. Goal: drop stale fields that would silently contaminate downstream
stages -- most importantly the NeMo-Gym rollout metadata and the ``generation``
/ ``reasoning_content`` keys that get overwritten by every Gym call.

Domain-specific fields (e.g. ``company_name``, ``file_path0``) are supplied
per recipe via the workflow YAML key ``domain_keep_fields`` and unioned with
``STAGE_KEEP[stage]`` at trim time. Generic stage code never hardcodes them.
"""

# Cross-stage scratch / noise that we *never* want to survive a stage boundary.
# These are always dropped on top of (i.e. removed from) the per-stage KEEP
# allowlist so that even if a future contributor adds one to STAGE_KEEP by
# mistake, the trim still filters it out.
ALWAYS_DROP: frozenset[str] = frozenset(
    {
        # NeMo-Gym rollout passthrough metadata (added by responses_api on every
        # Gym call; never read downstream).
        "_ng_task_index",
        "_ng_rollout_index",
        "agent_ref",
        "reward",
        "match_details",
        "verifier",
        # Generation-time bookkeeping added by responses_api / Gym workers.
        "serialized_output",
        "num_generated_tokens",
        "finish_reason",
        "generation_start_time",
        "generation_end_time",
        "generation_time",
        "responses_create_params",
    }
)


# Per-stage allowlist of *generic* fields (i.e. fields that the lib code
# produces or that downstream lib code needs). Domain-specific fields come
# from the workflow YAML's ``domain_keep_fields`` and are unioned at trim time.
#
# Stage 0 (``dg_sdg_preprocess``) is intentionally absent: it manufactures the
# initial JSONL from raw documents, so there is no upstream record to project
# from. The Stage 1 trim acts as the safety net if the recipe writes junk.
STAGE_KEEP: dict[str, frozenset[str]] = {
    # Q-side output (``verified/output-rs*.jsonl``): keep the Yes/No
    # ``generation`` because the A-prep step votes on it; drop the Q-verify
    # CoT (``reasoning_content``) -- nobody downstream reads it.
    "generate_verified_questions": frozenset(
        {
            "context",
            "problem",
            "question_type",
            "generation",
        }
    ),
    # A-side output (``generated/output-rs*.jsonl``): keep the answer text
    # (``generation``) and the answer CoT (``reasoning_content``); both get
    # snapshotted into ``reference_*`` by genselect.postprocess in Stage 3.
    #
    # ``answer_response`` / ``answer_responses_create_params`` carry the *full*
    # Responses-API original form of each candidate answer (the exact request +
    # response object the A-gen model produced).  They are the literal
    # ``response`` / ``responses_create_params`` snapshotted under a non-
    # ALWAYS_DROP alias by ``enrich_rollouts`` so the trim keeps them.
    # genselect collapses the per-seed ``answer_response`` into
    # ``answer_responses_list`` and selects one into ``reference_response`` for
    # the final post-process output (Responses-API ``final_result.jsonl``).
    "generate_answers": frozenset(
        {
            "context",
            "problem",
            "question_type",
            "question_voting_pass_rate",
            "question_voting_total",
            "generation",
            "reasoning_content",
            "answer_response",
            "answer_responses_create_params",
        }
    ),
    # GenSelect-picked output (``selected_answers.jsonl``): ``reference_*``
    # carry the selected answer through evaluate/aggregate/difficulty;
    # ``generation`` carries the same selected answer as the prompt input for
    # evaluate. Genselect scaffolding (solutions/generations_list/answer_N/...)
    # is dropped because it has served its purpose.
    "gym_genselect_answers": frozenset(
        {
            "context",
            "problem",
            "question_type",
            "question_voting_pass_rate",
            "question_voting_total",
            "reference_answer",
            "reference_reasoning",
            "reference_response",
            "reference_responses_create_params",
            "generation",
            "genselect_answers_metadata",
        }
    ),
    # Multi-seed eval rollouts: keep ``evaluate_generation`` for aggregate to
    # parse; drop ``reasoning_content`` which by now is the evaluate-judge CoT
    # (not the answer reasoning) and would otherwise silently overwrite the
    # real answer CoT carried in ``reference_reasoning``.
    "evaluate_answers": frozenset(
        {
            "context",
            "problem",
            "question_type",
            "question_voting_pass_rate",
            "question_voting_total",
            "reference_answer",
            "reference_reasoning",
            "reference_response",
            "reference_responses_create_params",
            "generation",
            "evaluate_generation",
        }
    ),
    # Aggregated answers: per-seed ``evaluate_generation`` and ``correct`` are
    # dropped; the consensus ``answerable`` survives.
    "aggregate_answers": frozenset(
        {
            "context",
            "problem",
            "question_type",
            "question_voting_pass_rate",
            "question_voting_total",
            "reference_answer",
            "reference_reasoning",
            "reference_response",
            "reference_responses_create_params",
            "generation",
            "answerable",
        }
    ),
    # Final training data (``final_result.jsonl``): post-process has already
    # renamed ``reference_reasoning -> reasoning_content`` and
    # ``reference_answer -> answer``, so the allowlist uses the post-rename
    # names. ``genselect_answers_metadata`` is intentionally dropped from the
    # final output -- it was useful for debugging mid-pipeline but is noise
    # for SFT / RL.
    # ``response`` + ``responses_create_params`` are the Responses-API original form
    # post-process restores (renamed from ``reference_response`` /
    # ``reference_responses_create_params``); ``expected_answer`` mirrors
    # ``answer``.  Note ``responses_create_params`` is in ALWAYS_DROP, so the
    # post-process stage re-adds it via ``build_trim_cmd(extra_keep_fields=...)``
    # -- listing it here is documentation; the trim would otherwise strip it.
    "dgsdg_post_process": frozenset(
        {
            "context",
            "problem",
            "answer",
            "reasoning_content",
            "question_type",
            "answerable",
            "question_voting_pass_rate",
            "question_voting_total",
            "expected_answer",
            "response",
            "responses_create_params",
        }
    ),
}
