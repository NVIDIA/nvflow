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
"""CLI entry point for building the local SEC full-text search index.

Produces both artifacts NeMo-Gym's ``edgar_search`` reads: the index and, beside
it, the ``.metadata`` companion holding the same rows without the filing text.

Invoked by the NVFlow stage as::

    python -m nvfinance.utils.index.build_index build \\
        --artifact_root /workspace/outputs/.../step-0-download \\
        --index_path /workspace/outputs/.../step-1-index/sp500_sec_fts.sqlite

Inspect what an existing index recorded about its own build::

    python -m nvfinance.utils.index.build_index preflight \\
        --index_path /workspace/outputs/.../sp500_sec_fts.sqlite
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import time

from nvflow.utils import setup_logger

from .index import SearchIndex, companion_path_for, write_metadata_companion

logger = setup_logger(__name__)

# Built here, then renamed into place, so readers never open a partial index.
BUILDING_SUFFIX = ".building"

_PROGRESS_EVERY_DOCUMENTS = 5_000


def _format_elapsed(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}m {seconds % 60:.1f}s"
    return f"{seconds:.1f}s"


def _publish(source: Path, destination: Path) -> None:
    """Move a finished index into place, leaving no stale sidecar journals.

    A ``-wal`` left over from the previous index would otherwise sit beside the
    new file and be recovered into it on first open.
    """
    source.replace(destination)
    for suffix in ("-wal", "-shm"):
        for path in (Path(f"{source}{suffix}"), Path(f"{destination}{suffix}")):
            path.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifact_root)
    index_path = Path(args.index_path)
    building_path = Path(f"{index_path}{BUILDING_SUFFIX}")
    index_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Corpus: %s", artifact_root)
    logger.info("Index: %s", index_path)

    if args.incremental and building_path.is_file():
        # Documents already in here are complete, so an interrupted build is
        # resumed rather than repeated. A full S&P 500 build is ~26h of HTML
        # parsing and will outlive most job time limits.
        logger.info("Resuming the unfinished build at %s", building_path)
        builder = SearchIndex.update
    elif args.incremental and index_path.is_file():
        logger.info("Seeding from the published index for an incremental build")
        shutil.copy2(index_path, building_path)
        builder = SearchIndex.update
    else:
        if args.incremental:
            logger.warning(
                "Incremental build found no index at %s and no unfinished build at %s; "
                "building from scratch",
                index_path,
                building_path,
            )
        builder = SearchIndex.build

    def progress(counts: dict[str, int]) -> None:
        if counts["new_documents"] % _PROGRESS_EVERY_DOCUMENTS == 0:
            logger.info(
                "Indexed %d new documents (%d already present)",
                counts["new_documents"],
                counts["already_indexed"],
            )

    started = time()
    index, report = builder(
        artifact_root,
        building_path,
        batch_size=args.batch_size,
        progress=progress,
    )
    logger.info("Build took %s", _format_elapsed(time() - started))
    logger.info("Preflight: %s", json.dumps(report.to_dict(), sort_keys=True, default=str))

    if report.missing_paths and not args.allow_missing_paths:
        for example in (report.excluded_examples or [])[:5]:
            logger.error("Missing: %s -> %s", example["url"], example["detail"])
        logger.error(
            "%d of %d candidate rows have no file on disk. The corpus and "
            "sec_metadata.parquet disagree, so this index would silently omit "
            "those filings. Re-run the download stage until every ticker reports "
            "zero filings to download, or pass --allow_missing_paths to accept "
            "the gap. Unpublished index left at %s, which --incremental resumes "
            "instead of reparsing",
            report.missing_paths,
            report.candidate_rows,
            building_path,
        )
        return 1

    # The companion lands first: a published index is then never without one,
    # and only the harmless opposite can be seen mid-publish.
    companion_path = companion_path_for(index_path)
    started = time()
    documents = write_metadata_companion(building_path, companion_path)
    logger.info(
        "Wrote %s with %d documents in %s",
        companion_path,
        documents,
        _format_elapsed(time() - started),
    )

    _publish(building_path, index_path)
    index = SearchIndex(index_path)
    logger.info("Published %s with %d documents", index_path, index.document_count())
    for form_type, documents in sorted(index.counts_by_form_type().items()):
        logger.info("  %-10s %7d", form_type, documents)
    return 0


def preflight(args: argparse.Namespace) -> int:
    index = SearchIndex(args.index_path)
    report = index.preflight_report()
    report["document_count"] = index.document_count()
    report["counts_by_form_type"] = index.counts_by_form_type()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local SEC full-text search index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build or update an index")
    build_parser.add_argument(
        "--artifact_root",
        type=str,
        required=True,
        help="Download stage output directory holding data/ and sec_metadata.parquet",
    )
    build_parser.add_argument("--index_path", type=str, required=True)
    build_parser.add_argument("--batch_size", type=int, default=500)
    build_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Add only documents missing from the existing index",
    )
    build_parser.add_argument(
        "--allow_missing_paths",
        action="store_true",
        help="Publish even if rows in the metadata parquet have no file on disk",
    )
    build_parser.set_defaults(handler=build)

    preflight_parser = subparsers.add_parser(
        "preflight", help="Print the build report stored in an index"
    )
    preflight_parser.add_argument("--index_path", type=str, required=True)
    preflight_parser.set_defaults(handler=preflight)

    args = parser.parse_args()
    try:
        sys.exit(args.handler(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
