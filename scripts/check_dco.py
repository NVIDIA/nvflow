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
"""Check that commit message(s) contain a DCO Signed-off-by line."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SIGNOFF_MARKER = "Signed-off-by:"


def check_message(content: str) -> bool:
    """Return True if the commit message contains a valid sign-off line."""
    return SIGNOFF_MARKER in content


def main() -> int:
    # commit-msg hook: single file path as argv[1]
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
        if not path.is_file():
            print(f"check_dco: not a file: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="replace")
        if not check_message(text):
            print(
                "DCO check failed: commit message must contain a 'Signed-off-by:' line.",
                'Use: git commit -s -m "Your message"',
                sep="\n",
                file=sys.stderr,
            )
            return 1
        return 0

    # CI: check all non-merge commits in current branch vs target (e.g. main)
    base = os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME") or os.environ.get(
        "BASE_BRANCH", "main"
    )

    # Try to resolve the base ref; on detached HEAD (post-merge pipelines on
    # main) the local branch ref may not exist, so fetch origin/<base> as
    # fallback reference.
    # --first-parent: only check direct commits on the branch, not commits
    # that came in via merged feature branches.  This avoids re-checking
    # historical unsigned commits from old MRs during cross-branch merges.
    result = subprocess.run(
        ["git", "rev-list", "--no-merges", "--first-parent", f"{base}..HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        # base ref not found locally – try origin/<base>
        result = subprocess.run(
            ["git", "rev-list", "--no-merges", "--first-parent", f"origin/{base}..HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    if result.returncode != 0:
        # Fallback: base still not found (e.g. shallow clone) – check only
        # HEAD, but skip merge commits.
        rev_list = subprocess.run(
            ["git", "rev-list", "--no-merges", "-n", "1", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        revs = rev_list.stdout.strip().splitlines() if rev_list.returncode == 0 else []
    else:
        revs = result.stdout.strip().splitlines()

    failed = []
    for rev in revs:
        if not rev:
            continue
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B", rev],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if msg.returncode != 0 or not check_message(msg.stdout):
            failed.append(rev[:8])
    if failed:
        print(
            "DCO check failed: the following commits are missing a 'Signed-off-by:' line:",
            *failed,
            "Amend with: git commit --amend -s --no-edit",
            sep="\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
