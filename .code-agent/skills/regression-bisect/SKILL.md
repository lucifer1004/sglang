---
name: regression-bisect
description: Bisect regressions to their introducing commit and make a minimal fix that preserves the original intent and performance. Use when Codex is asked to reproduce and debug a regression, identify the first bad commit, run git bisect, or fix a regression without undoing the introducing change's intended behavior.
---

# Regression Bisect

Use this workflow for regressions that require history bisection and a minimal fix.

## Workflow

1. Require a reproducible bug signal.
   - If the user did not provide reproduction steps, ask for them before editing code.
   - Convert the repro into a deterministic command, test, script, or short manual checklist.
   - Record the exact pass/fail condition.

2. Establish known bad and known good commits.
   - Use commits supplied by the user when available.
   - Otherwise confirm the current commit is bad, then find or ask for an older commit expected to be good.
   - Prefer a separate `git worktree` for checkout and bisect work so local uncommitted changes are not disturbed.

3. Bisect to the introducing commit.
   - Use `git bisect` manually or with an automated repro command.
   - Treat flaky results as inconclusive; rerun before marking a commit good or bad.
   - Save the first bad commit, its diff, and the relevant intent from commit message, PR notes, nearby tests, and changed code.

4. Make the smallest intuitive fix.
   - Preserve the introducing commit's intended behavior.
   - Avoid unrelated refactors, formatting churn, or broad abstractions.
   - Add a comment only when the minimal correct fix is not obvious from the code.

5. Validate both the bug fix and the original intent.
   - Rerun the repro after every code change.
   - Run targeted tests covering the bug and tests related to the introducing commit's purpose.
   - For performance-sensitive paths, compare an appropriate benchmark, smoke perf test, or existing metric before and after; do not leave an obvious regression unreported.

6. Review and tighten the diff.
   - Re-read the final diff for side effects, unnecessary edits, naming clarity, and test coverage.
   - Keep only changes needed for the fix.
   - If another edit is made during review, rerun the repro and relevant tests again.

## Report

In the final response, include the known good commit, known bad commit, first bad commit, fix summary, tests run, performance check result when relevant, and any remaining risk.
