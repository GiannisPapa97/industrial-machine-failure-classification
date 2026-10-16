# Project Instructions for Coding Agents

## 1. General Principles

- Help build a reliable, reproducible, maintainable Python machine-learning project for industrial machine failure classification.
- Read the existing code and project configuration before making changes.
- Prefer simple, clear, well-tested solutions over unnecessary complexity.
- Keep changes focused on the task. Do not make unrelated changes.
- Do not invent project requirements, data characteristics, test results, or command outputs.
- Explain significant decisions and assumptions.

## 2. Testing and Quality Checks

- Before reporting a task as complete, run all relevant tests and quality checks configured for the project.
- Before a change is considered ready for a commit, ensure that applicable checks have passed.
- This includes unit tests, integration tests, linters, formatters, type checkers, static analysis, and other configured validation tools, as applicable.
- Inspect `pyproject.toml`, project documentation, and configuration files to identify the actual checks and their correct commands.
- Add or update tests when changing behavior, fixing bugs, or introducing functionality.
- Do not claim that a check passed unless you actually ran it and verified its result.
- If a check fails, investigate and fix failures caused by your changes.
- Do not hide, skip, weaken, or remove tests or quality checks just to obtain a passing result.
- Do not claim that the project is fully validated if any relevant check has not been run.
- If a check cannot be run, report why. Clearly distinguish passing checks, failing checks, and checks not run.
- Do not introduce new development tools or dependencies without a good reason. Explain proposed additions when appropriate.

## 3. Git and GitHub Operations Require Explicit Permission

**Never perform a Git or GitHub operation that changes repository state unless the user explicitly authorizes that specific operation.**

This restriction includes, but is not limited to:

- Creating, switching, renaming, or deleting branches.
- Staging or unstaging files.
- Creating, amending, or rewriting commits.
- Pushing or force-pushing.
- Pulling or fetching.
- Merging or rebasing.
- Resetting, reverting, cleaning, or discarding changes.
- Stashing or applying stashes.
- Creating or modifying tags.
- Opening, modifying, or merging pull requests.
- Creating, modifying, or triggering GitHub Actions workflows or runs.
- Changing repository settings, permissions, or secrets.

Do not interpret a request to implement, fix, test, review, or prepare code as permission to perform any of these operations.

- You may inspect Git status, diffs, and history when useful, provided the inspection does not change repository state.
- You may edit project files and run appropriate local checks.
- You may recommend exact Git commands for the user to run.
- Before any explicitly authorized Git or GitHub operation, complete the relevant checks first when applicable and report their results.
- If authorization is ambiguous, stop and ask the user.
- Never perform commits, pushes, pulls, merges, or other repository-changing operations autonomously.

## 4. Branching and Pull Request Workflow

### One Task per Branch

- Every independent feature, bug fix, documentation change, test addition, refactoring task, maintenance task, or other meaningful change must have its own branch.
- All task branches must ultimately target `main` through a pull request.
- Do not combine unrelated tasks in one branch or pull request.
- Tightly coupled changes required to deliver one coherent outcome may share a branch.
- Do not commit task changes directly to `main`.
- Do not start a new task on a branch containing unrelated, unfinished work.
- Branch creation, switching, and other Git operations remain subject to the authorization rules in Section 3.

### Branch Naming Convention

Use the format `<type>/<short-description>`.

Allowed types and examples:

- `feat/` — new functionality; e.g. `feat/data-preprocessing`
- `fix/` — bug fixes; e.g. `fix/missing-value-handling`
- `docs/` — documentation changes; e.g. `docs/model-evaluation-guide`
- `ref/` — code restructuring without intended behavior changes; e.g. `ref/training-pipeline`
- `test/` — test-related changes; e.g. `test/class-imbalance-metrics`
- `chore/` — maintenance, dependencies, configuration, and general housekeeping; e.g. `chore/update-dev-dependencies`
- `perf/` — performance improvements; e.g. `perf/optimize-feature-engineering`
- `ci/` — continuous integration and automation; e.g. `ci/add-pytest-workflow`
- `revert/` — reverting a previous change; e.g. `revert/remove-old-preprocessing`

Naming rules:

- Use lowercase characters, numbers where useful, and hyphens between words.
- Keep names concise, descriptive, and specific to the task.
- Avoid spaces, colons, and unnecessary identifiers.
- Choose the prefix that best represents the primary purpose of the change.
- If uncertain about the appropriate branch name, suggest one and ask the user.

### Pull Request Requirements

- Every task branch must be proposed for merging into `main` through a pull request.
- Keep pull requests focused on one coherent task.
- Include a clear title, summary, rationale, and testing information.
- Describe relevant risks, limitations, data implications, and follow-up work.
- Do not claim tests or quality checks passed unless they were actually executed and verified.
- All applicable tests, unit tests, linters, formatters, type checkers, and other configured checks must pass before a change is considered ready for review.
- Do not bypass required checks or weaken project quality controls to get a passing result.
- Do not merge a pull request unless the user explicitly authorizes that specific merge.

### Agent Authorization and Git Safety

- Agents may recommend branch names and explain the proposed Git workflow.
- Agents must not create, switch, or delete branches; stage files; commit; push; pull; merge; rebase; or perform other repository-changing Git operations without the user's explicit authorization.
- Agents must not create, update, or merge pull requests, or trigger or modify GitHub Actions workflows, without the user's explicit authorization.
- A request to implement a feature, fix a bug, or prepare a pull request does not automatically authorize all Git or GitHub operations.
- Before any authorized operation, report relevant validation results and any outstanding issues.
- If authorization is ambiguous, ask before proceeding.

### Expected Completion Handoff

When a task is implemented but not yet committed or submitted for review, report:

1. The proposed branch name.
2. A summary of the changes.
3. Tests and quality checks run, including results.
4. Any checks that failed or could not be run.
5. A suggested commit message, if useful.
6. A proposed pull request title and description.

Leave branch creation, staging, committing, pushing, and pull-request creation to the user unless explicitly authorized.

## 5. Python Environment and Dependencies

- Use `uv` to manage the Python environment and project dependencies.
- Respect the Python version and dependency constraints declared in `pyproject.toml`.
- Prefer reproducible dependency management and keep `uv.lock` consistent when applicable.
- Do not install global Python packages or change the system Python installation for project needs.
- Do not add dependencies unless they are needed for the task.

## 6. Data Science and Machine Learning

- Prevent data leakage between training, validation, and test data.
- Fit preprocessing steps only on the appropriate training data.
- Use reproducible random seeds where applicable.
- Choose evaluation metrics appropriate to the classification problem, including class imbalance where relevant.
- Keep training, evaluation, and preprocessing logic understandable and testable.
- Do not assume that a high score alone demonstrates a good model.
- Never overwrite, delete, or commit datasets or model artifacts without an explicit reason and appropriate permission.
- Do not expose sensitive data, credentials, or secrets.

## 7. Security and Data Integrity

- Never place passwords, tokens, private keys, or other secrets in source code or committed files.
- Respect `.gitignore` and avoid committing virtual environments, caches, local configuration, or generated artifacts unless explicitly required.
- Do not delete or overwrite user data without asking first.
- Treat external data and downloaded files as untrusted input.

## 8. Completion Report

When finishing a task, provide a concise summary containing:

1. What changed.
2. Tests and quality checks run, with their results.
3. Checks that failed or could not be run, and why.
4. Any remaining risks, assumptions, or follow-up work.

Do not state that a task is fully complete if important checks remain outstanding.