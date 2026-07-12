# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a blank project scaffold. As code is introduced, keep the root focused on configuration and documentation. Place application code in `src/`, tests in `tests/`, reusable scripts in `scripts/`, and non-code resources in `assets/`. Mirror source paths in the test tree—for example, test `src/solver/reward.py` with `tests/solver/test_reward.py`. Document any intentional departure from this layout in the pull request that introduces it.

## Build, Test, and Development Commands

No build system or package manager has been configured yet. When adding one, expose a small, predictable command set and update this section. Recommended entry points are:

- `make setup` — install development dependencies.
- `make test` — run the complete automated test suite.
- `make lint` — run formatters, linters, and static checks.
- `make run` — start the project locally.

Keep these targets as thin wrappers around the native tooling so local development and CI use the same commands.

## Coding Style & Naming Conventions

Follow the standard formatter and linter for the chosen language, committed as project configuration rather than relying on editor defaults. Use spaces for indentation unless the language ecosystem requires otherwise. Prefer descriptive names: `snake_case` for Python modules and functions, `PascalCase` for types, and kebab-case for documentation filenames. Keep modules focused and avoid unrelated cleanup in feature changes.

## Testing Guidelines

Add tests with every behavior change and bug fix. Tests should be deterministic, isolated from external services by default, and named after the behavior under test. Store fixtures under `tests/fixtures/`. New tooling should provide one command that runs all tests from the repository root and should fail on test errors or unmet coverage thresholds.

## Commit & Pull Request Guidelines

There is no Git history from which to infer an existing convention. Use short, imperative commit subjects, optionally following Conventional Commits, such as `feat: add reward parser` or `fix: handle empty response`. Keep commits scoped and independently understandable.

Pull requests should explain the motivation, summarize the implementation, list verification commands, and link relevant issues. Include screenshots or logs for visible behavior changes. Note configuration changes, migrations, and follow-up work explicitly; request review only after automated checks pass.

## Safety & Artifact Rules

Store model caches, datasets, outputs, and checkpoints only under `/root/autodl-tmp`. Never commit generated artifacts. Do not batch-delete files or directories: commands such as `rm -rf`, recursive `rm`, and recursive `rmdir` are prohibited. If deletion is necessary, identify and remove only one explicit file after confirming its path. Do not execute generated code unless a verified isolation backend is active; ordinary subprocess execution is not a security sandbox.
