# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Rust CLI application. The crate root is [Cargo.toml](/Users/huangyanxing/Desktop/docs/Research/AIMv2/Cargo.toml), and the executable entrypoint is [src/core.rs](/Users/huangyanxing/Desktop/docs/Research/AIMv2/src/core.rs). Supporting modules live in `src/`: `llm.rs` handles model requests, `history.rs` manages session history and token compaction, and `ui.rs` contains terminal formatting and prompts. There is no separate `tests/` or `assets/` directory yet; add integration tests under `tests/` if coverage expands.

## Build, Test, and Development Commands
Use standard Cargo workflows from the repository root:

- `cargo run -- --help`: show available CLI flags.
- `cargo run -- --model gpt-5.4`: run the interactive CLI with explicit model selection.
- `cargo check`: verify the project compiles quickly without producing a release binary.
- `cargo test`: run unit and integration tests. At present this builds the crate and runs `0` tests.
- `cargo fmt --check`: verify Rust formatting before opening a PR.

If you add dependencies or environment-based behavior, document required variables in `README.md` and keep local secrets in `.env`.

## Coding Style & Naming Conventions
Follow `rustfmt` defaults: 4-space indentation, trailing commas where idiomatic, and imports grouped by standard library, third-party crates, then local modules. Use `snake_case` for functions, variables, and modules; `PascalCase` for structs and enums; and `SCREAMING_SNAKE_CASE` for constants such as `DEFAULT_BASE_URL`. Keep modules focused: core flow in `core.rs`, presentation in `ui.rs`, transport in `llm.rs`, and persistence logic in `history.rs`.

## Testing Guidelines
Prefer narrow unit tests near the code they cover with `#[cfg(test)]` blocks, and use `tests/` for CLI- or workflow-level integration tests. Name tests for observable behavior, for example `loads_env_from_workspace_root` or `compacts_history_before_turn`. Run `cargo test` before submitting changes; add tests for new parsing, history, or command execution branches.

## Commit & Pull Request Guidelines
The current history uses short, imperative commit messages (`Initial commit`). Continue with concise subjects such as `Add session resume flag` or `Refactor history compaction`. Pull requests should include a short description, the reason for the change, test evidence (`cargo test`, `cargo fmt --check`), and terminal screenshots only when UI output changes.
