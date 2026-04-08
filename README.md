# AIMv2

`aimv2` is a Rust CLI for running AIM, an agentic mathematical assistant, inside a local workspace. It is designed for proof exploration, theorem tracking, review passes, and iterative repair of proof paths.

## Installation

It is recommended to install `aimv2` with `cargo install` so it can be used from any workspace:

```bash
cargo install --path .
```

After installation, you can run `aimv2` from any project directory:

```bash
cd /path/to/workspace
aimv2
```

During development, you can still run it directly from this repository with:

```bash
cargo run -- --help
```

## Configuration

`aimv2` reads API settings from the environment. The most common setup is:

```bash
export OPENAI_API_KEY=...
```

You can also use the AIM-prefixed variables:

- `AIM_API_KEY`
- `AIM_BASE_URL`

If both are absent, `aimv2` will also look for:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

When starting inside a workspace, `aimv2` will load a local `.env` file from that workspace root if one exists.

## Recommended Workflow

Install `aimv2` once with `cargo install`, then use it inside the workspace where you want AIM to read files, maintain history, and build theorem-graph entries.

Typical usage:

```bash
cd /path/to/workspace
aimv2 --model gpt-5.4
```

If you want AIM to inspect files or run local commands, enable the shell tool:

```bash
aimv2 --enable-shell
```

If you want shell commands to run without interactive confirmation:

```bash
aimv2 --enable-shell --auto
```

## Skills

Minimal Agent Skills support is available when the shell tool is enabled. On startup, `aimv2` scans these locations for skill folders that contain a `SKILL.md` file:

- `<project>/.aim/skills/`
- `<project>/.agents/skills/`
- `~/.aim/skills/`
- `~/.agents/skills/`

Each discovered skill is loaded from the heading and leading description in `SKILL.md`, and those metadata are injected into the system prompt so the agent can choose and use the skill.

Skills are not loaded unless `--enable-shell` is active, because the prompt instructions tell the agent to read `SKILL.md` through the shell tool when a skill applies.

You can also add extra scan roots with `--external-skills`:

```bash
aimv2 --enable-shell --external-skills /path/to/skills
aimv2 --enable-shell --external-skills /path/to/team-skills --external-skills /path/to/private-skills
```

## CLI Examples

Start a new interactive session:

```bash
aimv2
```

Choose reviewer mode and parameters:

```bash
aimv2 --reviewer simple --simple-reviews 4
aimv2 --reviewer progressive --progressive-iterations 3
```

Resume the latest session for the current workspace:

```bash
aimv2 resume --last
```

Inspect a saved theorem entry:

```bash
aimv2 view --last --id 12
```

Inspect a theorem entry together with its dependency path:

```bash
aimv2 view --last --path-to 12
```

## Interactive Commands

Inside the CLI session, the following slash commands are available:

- `/help`: show interactive help
- `/continue`: retry from the current saved session without adding a new user message
- `/compact`: manually trigger pre-turn history compaction
- `/reviewer simple`
- `/reviewer progressive`
- `/reviews <N>`: set the simple reviewer parallel review count
- `/iterations <N>`: set the progressive reviewer iteration limit
- `/auto on`: enable shell auto approval for the current run
- `/auto off`: require approval before shell commands
- `/exit`
- `/quit`

## Theorem Graph and Reviewers

AIM maintains a theorem graph in the session log. It uses this graph to record:

- context entries gathered from the user, files, or other sources
- theorem entries derived by the agent
- proof dependencies
- review counts
- reviewer comments when flaws are found

There are currently two reviewer modes:

- `simple`: runs several independent review sub-sessions in parallel
- `progressive`: starts with a broad review and, if no error is found, recursively focuses on chunks of the proof

In both modes, the reviewer uses the `comment` tool to append detected proof issues to the reviewed theorem entry.

## Session Logs

Each session is saved as a JSON log. By default, logs are stored in a temp directory managed by `aimv2`. You can also choose an explicit path:

```bash
aimv2 --log-path .aim/session.json
```

Relative log paths are resolved from the current workspace.

## Notes

- `aimv2` is intended to be launched from the workspace you want AIM to reason about.
- Shell access is workspace-scoped and should be enabled only when needed.
- Background reviewer and compaction sessions run silently; only the main session prints normal assistant replies and tool activity.
