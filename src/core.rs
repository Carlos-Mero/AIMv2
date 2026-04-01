mod history;
mod llm;
mod prompt;
mod theorem_graph;
mod ui;

use anyhow::{Context, Result, anyhow, bail};
use async_openai::types::chat::ReasoningEffort;
use chrono::{Local, TimeZone};
use clap::{Parser, Subcommand, ValueEnum};
use futures::future::try_join_all;
use history::{
    AssistantToolCall, CompactionMode, HistoryEntry, HistoryFile, SessionConfigSnapshot,
    build_messages, token_limit_for_model,
};
use llm::{
    LlmConfig, ToolCall, ToolMode, build_client, call_model, report_api_error,
    tool_arguments_as_object,
};
use prompt::{progressive_review_prompt, simple_review_prompt};
use serde::Deserialize;
use serde_json::Value;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use theorem_graph::TheoremEntryType;
use tokio::sync::Mutex as AsyncMutex;
use ui::{
    COLOR_BLUE, COLOR_BOLD, COLOR_CYAN, COLOR_DIM, COLOR_GREEN, COLOR_RED, COLOR_YELLOW,
    editor_prompt, print_background_error, print_background_wait, print_named_tool_call,
    print_statusline, print_tool_call, print_tool_result, prompt_for_approval, role_prefix, style,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-5.4";
const LOOP_LIMIT: usize = 1000000;
const PROGRESSIVE_REVIEW_MIN_CHUNK_LINES: usize = 4;

#[derive(Parser, Debug)]
#[command(
    name = "aimv2",
    version,
    about = "AI mathematician for local workspaces"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<CliCommand>,

    #[arg(
        long,
        default_value = DEFAULT_MODEL,
        help = "Model name used for chat completions"
    )]
    model: String,

    #[arg(
        long,
        value_enum,
        default_value_t = CliReasoningEffort::Medium,
        help = "Reasoning budget requested from the model"
    )]
    reasoning_effort: CliReasoningEffort,

    #[arg(
        long,
        help = "Override the history compaction threshold in estimated tokens"
    )]
    token_limit: Option<u64>,

    #[arg(
        long,
        help = "Auto-approve shell commands when the shell tool is enabled"
    )]
    auto: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = ReviewerKind::Progressive,
        help = "Reviewer strategy used by theorem_graph_review"
    )]
    reviewer: ReviewerKind,

    #[arg(
        long,
        default_value_t = 4,
        help = "Number of parallel review runs used by the simple reviewer"
    )]
    simple_reviews: u32,

    #[arg(
        long,
        default_value_t = 3,
        help = "Maximum iterations used by the progressive reviewer"
    )]
    progressive_iterations: u32,

    #[arg(
        long,
        global = true,
        value_name = "FILE",
        help = "Path to the session log file to load or write; relative paths are resolved from the current workspace"
    )]
    log_path: Option<PathBuf>,

    #[arg(
        long = "enable-shell",
        default_value_t = false,
        help = "Enable the optional shell tool for workspace inspection, editing, and experiments"
    )]
    enable_shell: bool,
}

#[derive(Subcommand, Debug)]
enum CliCommand {
    /// Resume a saved session. By default, sessions are discovered for the current workspace.
    Resume {
        #[arg(
            long,
            help = "Resume the most recent matching session without prompting"
        )]
        last: bool,
    },
    /// View theorem-graph contents from a saved session log.
    View {
        #[arg(long, help = "Use the most recent session for this workspace")]
        last: bool,
        #[arg(
            long,
            conflicts_with_all = ["path_to", "all"],
            help = "Print one theorem entry by id"
        )]
        id: Option<usize>,
        #[arg(
            long = "path-to",
            conflicts_with_all = ["id", "all"],
            help = "Print one theorem entry and all of its dependencies"
        )]
        path_to: Option<usize>,
        #[arg(
            long,
            conflicts_with_all = ["id", "path_to"],
            help = "Print all theorem-graph entries in id order"
        )]
        all: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl From<CliReasoningEffort> for ReasoningEffort {
    fn from(value: CliReasoningEffort) -> Self {
        match value {
            CliReasoningEffort::Minimal => ReasoningEffort::Minimal,
            CliReasoningEffort::Low => ReasoningEffort::Low,
            CliReasoningEffort::Medium => ReasoningEffort::Medium,
            CliReasoningEffort::High => ReasoningEffort::High,
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    llm: LlmConfig,
    base_url: String,
    history_token_limit: u64,
    reviewer: ReviewerConfig,
    workspace_root: PathBuf,
    enable_shell: bool,
}

struct App {
    history_path: PathBuf,
    session: Session,
}

#[derive(Clone)]
struct Session {
    client: llm::LlmClient,
    config: Config,
    theorem_graph: Arc<AsyncMutex<theorem_graph::TheoremGraph>>,
    history: HistoryFile,
    history_path: Option<Arc<PathBuf>>,
    auto_approve: Arc<Mutex<bool>>,
    allow_auto_compaction: bool,
    emit_output: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResumeMode {
    New,
    Select,
    Last,
}

#[derive(Debug)]
struct ShellRequest {
    command: String,
    workdir: Option<String>,
}

#[derive(Debug)]
struct CommandOutcome {
    success: bool,
    tool_content: String,
}

#[derive(Debug)]
struct SessionSummary {
    path: PathBuf,
    history: HistoryFile,
}

#[derive(Debug, Deserialize)]
struct ShellToolArgs {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TheoremGraphPushArgs {
    #[serde(rename = "type")]
    entry_type: TheoremEntryType,
    statement: String,
    proof: String,
    dependencies: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct TheoremGraphListArgs {
    start: usize,
    end: usize,
}

#[derive(Debug, Deserialize)]
struct TheoremGraphIdArgs {
    id: usize,
}

#[derive(Debug, Deserialize)]
struct TheoremGraphReviseArgs {
    id: usize,
    proof: String,
    dependencies: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ReviewerKind {
    Simple,
    Progressive,
}

#[derive(Clone, Debug)]
struct ReviewerConfig {
    kind: ReviewerKind,
    simple_reviews: u32,
    progressive_iterations: u32,
}

impl Config {
    fn snapshot(&self, auto_approve: bool) -> SessionConfigSnapshot {
        SessionConfigSnapshot {
            model: self.llm.model.clone(),
            reasoning_effort: reasoning_effort_label(&self.llm.reasoning_effort).to_string(),
            base_url: self.base_url.clone(),
            history_token_limit: self.history_token_limit,
            reviewer_kind: reviewer_kind_label(self.reviewer.kind).to_string(),
            simple_reviews: self.reviewer.simple_reviews,
            progressive_iterations: self.reviewer.progressive_iterations,
            enable_shell: self.enable_shell,
            auto_approve,
        }
    }

    fn from_snapshot(snapshot: &SessionConfigSnapshot, workspace_root: PathBuf) -> Result<Self> {
        Ok(Self {
            llm: LlmConfig {
                model: snapshot.model.clone(),
                reasoning_effort: parse_reasoning_effort(&snapshot.reasoning_effort)?,
            },
            base_url: snapshot.base_url.clone(),
            history_token_limit: snapshot.history_token_limit.max(1),
            reviewer: ReviewerConfig {
                kind: parse_reviewer_kind(&snapshot.reviewer_kind)?,
                simple_reviews: snapshot.simple_reviews.max(1),
                progressive_iterations: snapshot.progressive_iterations.max(1),
            },
            workspace_root,
            enable_shell: snapshot.enable_shell,
        })
    }
}

impl ReviewerConfig {
    fn active_label(&self) -> &'static str {
        match self.kind {
            ReviewerKind::Simple => "simple",
            ReviewerKind::Progressive => "progressive",
        }
    }

    fn description(&self) -> String {
        match self.kind {
            ReviewerKind::Simple => format!(
                "simple reviewer with {} parallel reviews",
                self.simple_reviews
            ),
            ReviewerKind::Progressive => format!(
                "progressive reviewer with {} iterations",
                self.progressive_iterations
            ),
        }
    }
}

fn reasoning_effort_label(value: &ReasoningEffort) -> &'static str {
    match value {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
    }
}

fn parse_reasoning_effort(value: &str) -> Result<ReasoningEffort> {
    match value.trim().to_ascii_lowercase().as_str() {
        "none" => Ok(ReasoningEffort::None),
        "minimal" => Ok(ReasoningEffort::Minimal),
        "low" => Ok(ReasoningEffort::Low),
        "medium" => Ok(ReasoningEffort::Medium),
        "high" => Ok(ReasoningEffort::High),
        "xhigh" => Ok(ReasoningEffort::Xhigh),
        other => bail!("unsupported saved reasoning effort: {other}"),
    }
}

fn reviewer_kind_label(value: ReviewerKind) -> &'static str {
    match value {
        ReviewerKind::Simple => "simple",
        ReviewerKind::Progressive => "progressive",
    }
}

fn parse_reviewer_kind(value: &str) -> Result<ReviewerKind> {
    match value.trim().to_ascii_lowercase().as_str() {
        "simple" => Ok(ReviewerKind::Simple),
        "progressive" => Ok(ReviewerKind::Progressive),
        other => bail!("unsupported saved reviewer kind: {other}"),
    }
}

#[derive(Debug, Deserialize)]
struct TheoremGraphCommentArgs {
    id: usize,
    comment: String,
}

enum ReviewRunOutcome {
    NoError,
    Commented,
}

#[derive(Clone, Copy)]
enum SessionMode {
    Normal,
    Review,
}

struct SessionOutcome {
    assistant_reply: Option<String>,
    review_outcome: ReviewRunOutcome,
}

struct ReviewBatchSummary {
    reviews_run: u32,
    commented_reviews: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(CliCommand::View { .. }) = &cli.command {
        return run_view(&cli);
    }
    App::load(cli).await?.run().await
}

impl App {
    async fn load(cli: Cli) -> Result<Self> {
        let workspace_root = env::current_dir().context("failed to determine current directory")?;
        let env_path = workspace_root.join(".env");
        if env_path.exists() {
            let _ = dotenvy::from_path(&env_path);
        }

        let resume = match cli.command {
            Some(CliCommand::Resume { last: true }) => ResumeMode::Last,
            Some(CliCommand::Resume { last: false }) => ResumeMode::Select,
            Some(CliCommand::View { .. }) => ResumeMode::New,
            None => ResumeMode::New,
        };

        let api_key = read_env(&["AIM_API_KEY", "OPENAI_API_KEY"])
            .ok_or_else(|| anyhow!("missing AIM_API_KEY or OPENAI_API_KEY"))?;
        let base_url = read_env(&["AIM_BASE_URL", "OPENAI_BASE_URL"])
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let explicit_log_path = resolve_log_path(&workspace_root, cli.log_path.clone())?;
        let (history_path, history) =
            load_or_create_session(&workspace_root, explicit_log_path.as_deref(), resume)?;
        let fallback_config = config_from_cli(&cli, workspace_root.clone(), base_url);
        let (config, auto_approve) =
            resolve_session_settings(&history, resume, fallback_config, cli.auto)?;

        let theorem_graph = Arc::new(AsyncMutex::new(history.theorem_graph.clone()));
        let auto_approve = Arc::new(Mutex::new(auto_approve));
        let session = Session {
            client: build_client(&api_key, &config.base_url),
            config,
            theorem_graph,
            history,
            history_path: Some(Arc::new(history_path.clone())),
            auto_approve,
            allow_auto_compaction: true,
            emit_output: true,
        };

        Ok(Self {
            history_path,
            session,
        })
    }

    async fn run(&mut self) -> Result<()> {
        println!("{}", style(COLOR_BOLD, "aimv2"));
        println!(
            "{} {}",
            style(COLOR_DIM, "workspace:"),
            self.session.config.workspace_root.display()
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "model:"),
            self.session.config.llm.model
        );
        println!(
            "{} {:?}",
            style(COLOR_DIM, "reasoning effort:"),
            self.session.config.llm.reasoning_effort
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "history token limit:"),
            self.session.config.history_token_limit
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "reviewer:"),
            self.session.config.reviewer.active_label()
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "simple reviews:"),
            self.session.config.reviewer.simple_reviews
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "progressive iterations:"),
            self.session.config.reviewer.progressive_iterations
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "shell tool:"),
            if self.session.config.enable_shell {
                style(COLOR_GREEN, "enabled")
            } else {
                style(COLOR_DIM, "disabled")
            }
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "session:"),
            self.session.history.session_id
        );
        println!(
            "{} {}",
            style(COLOR_DIM, "history:"),
            self.history_path.display()
        );
        if self.session.config.enable_shell {
            println!(
                "{} {}",
                style(COLOR_DIM, "approval mode:"),
                if self.session.auto_approve() {
                    style(COLOR_YELLOW, "auto")
                } else {
                    style(COLOR_CYAN, "ask")
                }
            );
        }
        println!("{} /help", style(COLOR_DIM, "type"));
        self.print_history();

        let mut editor =
            rustyline::DefaultEditor::new().context("failed to initialize line editor")?;

        loop {
            self.print_statusline();
            let line = match editor.readline(&editor_prompt("you")) {
                Ok(line) => line,
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    println!();
                    continue;
                }
                Err(rustyline::error::ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(err) => return Err(err).context("failed to read user input"),
            };

            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let _ = editor.add_history_entry(line.as_str());

            match line.as_str() {
                "/exit" | "/quit" => break,
                "/continue" => {
                    if let Err(err) = self.continue_turn().await {
                        eprintln!("{} {err:#}", style(COLOR_RED, "error>"));
                    }
                    continue;
                }
                "/help" => {
                    print_repl_help(
                        self.session.auto_approve(),
                        self.session.config.enable_shell,
                        &self.session.config.reviewer,
                        &self.history_path,
                    );
                    continue;
                }
                "/compact" => {
                    match self.compact_history(CompactionMode::BeforeTurn, true).await {
                        Ok(true) => println!("{}", style(COLOR_YELLOW, "history compacted")),
                        Ok(false) => println!("{}", style(COLOR_YELLOW, "nothing to compact yet")),
                        Err(err) => eprintln!("{} {err:#}", style(COLOR_RED, "error>")),
                    }
                    continue;
                }
                "/auto on" => {
                    if !self.session.config.enable_shell {
                        println!("{}", style(COLOR_YELLOW, "shell tool is disabled"));
                    } else {
                        self.session.set_auto_approve(true);
                        println!("{}", style(COLOR_YELLOW, "auto approval enabled"));
                    }
                    continue;
                }
                "/auto off" => {
                    if !self.session.config.enable_shell {
                        println!("{}", style(COLOR_YELLOW, "shell tool is disabled"));
                    } else {
                        self.session.set_auto_approve(false);
                        println!("{}", style(COLOR_YELLOW, "auto approval disabled"));
                    }
                    continue;
                }
                _ => {}
            }

            if let Some(kind) = parse_reviewer_command(&line) {
                self.session.config.reviewer.kind = kind;
                println!(
                    "{} {}",
                    style(COLOR_YELLOW, "reviewer set to"),
                    self.session.config.reviewer.description()
                );
                continue;
            }
            if let Some(reviews) = parse_u32_command(&line, "/reviews") {
                self.session.config.reviewer.simple_reviews = reviews.max(1);
                println!(
                    "{} {}",
                    style(COLOR_YELLOW, "simple reviews set to"),
                    self.session.config.reviewer.simple_reviews
                );
                continue;
            }
            if let Some(iterations) = parse_u32_command(&line, "/iterations") {
                self.session.config.reviewer.progressive_iterations = iterations.max(1);
                println!(
                    "{} {}",
                    style(COLOR_YELLOW, "progressive iterations set to"),
                    self.session.config.reviewer.progressive_iterations
                );
                continue;
            }

            if let Err(err) = self.run_turn(line).await {
                eprintln!("{} {err:#}", style(COLOR_RED, "error>"));
            }
        }

        Ok(())
    }

    async fn run_turn(&mut self, user_input: String) -> Result<()> {
        self.compact_history(CompactionMode::BeforeTurn, false)
            .await?;
        self.session.history.push_user(user_input);
        self.save_history().await?;
        self.session.run_agent_loop(SessionMode::Normal).await?;
        self.save_history().await
    }

    async fn continue_turn(&mut self) -> Result<()> {
        self.session.run_agent_loop(SessionMode::Normal).await?;
        self.save_history().await
    }

    async fn compact_history(&mut self, mode: CompactionMode, force: bool) -> Result<bool> {
        if self.session.history.entries.is_empty() {
            return Ok(false);
        }
        if !force
            && !self
                .session
                .history
                .needs_compaction(self.session.config.history_token_limit)
        {
            return Ok(false);
        }

        let resume_user = if mode == CompactionMode::MidTurn {
            self.session.history.last_user_content()
        } else {
            None
        };
        let mut sub_session = self.session.spawn_subsession(false, false);
        sub_session
            .history
            .push_user(prompt::compaction_prompt(mode));
        let base_history = self.session.history.clone();
        if self.session.emit_output {
            print_background_wait("waiting for background compaction session to complete");
        }
        let outcome = Box::pin(sub_session.run_agent_loop(SessionMode::Normal))
            .await
            .map_err(|err| {
                if self.session.emit_output {
                    print_background_error(&format!("background compaction failed: {err:#}"));
                }
                err
            })?;
        self.session
            .history
            .apply_usage_delta(usage_delta(&base_history, &sub_session.history));
        let summary = outcome
            .assistant_reply
            .ok_or_else(|| anyhow!("compaction session did not produce an assistant response"))?;
        self.session.history.apply_compaction(summary, resume_user);
        self.save_history().await?;
        Ok(true)
    }

    fn print_history(&self) {
        if self.session.history.entries.is_empty() {
            return;
        }

        println!("{}", style(COLOR_BOLD, "resumed history"));
        for entry in &self.session.history.entries {
            match entry {
                HistoryEntry::System { content, .. } => {
                    println!("{}{}", role_prefix("system", COLOR_YELLOW), content);
                }
                HistoryEntry::User { content, .. } => {
                    println!("{}{}", role_prefix("you", COLOR_BLUE), content);
                }
                HistoryEntry::Assistant { content, .. } => {
                    if !content.trim().is_empty() {
                        println!("{}{}", role_prefix("assistant", COLOR_GREEN), content);
                    }
                }
                HistoryEntry::Tool { content, .. } => {
                    print_tool_result(
                        &normalize_tool_content_for_display(content),
                        tool_content_success(content),
                    );
                }
            }
        }
    }

    fn print_statusline(&self) {
        print_statusline(
            &self.session.config.workspace_root,
            &self.session.config.llm.model,
            self.session.history.active_token_usage(),
            self.session.config.history_token_limit,
            self.session.history.total_input_usage(),
            self.session.history.total_output_usage(),
        );
    }

    async fn save_history(&mut self) -> Result<()> {
        self.session.persist_history().await
    }
}

impl Session {
    fn auto_approve(&self) -> bool {
        *self
            .auto_approve
            .lock()
            .expect("auto approval mutex poisoned")
    }

    fn set_auto_approve(&self, value: bool) {
        *self
            .auto_approve
            .lock()
            .expect("auto approval mutex poisoned") = value;
    }

    fn spawn_subsession(&self, allow_auto_compaction: bool, emit_output: bool) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            theorem_graph: Arc::clone(&self.theorem_graph),
            history: self.history.clone_at_model_boundary(),
            history_path: None,
            auto_approve: Arc::clone(&self.auto_approve),
            allow_auto_compaction,
            emit_output,
        }
    }

    async fn snapshot_history(&self) -> HistoryFile {
        let mut snapshot = self.history.clone();
        snapshot.last_active_at_ms = now_millis();
        snapshot.theorem_graph = self.theorem_graph.lock().await.clone();
        snapshot.session_config = Some(self.config.snapshot(self.auto_approve()));
        snapshot
    }

    async fn persist_history(&self) -> Result<()> {
        let Some(history_path) = &self.history_path else {
            return Ok(());
        };

        let snapshot = self.snapshot_history().await;
        let text = serde_json::to_string_pretty(&snapshot).context("failed to encode history")?;
        fs::write(history_path.as_ref(), text)
            .with_context(|| format!("failed to write history file {}", history_path.display()))
    }

    async fn run_agent_loop(&mut self, mode: SessionMode) -> Result<SessionOutcome> {
        for _ in 0..LOOP_LIMIT {
            if self.allow_auto_compaction {
                self.compact_history_if_needed(CompactionMode::MidTurn)
                    .await?;
            }
            let messages = build_messages(
                &self.config.workspace_root,
                &self.history.entries,
                self.config.enable_shell,
                &self.config.reviewer.description(),
            );
            let mut started_stream = false;
            let emit_output = self.emit_output;
            let reply = match call_model(
                &self.client,
                &self.config.llm,
                messages,
                Some(ToolMode::Agent {
                    enable_shell: self.config.enable_shell,
                }),
                self.emit_output,
                |chunk| {
                    if !emit_output {
                        return;
                    }
                    if !started_stream {
                        print!("{}", role_prefix("assistant", COLOR_GREEN));
                        io::stdout().flush().ok();
                        started_stream = true;
                    }
                    print!("{chunk}");
                    io::stdout().flush().ok();
                },
            )
            .await
            {
                Ok(reply) => reply,
                Err(err) => {
                    if self.emit_output {
                        report_api_error(&err);
                    }
                    return Err(err);
                }
            };
            if started_stream {
                println!();
            }

            self.history.note_api_usage(
                reply.input_tokens,
                reply.output_tokens,
                reply.total_tokens,
            );
            let assistant_reply = if reply.content.trim().is_empty() {
                None
            } else {
                Some(reply.content.clone())
            };
            let assistant_tool_calls = reply
                .tool_calls
                .iter()
                .map(|call| AssistantToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            self.history.push_assistant(
                reply.content.clone(),
                reply.reasoning.clone(),
                assistant_tool_calls,
            );
            self.persist_history().await?;

            if reply.tool_calls.is_empty() {
                return Ok(SessionOutcome {
                    assistant_reply,
                    review_outcome: ReviewRunOutcome::NoError,
                });
            }

            let review_outcome = self.handle_tool_calls(reply.tool_calls).await?;
            if matches!(mode, SessionMode::Review)
                && matches!(review_outcome, ReviewRunOutcome::Commented)
            {
                return Ok(SessionOutcome {
                    assistant_reply,
                    review_outcome,
                });
            }
        }

        Err(anyhow!(
            "agent loop exceeded {LOOP_LIMIT} steps without producing a final response"
        ))
    }

    async fn compact_history_if_needed(&mut self, mode: CompactionMode) -> Result<bool> {
        if self.history.entries.is_empty()
            || !self
                .history
                .needs_compaction(self.config.history_token_limit)
        {
            return Ok(false);
        }

        let base_history = self.history.clone();
        let resume_user = if mode == CompactionMode::MidTurn {
            self.history.last_user_content()
        } else {
            None
        };
        let mut sub_session = self.spawn_subsession(false, false);
        sub_session
            .history
            .push_user(prompt::compaction_prompt(mode));
        if self.emit_output {
            print_background_wait("waiting for background compaction session to complete");
        }
        let outcome = Box::pin(sub_session.run_agent_loop(SessionMode::Normal))
            .await
            .map_err(|err| {
                if self.emit_output {
                    print_background_error(&format!("background compaction failed: {err:#}"));
                }
                err
            })?;
        self.history
            .apply_usage_delta(usage_delta(&base_history, &sub_session.history));
        let summary = outcome
            .assistant_reply
            .ok_or_else(|| anyhow!("compaction session did not produce an assistant response"))?;
        self.history.apply_compaction(summary, resume_user);
        Ok(true)
    }

    async fn handle_tool_calls(&mut self, tool_calls: Vec<ToolCall>) -> Result<ReviewRunOutcome> {
        let mut review_outcome = ReviewRunOutcome::NoError;
        for tool_call in tool_calls {
            let (success, content, current_review_outcome) = match tool_call.name.as_str() {
                "shell_tool" => {
                    if !self.config.enable_shell {
                        (
                            false,
                            "shell_tool is disabled for this session; restart with --enable-shell to allow workspace commands".to_string(),
                            ReviewRunOutcome::NoError,
                        )
                    } else {
                        match parse_shell_tool_args(&tool_call)
                            .and_then(|request| self.run_shell(request))
                        {
                            Ok(outcome) => (
                                outcome.success,
                                outcome.tool_content,
                                ReviewRunOutcome::NoError,
                            ),
                            Err(err) => (
                                false,
                                format!("tool error: {err:#}"),
                                ReviewRunOutcome::NoError,
                            ),
                        }
                    }
                }
                "theorem_graph_push" => match self.handle_theorem_graph_push(&tool_call).await {
                    Ok(content) => (true, content, ReviewRunOutcome::NoError),
                    Err(err) => (
                        false,
                        format!("tool error: {err:#}"),
                        ReviewRunOutcome::NoError,
                    ),
                },
                "theorem_graph_list" => match self.handle_theorem_graph_list(&tool_call).await {
                    Ok(content) => (true, content, ReviewRunOutcome::NoError),
                    Err(err) => (
                        false,
                        format!("tool error: {err:#}"),
                        ReviewRunOutcome::NoError,
                    ),
                },
                "theorem_graph_list_deps" => {
                    match self.handle_theorem_graph_list_deps(&tool_call).await {
                        Ok(content) => (true, content, ReviewRunOutcome::NoError),
                        Err(err) => (
                            false,
                            format!("tool error: {err:#}"),
                            ReviewRunOutcome::NoError,
                        ),
                    }
                }
                "theorem_graph_examine" => {
                    match self.handle_theorem_graph_examine(&tool_call).await {
                        Ok(content) => (true, content, ReviewRunOutcome::NoError),
                        Err(err) => (
                            false,
                            format!("tool error: {err:#}"),
                            ReviewRunOutcome::NoError,
                        ),
                    }
                }
                "theorem_graph_review" => {
                    match self.handle_theorem_graph_review(&tool_call).await {
                        Ok(content) => (true, content, ReviewRunOutcome::NoError),
                        Err(err) => (
                            false,
                            format!("tool error: {err:#}"),
                            ReviewRunOutcome::NoError,
                        ),
                    }
                }
                "theorem_graph_comment" => {
                    match self.handle_theorem_graph_comment(&tool_call).await {
                        Ok(content) => (true, content, ReviewRunOutcome::Commented),
                        Err(err) => (
                            false,
                            format!("tool error: {err:#}"),
                            ReviewRunOutcome::NoError,
                        ),
                    }
                }
                "theorem_graph_revise" => {
                    match self.handle_theorem_graph_revise(&tool_call).await {
                        Ok(content) => (true, content, ReviewRunOutcome::NoError),
                        Err(err) => (
                            false,
                            format!("tool error: {err:#}"),
                            ReviewRunOutcome::NoError,
                        ),
                    }
                }
                other => (
                    false,
                    format!("unsupported tool: {other}"),
                    ReviewRunOutcome::NoError,
                ),
            };

            let display = if tool_call.name == "shell_tool" {
                normalize_tool_content_for_display(&content)
            } else {
                content.clone()
            };
            if self.emit_output {
                print_tool_result(&display, success);
            }
            self.history
                .push_tool(tool_call.id, tool_call.name, content);
            self.persist_history().await?;
            if matches!(current_review_outcome, ReviewRunOutcome::Commented) {
                review_outcome = ReviewRunOutcome::Commented;
            }
        }
        Ok(review_outcome)
    }

    async fn handle_theorem_graph_push(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphPushArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call(
                "theorem_graph_push",
                &format!(
                    "type: {}\nstatement: {}\ndependencies: {:?}",
                    match args.entry_type {
                        TheoremEntryType::Context => "context",
                        TheoremEntryType::Theorem => "theorem",
                    },
                    preview_text(&args.statement, 120),
                    args.dependencies
                ),
            );
        }
        self.theorem_graph.lock().await.push(
            args.entry_type,
            args.statement,
            args.proof,
            args.dependencies,
        )
    }

    async fn handle_theorem_graph_list(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphListArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call(
                "theorem_graph_list",
                &format!("start: {}\nend: {}", args.start, args.end),
            );
        }
        self.theorem_graph.lock().await.list(args.start, args.end)
    }

    async fn handle_theorem_graph_list_deps(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphIdArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call("theorem_graph_list_deps", &format!("id: {}", args.id));
        }
        self.theorem_graph.lock().await.list_deps(args.id)
    }

    async fn handle_theorem_graph_examine(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphIdArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call("theorem_graph_examine", &format!("id: {}", args.id));
        }
        self.theorem_graph.lock().await.examine(args.id)
    }

    async fn handle_theorem_graph_review(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphIdArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call("theorem_graph_review", &format!("id: {}", args.id));
        }
        match self.config.reviewer.kind {
            ReviewerKind::Simple => self.run_simple_reviewer(args.id).await,
            ReviewerKind::Progressive => self.run_progressive_reviewer(args.id).await,
        }
    }

    async fn handle_theorem_graph_comment(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphCommentArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call(
                "theorem_graph_comment",
                &format!(
                    "id: {}\ncomment: {}",
                    args.id,
                    preview_text(&args.comment, 120)
                ),
            );
        }
        self.theorem_graph
            .lock()
            .await
            .append_comment(args.id, args.comment)
    }

    async fn handle_theorem_graph_revise(&mut self, tool_call: &ToolCall) -> Result<String> {
        let args: TheoremGraphReviseArgs = parse_tool_args(tool_call)?;
        if self.emit_output {
            print_named_tool_call(
                "theorem_graph_revise",
                &format!(
                    "id: {}\nproof: {}\ndependencies: {:?}",
                    args.id,
                    preview_text(&args.proof, 120),
                    args.dependencies
                ),
            );
        }
        self.theorem_graph
            .lock()
            .await
            .revise(args.id, args.proof, args.dependencies)
    }

    async fn run_simple_reviewer(&mut self, id: usize) -> Result<String> {
        let review_count = self.config.reviewer.simple_reviews.max(1);
        self.theorem_graph.lock().await.examine(id)?;
        let prompts = (0..review_count)
            .map(|_| simple_review_prompt(id))
            .collect::<Vec<_>>();
        let summary = self.run_review_batch(prompts).await?;

        let total_reviews = self
            .theorem_graph
            .lock()
            .await
            .add_reviews(id, review_count)?;
        Ok(format!(
            "completed {review_count} simple reviews for theorem entry {id}; flagged by {}; total reviews: {total_reviews}",
            summary.commented_reviews
        ))
    }

    async fn run_progressive_reviewer(&mut self, id: usize) -> Result<String> {
        self.theorem_graph.lock().await.examine(id)?;

        let iteration_limit = self.config.reviewer.progressive_iterations.max(1);
        let mut reviews_run = 0_u32;
        let mut commented_reviews = 0_u32;

        let initial = self
            .run_review_batch(vec![simple_review_prompt(id)])
            .await?;
        reviews_run = reviews_run.saturating_add(initial.reviews_run);
        commented_reviews = commented_reviews.saturating_add(initial.commented_reviews);

        if commented_reviews == 0 {
            for iteration in 1..iteration_limit {
                let entry = self.theorem_graph.lock().await.entry_snapshot(id)?;
                let prompts = progressive_review_prompts(
                    id,
                    &entry.statement,
                    &entry.proof,
                    iteration as usize,
                );
                let batch = self.run_review_batch(prompts).await?;
                reviews_run = reviews_run.saturating_add(batch.reviews_run);
                commented_reviews = commented_reviews.saturating_add(batch.commented_reviews);
                if batch.commented_reviews > 0 {
                    break;
                }
            }
        }

        let total_reviews = self
            .theorem_graph
            .lock()
            .await
            .add_reviews(id, reviews_run)?;
        Ok(format!(
            "completed {reviews_run} progressive reviews for theorem entry {id}; flagged by {commented_reviews}; total reviews: {total_reviews}"
        ))
    }

    async fn run_review_batch(&mut self, prompts: Vec<String>) -> Result<ReviewBatchSummary> {
        let base_history = self.history.clone();
        let review_count = prompts.len();
        let review_tasks = prompts
            .into_iter()
            .enumerate()
            .map(|(index, prompt)| {
                let mut sub_session = self.spawn_subsession(false, false);
                sub_session.history.push_user(prompt);
                async move {
                    sub_session
                        .run_agent_loop(SessionMode::Review)
                        .await
                        .map(|outcome| (sub_session.history, outcome))
                        .with_context(|| format!("background review session {} failed", index + 1))
                }
            })
            .collect::<Vec<_>>();
        if self.emit_output {
            print_background_wait(&format!(
                "waiting for {review_count} background review session(s) to complete"
            ));
        }
        let outcomes = try_join_all(review_tasks).await.map_err(|err| {
            if self.emit_output {
                print_background_error(&format!("{err:#}"));
            }
            err
        })?;

        let mut total_input_delta = 0_u64;
        let mut total_output_delta = 0_u64;
        let mut total_delta = 0_u64;
        let mut commented_reviews = 0_u32;
        let reviews_run = outcomes.len() as u32;
        for (history, outcome) in outcomes {
            let (input_delta, output_delta, delta_total) = usage_delta(&base_history, &history);
            total_input_delta = total_input_delta.saturating_add(input_delta);
            total_output_delta = total_output_delta.saturating_add(output_delta);
            total_delta = total_delta.saturating_add(delta_total);
            if matches!(outcome.review_outcome, ReviewRunOutcome::Commented) {
                commented_reviews = commented_reviews.saturating_add(1);
            }
        }
        self.history
            .apply_usage_delta((total_input_delta, total_output_delta, total_delta));

        Ok(ReviewBatchSummary {
            reviews_run,
            commented_reviews,
        })
    }

    fn run_shell(&self, request: ShellRequest) -> Result<CommandOutcome> {
        let workdir = resolve_workdir(&self.config.workspace_root, request.workdir.as_deref())?;

        let workdir_text = workdir.display().to_string();
        if self.emit_output {
            print_tool_call(&request.command, &workdir_text);
        }
        let mut auto_approve = self
            .auto_approve
            .lock()
            .expect("auto approval mutex poisoned");
        if !self.emit_output && !*auto_approve {
            return Ok(CommandOutcome {
                success: false,
                tool_content: format_tool_content(
                    &request.command,
                    &workdir_text,
                    false,
                    "command rejected because interactive approval is unavailable in background sessions".to_string(),
                ),
            });
        }
        if !*auto_approve && !prompt_for_approval(&mut *auto_approve)? {
            return Ok(CommandOutcome {
                success: false,
                tool_content: format_tool_content(
                    &request.command,
                    &workdir.display().to_string(),
                    false,
                    "command rejected by user".to_string(),
                ),
            });
        }

        // let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let shell = "/bin/bash".to_string();
        let output = Command::new(&shell)
            .arg("-lc")
            .arg(&request.command)
            .current_dir(&workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("failed to run shell command via {shell}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut content = String::new();
        if !stdout.trim().is_empty() {
            content.push_str("stdout:\n");
            content.push_str(stdout.as_ref());
        }
        if !stderr.trim().is_empty() {
            if !content.is_empty() && !content.ends_with('\n') {
                content.push('\n');
            }
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str("stderr:\n");
            content.push_str(stderr.as_ref());
        }
        if content.trim().is_empty() {
            content.push_str("(no output)");
        }

        Ok(CommandOutcome {
            success: output.status.success(),
            tool_content: format_tool_content(
                &request.command,
                &workdir_text,
                output.status.success(),
                content,
            ),
        })
    }
}

fn parse_shell_tool_args(tool_call: &ToolCall) -> Result<ShellRequest> {
    let args: ShellToolArgs = parse_tool_args(tool_call)?;
    let command = args.command.trim().to_string();
    if command.is_empty() {
        bail!("shell_tool requires a non-empty command");
    }
    Ok(ShellRequest {
        command,
        workdir: args
            .workdir
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    })
}

fn parse_tool_args<T>(tool_call: &ToolCall) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let object = tool_arguments_as_object(&tool_call.arguments)?;
    serde_json::from_value::<T>(Value::Object(object)).with_context(|| {
        format!(
            "failed to decode arguments for tool {}: {}",
            tool_call.name,
            render_tool_arguments(&tool_call.arguments)
        )
    })
}

fn render_tool_arguments(arguments: &Value) -> String {
    serde_json::to_string(arguments).unwrap_or_else(|_| "<unserializable json>".to_string())
}

fn parse_reviewer_command(line: &str) -> Option<ReviewerKind> {
    let mut parts = line.split_whitespace();
    match (parts.next(), parts.next(), parts.next()) {
        (Some("/reviewer"), Some("simple"), None) => Some(ReviewerKind::Simple),
        (Some("/reviewer"), Some("progressive"), None) => Some(ReviewerKind::Progressive),
        _ => None,
    }
}

fn parse_u32_command(line: &str, prefix: &str) -> Option<u32> {
    let mut parts = line.split_whitespace();
    match (parts.next(), parts.next(), parts.next()) {
        (Some(command), Some(value), None) if command == prefix => value.parse().ok(),
        _ => None,
    }
}

fn usage_delta(base: &HistoryFile, updated: &HistoryFile) -> (u64, u64, u64) {
    (
        updated
            .total_input_tokens
            .saturating_sub(base.total_input_tokens),
        updated
            .total_output_tokens
            .saturating_sub(base.total_output_tokens),
        updated.total_tokens.saturating_sub(base.total_tokens),
    )
}

fn progressive_review_prompts(
    id: usize,
    statement: &str,
    proof: &str,
    iteration: usize,
) -> Vec<String> {
    let proof_chunks = split_proof_into_chunks(
        proof,
        2_usize.saturating_pow(iteration as u32),
        PROGRESSIVE_REVIEW_MIN_CHUNK_LINES,
    );
    proof_chunks
        .into_iter()
        .map(|chunk| progressive_review_prompt(id, statement, &chunk))
        .collect()
}

fn split_proof_into_chunks(
    proof: &str,
    target_chunks: usize,
    min_chunk_lines: usize,
) -> Vec<String> {
    let lines = proof.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return vec![String::new()];
    }

    let max_chunks = if min_chunk_lines == 0 {
        target_chunks.max(1)
    } else {
        target_chunks
            .max(1)
            .min((lines.len() / min_chunk_lines).max(1))
    };
    let mut chunks = Vec::with_capacity(max_chunks);
    let mut start = 0usize;

    for remaining_chunks in (1..=max_chunks).rev() {
        let remaining_lines = lines.len().saturating_sub(start);
        let chunk_len = remaining_lines.div_ceil(remaining_chunks).max(1);
        let end = (start + chunk_len).min(lines.len());
        chunks.push(lines[start..end].join("\n"));
        start = end;
    }

    chunks
}

fn format_tool_content(command: &str, workdir: &str, success: bool, output: String) -> String {
    format!("command: {command}\nworkdir: {workdir}\nsuccess: {success}\n\n{output}")
}

fn normalize_tool_content_for_display(content: &str) -> String {
    let mut lines = content.lines().peekable();

    for prefix in ["command: ", "workdir: ", "success: "] {
        match lines.peek().copied() {
            Some(line) if line.starts_with(prefix) => {
                lines.next();
            }
            _ => return content.to_string(),
        }
    }

    while matches!(lines.peek(), Some(line) if line.trim().is_empty()) {
        lines.next();
    }

    let normalized = lines.collect::<Vec<_>>().join("\n");
    if normalized.is_empty() {
        "(no output)".to_string()
    } else {
        normalized
    }
}

fn tool_content_success(content: &str) -> bool {
    content
        .lines()
        .nth(2)
        .and_then(|line| line.strip_prefix("success: "))
        .map(|value| value.trim() == "true")
        .unwrap_or(false)
}

fn resolve_workdir(workspace_root: &Path, requested: Option<&str>) -> Result<PathBuf> {
    let root = workspace_root.canonicalize().with_context(|| {
        format!(
            "workspace root does not exist: {}",
            workspace_root.display()
        )
    })?;
    let candidate = match requested {
        None | Some("") | Some(".") => root.clone(),
        Some(path) => root.join(path),
    };
    let candidate = candidate
        .canonicalize()
        .with_context(|| format!("workdir does not exist: {}", candidate.display()))?;

    if !candidate.starts_with(&root) {
        bail!("workdir escapes workspace: {}", candidate.display());
    }

    Ok(candidate)
}

fn config_from_cli(cli: &Cli, workspace_root: PathBuf, base_url: String) -> Config {
    let history_token_limit = cli
        .token_limit
        .unwrap_or_else(|| token_limit_for_model(&cli.model));
    Config {
        llm: LlmConfig {
            model: cli.model.clone(),
            reasoning_effort: cli.reasoning_effort.into(),
        },
        base_url,
        history_token_limit,
        reviewer: ReviewerConfig {
            kind: cli.reviewer,
            simple_reviews: cli.simple_reviews,
            progressive_iterations: cli.progressive_iterations,
        },
        workspace_root,
        enable_shell: cli.enable_shell,
    }
}

fn resolve_session_settings(
    history: &HistoryFile,
    resume: ResumeMode,
    fallback_config: Config,
    fallback_auto_approve: bool,
) -> Result<(Config, bool)> {
    if matches!(resume, ResumeMode::New) {
        return Ok((fallback_config, fallback_auto_approve));
    }

    match history.session_config.as_ref() {
        Some(snapshot) => Ok((
            Config::from_snapshot(snapshot, fallback_config.workspace_root.clone())?,
            snapshot.auto_approve,
        )),
        None => Ok((fallback_config, fallback_auto_approve)),
    }
}

fn load_or_create_session(
    workspace_root: &Path,
    explicit_log_path: Option<&Path>,
    resume: ResumeMode,
) -> Result<(PathBuf, HistoryFile)> {
    match resume {
        ResumeMode::New => {
            let history = HistoryFile {
                version: 7,
                session_id: format!("session-{}-{}", now_millis(), std::process::id()),
                workspace_root: workspace_root.display().to_string(),
                last_active_at_ms: now_millis(),
                total_input_tokens: 0,
                total_output_tokens: 0,
                total_tokens: 0,
                theorem_graph: theorem_graph::TheoremGraph::default(),
                session_config: None,
                entries: Vec::new(),
            };
            let path = match explicit_log_path {
                Some(path) => {
                    ensure_log_parent(path)?;
                    path.to_path_buf()
                }
                None => default_sessions_root()?.join(format!("{}.json", history.session_id)),
            };
            Ok((path, history))
        }
        ResumeMode::Last | ResumeMode::Select if explicit_log_path.is_some() => {
            let path = explicit_log_path.expect("checked is_some");
            let history = load_session_file(path)?;
            Ok((path.to_path_buf(), history))
        }
        ResumeMode::Last => {
            let mut sessions = list_sessions(workspace_root)?;
            let session = sessions
                .drain(..)
                .next()
                .ok_or_else(|| anyhow!("no previous sessions found for this workspace"))?;
            Ok((session.path, session.history))
        }
        ResumeMode::Select => {
            let sessions = list_sessions(workspace_root)?;
            if sessions.is_empty() {
                bail!("no previous sessions found for this workspace");
            }

            println!("available sessions:");
            for (index, session) in sessions.iter().enumerate() {
                println!("  {}. {}", index + 1, session.history.session_id);
                println!(
                    "     last active: {}",
                    format_last_active(session.history.last_active_at_ms)
                );
                println!(
                    "     last prompt: {}",
                    format_last_user_prompt(&session.history)
                );
                println!("     path: {}", session.path.display());
            }

            let selected = loop {
                print!("resume which session? [1-{}]> ", sessions.len());
                io::stdout().flush().ok();
                let mut line = String::new();
                io::stdin()
                    .read_line(&mut line)
                    .context("failed to read session selection")?;
                let trimmed = line.trim();
                let index = trimmed
                    .parse::<usize>()
                    .with_context(|| format!("invalid session selection: {trimmed}"))?;
                if (1..=sessions.len()).contains(&index) {
                    break index - 1;
                }
                println!("please enter a number between 1 and {}", sessions.len());
            };

            let session = sessions
                .into_iter()
                .nth(selected)
                .ok_or_else(|| anyhow!("invalid session selection"))?;
            Ok((session.path, session.history))
        }
    }
}

fn list_sessions(workspace_root: &Path) -> Result<Vec<SessionSummary>> {
    let root = default_sessions_root()?;
    if !root.exists() {
        return Ok(Vec::new());
    }

    let workspace_key = workspace_root.display().to_string();
    let mut sessions = Vec::new();
    for entry in
        fs::read_dir(&root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let history: HistoryFile = match serde_json::from_str(&text) {
            Ok(history) => history,
            Err(_) => continue,
        };
        if history.workspace_root == workspace_key {
            sessions.push(SessionSummary { path, history });
        }
    }

    sessions.sort_by(|left, right| {
        right
            .history
            .last_active_at_ms
            .cmp(&left.history.last_active_at_ms)
    });
    Ok(sessions)
}

fn load_session_file(path: &Path) -> Result<HistoryFile> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read session log {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode session log {}", path.display()))
}

fn load_history_from_path(path: &Path) -> Result<HistoryFile> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read session log {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to decode session log {}", path.display()))
}

fn run_view(cli: &Cli) -> Result<()> {
    let workspace_root = env::current_dir().context("failed to determine current directory")?;
    let explicit_log_path = resolve_log_path(&workspace_root, cli.log_path.clone())?;
    let CliCommand::View {
        last,
        id,
        path_to,
        all,
    } = cli
        .command
        .as_ref()
        .ok_or_else(|| anyhow!("view command missing"))?
    else {
        bail!("view command missing");
    };

    let history = match (explicit_log_path.as_deref(), *last) {
        (Some(path), false) | (Some(path), true) => load_history_from_path(path)?,
        (None, true) => {
            let session = list_sessions(&workspace_root)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no previous sessions found for this workspace"))?;
            load_history_from_path(&session.path)?
        }
        (None, false) => bail!("view requires --log-path <FILE> or --last"),
    };

    let output = match (id, path_to, *all) {
        (Some(id), None, false) => history.theorem_graph.examine_markdown(*id)?,
        (None, Some(id), false) => history.theorem_graph.path_to_markdown(*id)?,
        (None, None, true) => history.theorem_graph.all_markdown(),
        (None, None, false) => bail!("view requires exactly one of --id, --path-to, or --all"),
        _ => bail!("view accepts only one of --id, --path-to, or --all"),
    };

    eprintln!("{}", build_view_save_hint(cli));
    println!("{output}");
    Ok(())
}

fn build_view_save_hint(cli: &Cli) -> String {
    let mut command = String::from("aimv2 view");
    if let Some(log_path) = &cli.log_path {
        command.push_str(&format!(
            " --log-path {}",
            shell_escape(&log_path.display().to_string())
        ));
    } else {
        command.push_str(" --last");
    }

    if let Some(CliCommand::View {
        last,
        id,
        path_to,
        all,
    }) = &cli.command
    {
        if *last && cli.log_path.is_none() {
            command = String::from("aimv2 view --last");
        }
        if let Some(id) = id {
            command.push_str(&format!(" --id {id}"));
            command.push_str(&format!(" > theorem-{id}.md"));
        }
        if let Some(id) = path_to {
            command.push_str(&format!(" --path-to {id}"));
            command.push_str(&format!(" > theorem-path-{id}.md"));
        }
        if *all {
            command.push_str(" --all");
            command.push_str(" > theorem-graph.md");
        }
    }

    format!(
        "note: save this markdown to a local file with:\n  {command}\nthen open the generated `.md` file in your markdown viewer."
    )
}

fn shell_escape(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "/._-".contains(ch))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

fn resolve_log_path(workspace_root: &Path, path: Option<PathBuf>) -> Result<Option<PathBuf>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let resolved = if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    };
    Ok(Some(resolved))
}

fn ensure_log_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("log path has no parent directory: {}", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))
}

fn default_sessions_root() -> Result<PathBuf> {
    let root = env::temp_dir().join("aim-logs");
    fs::create_dir_all(&root).with_context(|| format!("failed to create {}", root.display()))?;
    Ok(root)
}

fn format_last_active(timestamp_ms: u128) -> String {
    let timestamp_ms = match i64::try_from(timestamp_ms) {
        Ok(value) => value,
        Err(_) => return timestamp_ms.to_string(),
    };
    match Local.timestamp_millis_opt(timestamp_ms).single() {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        None => timestamp_ms.to_string(),
    }
}

fn format_last_user_prompt(history: &HistoryFile) -> String {
    history
        .entries
        .iter()
        .rev()
        .find_map(|entry| match entry {
            HistoryEntry::User { content, .. } => Some(preview_text(content, 100)),
            _ => None,
        })
        .unwrap_or_else(|| "(no user prompt yet)".to_string())
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut preview = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        preview.push('…');
    }
    if preview.is_empty() {
        "(empty)".to_string()
    } else {
        preview
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn read_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn print_repl_help(
    auto_approve: bool,
    enable_shell: bool,
    reviewer: &ReviewerConfig,
    history_path: &Path,
) {
    println!("{}", style(COLOR_BOLD, "interactive help"));
    println!();
    println!("{}", style(COLOR_DIM, "slash commands:"));
    println!(
        "  {}  Retry the previous turn without adding a new user message",
        style(COLOR_DIM, "/continue")
    );
    println!(
        "  {}  Manually trigger a pre-turn history compaction",
        style(COLOR_DIM, "/compact")
    );
    println!(
        "  {}  Switch reviewer strategy",
        style(COLOR_DIM, "/reviewer simple|progressive")
    );
    println!(
        "  {}  Set the simple reviewer parallel review count",
        style(COLOR_DIM, "/reviews <N>")
    );
    println!(
        "  {}  Set the progressive reviewer iteration count",
        style(COLOR_DIM, "/iterations <N>")
    );
    println!("  {}  Show this help", style(COLOR_DIM, "/help"));
    if enable_shell {
        println!(
            "  {}  Enable auto approval for shell commands",
            style(COLOR_DIM, "/auto on")
        );
        println!(
            "  {}  Require approval before shell commands",
            style(COLOR_DIM, "/auto off")
        );
    }
    println!("  {}  Exit the current session", style(COLOR_DIM, "/exit"));
    println!("  {}  Exit the current session", style(COLOR_DIM, "/quit"));
    println!();
    println!("{}", style(COLOR_DIM, "current session:"));
    println!(
        "  shell tool: {}",
        if enable_shell {
            style(COLOR_GREEN, "enabled")
        } else {
            style(COLOR_DIM, "disabled")
        }
    );
    if enable_shell {
        println!(
            "  approval mode: {}",
            if auto_approve {
                style(COLOR_YELLOW, "auto")
            } else {
                style(COLOR_CYAN, "ask")
            }
        );
    }
    println!("  reviewer: {}", reviewer.description());
    println!("  simple reviews: {}", reviewer.simple_reviews);
    println!(
        "  progressive iterations: {}",
        reviewer.progressive_iterations
    );
    println!("  history file: {}", history_path.display());
    println!();
    println!("{}", style(COLOR_DIM, "how to use it:"));
    println!("  - Type mathematical requests in plain language.");
    println!("  - Use /continue to retry the previous turn from the current saved history.");
    println!(
        "  - The assistant aims for rigorous reasoning and will say when a proof is incomplete."
    );
    if enable_shell {
        println!("  - Shell commands run inside the current workspace only.");
        println!("  - In ask mode, you can approve or reject each shell command before execution.");
    }
    println!("  - Ctrl-C cancels the current input line; Ctrl-D exits the session.");
}

#[cfg(test)]
mod tests {
    use super::{
        Config, PROGRESSIVE_REVIEW_MIN_CHUNK_LINES, ResumeMode, ReviewerConfig, ReviewerKind,
        SessionConfigSnapshot, build_view_save_hint, load_session_file, resolve_session_settings,
        split_proof_into_chunks,
    };
    use crate::history::HistoryFile;
    use crate::llm::LlmConfig;
    use async_openai::types::chat::ReasoningEffort;
    use clap::Parser;
    use std::{fs, path::PathBuf};

    #[test]
    fn split_proof_respects_chunk_count_and_order() {
        let proof = (1..=10)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");

        let chunks = split_proof_into_chunks(&proof, 4, PROGRESSIVE_REVIEW_MIN_CHUNK_LINES);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "line 1\nline 2\nline 3\nline 4\nline 5");
        assert_eq!(chunks[1], "line 6\nline 7\nline 8\nline 9\nline 10");
    }

    #[test]
    fn resolve_session_settings_prefers_saved_resume_config() {
        let workspace_root = PathBuf::from("/tmp/workspace");
        let fallback = fallback_config(workspace_root.clone());
        let history = HistoryFile {
            version: 7,
            session_id: "session".to_string(),
            workspace_root: workspace_root.display().to_string(),
            last_active_at_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            theorem_graph: Default::default(),
            session_config: Some(SessionConfigSnapshot {
                model: "gpt-5.4-mini".to_string(),
                reasoning_effort: "high".to_string(),
                base_url: "https://example.invalid/v1".to_string(),
                history_token_limit: 2048,
                reviewer_kind: "simple".to_string(),
                simple_reviews: 6,
                progressive_iterations: 9,
                enable_shell: true,
                auto_approve: true,
            }),
            entries: Vec::new(),
        };

        let (config, auto_approve) =
            resolve_session_settings(&history, ResumeMode::Last, fallback, false).unwrap();

        assert_eq!(config.llm.model, "gpt-5.4-mini");
        assert!(matches!(config.llm.reasoning_effort, ReasoningEffort::High));
        assert_eq!(config.base_url, "https://example.invalid/v1");
        assert_eq!(config.history_token_limit, 2048);
        assert_eq!(config.reviewer.kind, ReviewerKind::Simple);
        assert_eq!(config.reviewer.simple_reviews, 6);
        assert_eq!(config.reviewer.progressive_iterations, 9);
        assert!(config.enable_shell);
        assert!(auto_approve);
    }

    #[test]
    fn resolve_session_settings_uses_fallback_for_legacy_resume_logs() {
        let workspace_root = PathBuf::from("/tmp/workspace");
        let fallback = fallback_config(workspace_root.clone());
        let history = HistoryFile {
            version: 6,
            session_id: "session".to_string(),
            workspace_root: workspace_root.display().to_string(),
            last_active_at_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            theorem_graph: Default::default(),
            session_config: None,
            entries: Vec::new(),
        };

        let (config, auto_approve) =
            resolve_session_settings(&history, ResumeMode::Select, fallback.clone(), true).unwrap();

        assert_eq!(config.llm.model, fallback.llm.model);
        assert!(matches!(
            config.llm.reasoning_effort,
            ReasoningEffort::Medium
        ));
        assert_eq!(config.base_url, fallback.base_url);
        assert_eq!(config.history_token_limit, fallback.history_token_limit);
        assert_eq!(config.reviewer.kind, fallback.reviewer.kind);
        assert_eq!(
            config.reviewer.progressive_iterations,
            fallback.reviewer.progressive_iterations
        );
        assert_eq!(config.enable_shell, fallback.enable_shell);
        assert!(auto_approve);
    }

    #[test]
    fn build_view_save_hint_supports_all_output() {
        let cli = super::Cli::parse_from(["aimv2", "view", "--last", "--all"]);

        let hint = build_view_save_hint(&cli);

        assert!(hint.contains("aimv2 view --last --all > theorem-graph.md"));
    }

    #[test]
    fn load_session_file_allows_cross_workspace_resume() {
        let temp_dir = std::env::temp_dir().join(format!(
            "aimv2-test-{}-{}",
            std::process::id(),
            super::now_millis()
        ));
        fs::create_dir_all(&temp_dir).unwrap();
        let log_path = temp_dir.join("foreign-session.json");
        let foreign_workspace = temp_dir.join("other-workspace");
        let history = HistoryFile {
            version: 7,
            session_id: "session".to_string(),
            workspace_root: foreign_workspace.display().to_string(),
            last_active_at_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            theorem_graph: Default::default(),
            session_config: None,
            entries: Vec::new(),
        };
        fs::write(&log_path, serde_json::to_string(&history).unwrap()).unwrap();

        let loaded = load_session_file(&log_path).unwrap();

        assert_eq!(
            loaded.workspace_root,
            foreign_workspace.display().to_string()
        );

        let _ = fs::remove_file(&log_path);
        let _ = fs::remove_dir(&temp_dir);
    }

    fn fallback_config(workspace_root: PathBuf) -> Config {
        Config {
            llm: LlmConfig {
                model: "gpt-5.4".to_string(),
                reasoning_effort: ReasoningEffort::Medium,
            },
            base_url: "https://api.openai.com/v1".to_string(),
            history_token_limit: 1_000_000,
            reviewer: ReviewerConfig {
                kind: ReviewerKind::Progressive,
                simple_reviews: 4,
                progressive_iterations: 3,
            },
            workspace_root,
            enable_shell: false,
        }
    }
}
