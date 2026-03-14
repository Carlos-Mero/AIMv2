use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

pub const COLOR_RESET: &str = "\x1b[0m";
pub const COLOR_BOLD: &str = "\x1b[1m";
pub const COLOR_BLUE: &str = "\x1b[34m";
pub const COLOR_CYAN: &str = "\x1b[36m";
pub const COLOR_GREEN: &str = "\x1b[32m";
pub const COLOR_YELLOW: &str = "\x1b[33m";
pub const COLOR_RED: &str = "\x1b[31m";
pub const COLOR_DIM: &str = "\x1b[2m";

const MAX_PREVIEW_CHARS: usize = 1_500;
const MAX_PREVIEW_LINES: usize = 10;
const MAX_TOOL_CALL_PREVIEW_LINES: usize = 10;

pub fn style(color: &str, text: &str) -> String {
    format!("{color}{text}{COLOR_RESET}")
}

pub fn editor_prompt(label: &str) -> String {
    format!("{label}> ")
}

pub fn role_prefix(label: &str, color: &str) -> String {
    format!("{}  ", style(color, &format!("{label}>")))
}

pub fn print_statusline(
    workspace_root: &Path,
    model: &str,
    active_tokens: u64,
    total_context_tokens: u64,
    cumulative_input_tokens: u64,
    cumulative_output_tokens: u64,
) {
    let workspace = shorten_path(workspace_root, 36);
    let context_percent = if total_context_tokens == 0 {
        0.0
    } else {
        (active_tokens as f64 / total_context_tokens as f64) * 100.0
    };
    let line = format!(
        "{}  {}  {}/{} ({:.1}%)  usage in {} out {}",
        workspace,
        model,
        format_count(active_tokens),
        format_count(total_context_tokens),
        context_percent,
        format_count(cumulative_input_tokens),
        format_count(cumulative_output_tokens),
    );
    println!("{}", style(COLOR_DIM, &line));
}

pub fn print_tool_result(content: &str, success: bool) {
    let label = if success {
        style(COLOR_CYAN, "tool>")
    } else {
        style(COLOR_RED, "tool>")
    };
    println!("{label} {}", fold_for_display(content));
}

pub fn print_api_error(message: &str) {
    println!(
        "{} {}",
        style(COLOR_RED, "api error>"),
        fold_for_display(message)
    );
}

pub fn print_tool_call(command: &str, workdir: &str) {
    let prefix = format!("{} ", style(COLOR_CYAN, "tool[shell]>"));
    let message = format!("{command}\n(cwd: {workdir})");
    println!(
        "{}{}",
        prefix,
        fold_lines_for_display(&message, MAX_TOOL_CALL_PREVIEW_LINES)
    );
}

pub fn prompt_for_approval(auto_approve: &mut bool) -> Result<bool> {
    loop {
        print!(
            "{}{}",
            style(COLOR_YELLOW, "approval> "),
            style(COLOR_DIM, "[y]es / [n]o / [a]uto> ")
        );
        io::stdout().flush().ok();

        let mut answer = String::new();
        io::stdin()
            .read_line(&mut answer)
            .context("failed to read approval input")?;

        match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => return Ok(true),
            "n" | "no" | "" => return Ok(false),
            "a" | "auto" => {
                *auto_approve = true;
                println!(
                    "{} auto approval enabled for the rest of this run",
                    style(COLOR_YELLOW, "approval>")
                );
                return Ok(true);
            }
            _ => println!("{} please answer y, n, or a", style(COLOR_RED, "approval>")),
        }
    }
}

#[allow(dead_code)]
pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

#[allow(dead_code)]
impl Spinner {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut index = 0usize;
            while !signal.load(Ordering::Relaxed) {
                print!(
                    "\r{} {}",
                    style(
                        COLOR_DIM,
                        &format!("working{} ", frames[index % frames.len()])
                    ),
                    style(COLOR_DIM, "waiting for model response")
                );
                io::stdout().flush().ok();
                thread::sleep(Duration::from_millis(120));
                index += 1;
            }
            print!("\r{}\r", " ".repeat(64));
            io::stdout().flush().ok();
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn fold_for_display(text: &str) -> String {
    fold_lines_for_display(text, MAX_PREVIEW_LINES)
}

fn fold_lines_for_display(text: &str, max_lines: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let clipped_lines = lines
        .iter()
        .take(max_lines)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    let mut preview = if clipped_lines.chars().count() > MAX_PREVIEW_CHARS {
        clipped_lines
            .chars()
            .take(MAX_PREVIEW_CHARS)
            .collect::<String>()
    } else {
        clipped_lines
    };

    if lines.len() > max_lines || text.chars().count() > preview.chars().count() {
        if !preview.ends_with('\n') {
            preview.push('\n');
        }
        preview.push_str(&style(
            COLOR_DIM,
            "[... folded in terminal view; full output kept in session history ...]",
        ));
    }

    preview
}

fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push('_');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn shorten_path(path: &Path, max_chars: usize) -> String {
    let text = path.display().to_string();
    let count = text.chars().count();
    if count <= max_chars {
        return text;
    }
    if max_chars <= 1 {
        return "…".to_string();
    }
    let tail_len = max_chars.saturating_sub(1);
    let tail = text
        .chars()
        .rev()
        .take(tail_len)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("…{}", tail)
}
