use crate::{prompt, theorem_graph::TheoremGraph};
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, FunctionCall,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

const DEFAULT_TOKEN_LIMIT: u64 = 128 * 1024;
const GPT5_TOKEN_LIMIT: u64 = 1_000_000;
const GEMINI3_TOKEN_LIMIT: u64 = 1_000_000;
const COMPACTION_TRIGGER_NUMERATOR: u64 = 4;
const COMPACTION_TRIGGER_DENOMINATOR: u64 = 5;
const HISTORY_SUMMARY_PREFIX: &str = "[history summary]";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HistoryFile {
    pub(crate) version: u32,
    pub(crate) session_id: String,
    pub(crate) workspace_root: String,
    pub(crate) last_active_at_ms: u128,
    #[serde(default)]
    pub(crate) total_input_tokens: u64,
    #[serde(default)]
    pub(crate) total_output_tokens: u64,
    #[serde(default)]
    pub(crate) total_tokens: u64,
    #[serde(default)]
    pub(crate) theorem_graph: TheoremGraph,
    #[serde(default)]
    pub(crate) session_config: Option<SessionConfigSnapshot>,
    pub(crate) entries: Vec<HistoryEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SessionConfigSnapshot {
    pub(crate) model: String,
    pub(crate) reasoning_effort: String,
    pub(crate) base_url: String,
    pub(crate) history_token_limit: u64,
    pub(crate) reviewer_kind: String,
    pub(crate) simple_reviews: u32,
    pub(crate) progressive_iterations: u32,
    pub(crate) enable_shell: bool,
    pub(crate) auto_approve: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum HistoryEntry {
    System {
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
    User {
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
    Assistant {
        content: String,
        #[serde(default)]
        reasoning: Option<String>,
        #[serde(default)]
        tool_calls: Vec<AssistantToolCall>,
        #[serde(default)]
        estimated_tokens: u64,
    },
    Tool {
        #[serde(default)]
        tool_call_id: String,
        #[serde(default)]
        tool_name: String,
        content: String,
        #[serde(default)]
        estimated_tokens: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssistantToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactionMode {
    BeforeTurn,
    MidTurn,
}

impl HistoryFile {
    pub(crate) fn push_user(&mut self, content: String) {
        let mut entry = HistoryEntry::User {
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_assistant(
        &mut self,
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<AssistantToolCall>,
    ) {
        let mut entry = HistoryEntry::Assistant {
            content,
            reasoning,
            tool_calls,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_tool(&mut self, tool_call_id: String, tool_name: String, content: String) {
        let mut entry = HistoryEntry::Tool {
            tool_call_id,
            tool_name,
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn push_system(&mut self, content: String) {
        let mut entry = HistoryEntry::System {
            content,
            estimated_tokens: 0,
        };
        entry.set_estimated_tokens(entry.weight().saturating_mul(4));
        self.entries.push(entry);
    }

    pub(crate) fn note_api_usage(
        &mut self,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        total_tokens: Option<u64>,
    ) {
        let start = self.last_system_index().unwrap_or(0);
        let entry_count = self.entries[start..].len() as u64;
        let entry_estimate = estimate_entry_tokens(&self.entries, start);
        let (resolved_input, resolved_output, resolved_total) =
            resolve_usage(input_tokens, output_tokens, total_tokens);

        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(resolved_input.unwrap_or(0));
        self.total_output_tokens = self
            .total_output_tokens
            .saturating_add(resolved_output.unwrap_or(0));
        self.total_tokens = self
            .total_tokens
            .saturating_add(resolved_total.unwrap_or_else(|| {
                resolved_input
                    .unwrap_or(0)
                    .saturating_add(resolved_output.unwrap_or(0))
            }));

        let active_estimate = resolved_input
            .or(resolved_total)
            .unwrap_or_else(|| entry_estimate.max(entry_count));
        apply_estimated_tokens(&mut self.entries, start, active_estimate);
    }

    pub(crate) fn active_token_usage(&self) -> u64 {
        let start = self.last_system_index().unwrap_or(0);
        self.entries[start..]
            .iter()
            .map(HistoryEntry::estimated_tokens)
            .sum()
    }

    pub(crate) fn total_input_usage(&self) -> u64 {
        self.total_input_tokens
    }

    pub(crate) fn total_output_usage(&self) -> u64 {
        self.total_output_tokens
    }

    pub(crate) fn apply_usage_delta(&mut self, delta: (u64, u64, u64)) {
        let (input_delta, output_delta, total_delta) = delta;
        self.total_input_tokens = self.total_input_tokens.saturating_add(input_delta);
        self.total_output_tokens = self.total_output_tokens.saturating_add(output_delta);
        self.total_tokens = self
            .total_tokens
            .saturating_add(total_delta.max(input_delta.saturating_add(output_delta)));
    }

    pub(crate) fn needs_compaction(&self, token_limit: u64) -> bool {
        self.active_token_usage()
            >= token_limit.saturating_mul(COMPACTION_TRIGGER_NUMERATOR)
                / COMPACTION_TRIGGER_DENOMINATOR
    }

    pub(crate) fn last_user_content(&self) -> Option<String> {
        self.entries.iter().rev().find_map(|entry| match entry {
            HistoryEntry::User { content, .. } => Some(content.clone()),
            _ => None,
        })
    }

    pub(crate) fn apply_compaction(&mut self, summary: String, resume_user: Option<String>) {
        self.push_system(format!("{HISTORY_SUMMARY_PREFIX}\n{}", summary.trim()));
        if let Some(user) = resume_user {
            self.push_user(user);
        }
        let start = self.last_system_index().unwrap_or(0);
        let estimated = estimate_entry_tokens(&self.entries, start);
        apply_estimated_tokens(&mut self.entries, start, estimated.max(1));
    }

    fn last_system_index(&self) -> Option<usize> {
        self.entries
            .iter()
            .rposition(|entry| matches!(entry, HistoryEntry::System { .. }))
    }

    pub(crate) fn clone_at_model_boundary(&self) -> Self {
        let mut cloned = self.clone();
        let Some(last_assistant_index) = cloned.entries.iter().rposition(|entry| {
            matches!(
                entry,
                HistoryEntry::Assistant { tool_calls, .. } if !tool_calls.is_empty()
            )
        }) else {
            return cloned;
        };

        let trailing_entries = &cloned.entries[last_assistant_index + 1..];
        if trailing_entries
            .iter()
            .all(|entry| matches!(entry, HistoryEntry::Tool { .. }))
        {
            cloned.entries.truncate(last_assistant_index);
        }

        cloned
    }
}

impl HistoryEntry {
    pub(crate) fn estimated_tokens(&self) -> u64 {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens,
        }
    }

    fn weight(&self) -> u64 {
        match self {
            Self::System { content, .. }
            | Self::User { content, .. }
            | Self::Assistant { content, .. } => {
                let tool_call_weight = match self {
                    Self::Assistant { tool_calls, .. } => tool_calls
                        .iter()
                        .map(|call| text_weight(&call.name) + text_weight(&call.arguments))
                        .sum(),
                    _ => 0,
                };
                text_weight(content) + tool_call_weight
            }
            Self::Tool { content, .. } => text_weight(content),
        }
    }

    fn set_estimated_tokens(&mut self, value: u64) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = value,
        }
    }

    fn reset_estimated_tokens(&mut self) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = 0,
        }
    }

    fn add_estimated_tokens(&mut self, delta: u64) {
        match self {
            Self::System {
                estimated_tokens, ..
            }
            | Self::User {
                estimated_tokens, ..
            }
            | Self::Assistant {
                estimated_tokens, ..
            }
            | Self::Tool {
                estimated_tokens, ..
            } => *estimated_tokens = estimated_tokens.saturating_add(delta),
        }
    }
}

pub(crate) fn build_messages(
    workspace_root: &Path,
    entries: &[HistoryEntry],
    enable_shell: bool,
    reviewer_description: &str,
) -> Vec<ChatCompletionRequestMessage> {
    let mut messages = vec![
        ChatCompletionRequestSystemMessage {
            content: ChatCompletionRequestSystemMessageContent::Text(prompt::system_prompt(
                workspace_root,
                enable_shell,
                reviewer_description,
            )),
            name: None,
        }
        .into(),
    ];

    let active_entries = match entries
        .iter()
        .rposition(|entry| matches!(entry, HistoryEntry::System { .. }))
    {
        Some(index) => &entries[index..],
        None => entries,
    };

    for entry in active_entries {
        match entry {
            HistoryEntry::System { content, .. } => {
                messages.push(
                    ChatCompletionRequestSystemMessage {
                        content: ChatCompletionRequestSystemMessageContent::Text(content.clone()),
                        name: None,
                    }
                    .into(),
                );
            }
            HistoryEntry::User { content, .. } => {
                messages.push(
                    ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text(content.clone()),
                        name: None,
                    }
                    .into(),
                );
            }
            HistoryEntry::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let assistant_content = if tool_calls.is_empty() || !content.trim().is_empty() {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(
                        content.clone(),
                    ))
                } else {
                    None
                };
                let tool_calls = if tool_calls.is_empty() {
                    None
                } else {
                    Some(
                        tool_calls
                            .iter()
                            .map(|call| {
                                ChatCompletionMessageToolCalls::Function(
                                    ChatCompletionMessageToolCall {
                                        id: call.id.clone(),
                                        function: FunctionCall {
                                            name: call.name.clone(),
                                            arguments: call.arguments.clone(),
                                        },
                                    },
                                )
                            })
                            .collect(),
                    )
                };
                #[allow(deprecated)]
                let assistant_message = ChatCompletionRequestAssistantMessage {
                    content: assistant_content,
                    refusal: None,
                    name: None,
                    audio: None,
                    tool_calls,
                    function_call: None,
                };
                messages.push(assistant_message.into());
            }
            HistoryEntry::Tool {
                tool_call_id,
                content,
                ..
            } => {
                messages.push(
                    ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(content.clone()),
                        tool_call_id: tool_call_id.clone(),
                    }
                    .into(),
                );
            }
        }
    }

    messages
}

pub(crate) fn token_limit_for_model(model: &str) -> u64 {
    let normalized = model.trim().to_ascii_lowercase();
    if normalized.starts_with("gpt-5") {
        GPT5_TOKEN_LIMIT
    } else if normalized.starts_with("gemini-3") {
        GEMINI3_TOKEN_LIMIT
    } else {
        DEFAULT_TOKEN_LIMIT
    }
}

fn apply_estimated_tokens(entries: &mut [HistoryEntry], start: usize, target_total: u64) {
    if start >= entries.len() {
        return;
    }

    for entry in entries.iter_mut() {
        entry.reset_estimated_tokens();
    }

    distribute_tokens(&mut entries[start..], target_total);
}

fn estimate_entry_tokens(entries: &[HistoryEntry], start: usize) -> u64 {
    if start >= entries.len() {
        return 0;
    }

    let weight = entries[start..]
        .iter()
        .map(HistoryEntry::weight)
        .sum::<u64>();
    weight.saturating_mul(4)
}

fn distribute_tokens(entries: &mut [HistoryEntry], delta: u64) {
    if entries.is_empty() || delta == 0 {
        return;
    }

    let total_weight = entries.iter().map(HistoryEntry::weight).sum::<u64>();
    if total_weight == 0 {
        let base = delta / entries.len() as u64;
        let mut remainder = delta % entries.len() as u64;
        for entry in entries {
            let extra = u64::from(remainder > 0);
            entry.add_estimated_tokens(base + extra);
            remainder = remainder.saturating_sub(1);
        }
        return;
    }

    let mut assigned = 0_u64;
    let last_index = entries.len() - 1;
    for (index, entry) in entries.iter_mut().enumerate() {
        let share = if index == last_index {
            delta.saturating_sub(assigned)
        } else {
            delta.saturating_mul(entry.weight()) / total_weight
        };
        assigned = assigned.saturating_add(share);
        entry.add_estimated_tokens(share);
    }
}

fn text_weight(text: &str) -> u64 {
    text.split_whitespace().count().max(1) as u64
}

fn resolve_usage(
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
) -> (Option<u64>, Option<u64>, Option<u64>) {
    let resolved_input = input_tokens.or_else(|| {
        total_tokens.and_then(|total| output_tokens.map(|output| total.saturating_sub(output)))
    });
    let resolved_output = output_tokens.or_else(|| {
        total_tokens.and_then(|total| resolved_input.map(|input| total.saturating_sub(input)))
    });
    let resolved_total = total_tokens.or_else(|| {
        Some(
            resolved_input
                .unwrap_or(0)
                .saturating_add(resolved_output.unwrap_or(0)),
        )
    });
    (resolved_input, resolved_output, resolved_total)
}

#[cfg(test)]
mod tests {
    use super::{HistoryEntry, HistoryFile};

    fn history_with_user() -> HistoryFile {
        let mut history = HistoryFile {
            version: 1,
            session_id: "session".to_string(),
            workspace_root: ".".to_string(),
            last_active_at_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_tokens: 0,
            theorem_graph: Default::default(),
            session_config: None,
            entries: Vec::new(),
        };
        history.push_user("test prompt".to_string());
        history
    }

    #[test]
    fn note_api_usage_preserves_reported_total_without_split() {
        let mut history = history_with_user();

        history.note_api_usage(None, None, Some(42));

        assert_eq!(history.total_input_tokens, 0);
        assert_eq!(history.total_output_tokens, 0);
        assert_eq!(history.total_tokens, 42);
        assert_eq!(history.active_token_usage(), 42);
    }

    #[test]
    fn note_api_usage_derives_missing_output_from_total() {
        let mut history = history_with_user();

        history.note_api_usage(Some(30), None, Some(45));

        assert_eq!(history.total_input_tokens, 30);
        assert_eq!(history.total_output_tokens, 15);
        assert_eq!(history.total_tokens, 45);
    }

    #[test]
    fn apply_usage_delta_keeps_explicit_total_delta() {
        let mut history = history_with_user();
        history.total_tokens = 10;

        history.apply_usage_delta((0, 0, 7));

        assert_eq!(history.total_tokens, 17);
    }

    #[test]
    fn apply_usage_delta_uses_visible_split_when_total_delta_is_missing() {
        let mut history = history_with_user();

        history.apply_usage_delta((11, 13, 0));

        assert_eq!(history.total_input_tokens, 11);
        assert_eq!(history.total_output_tokens, 13);
        assert_eq!(history.total_tokens, 24);
    }

    #[test]
    fn note_api_usage_reweights_active_entries() {
        let mut history = history_with_user();
        history.push_assistant("reply".to_string(), None, Vec::new());

        history.note_api_usage(Some(60), Some(20), Some(80));

        let active_total = history
            .entries
            .iter()
            .map(HistoryEntry::estimated_tokens)
            .sum::<u64>();
        assert_eq!(active_total, 60);
    }
}
