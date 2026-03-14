use crate::ui::{COLOR_YELLOW, print_api_error, style};
use anyhow::{Context, Result, bail};
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls,
        ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionTools, CompletionUsage,
        CreateChatCompletionRequestArgs, FunctionObject, FunctionType, ReasoningEffort,
    },
};
use futures::StreamExt;
use serde_json::json;

#[derive(Clone, Debug)]
pub(crate) struct LlmConfig {
    pub(crate) model: String,
    pub(crate) reasoning_effort: ReasoningEffort,
}

pub(crate) type LlmClient = Client<OpenAIConfig>;

#[derive(Debug)]
pub(crate) struct LlmReply {
    pub(crate) content: String,
    pub(crate) reasoning: Option<String>,
    pub(crate) tool_calls: Vec<ToolCall>,
    pub(crate) input_tokens: Option<u64>,
    pub(crate) output_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

pub(crate) fn build_client(api_key: &str, base_url: &str) -> LlmClient {
    let config = OpenAIConfig::new()
        .with_api_key(api_key)
        .with_api_base(base_url);
    Client::with_config(config)
}

pub(crate) async fn call_model<F>(
    client: &LlmClient,
    config: &LlmConfig,
    messages: Vec<async_openai::types::chat::ChatCompletionRequestMessage>,
    enable_tools: bool,
    enable_shell: bool,
    mut on_text: F,
) -> Result<LlmReply>
where
    F: FnMut(&str),
{
    let mut request = CreateChatCompletionRequestArgs::default();
    request.model(&config.model);
    request.messages(messages);
    request.reasoning_effort(config.reasoning_effort.clone());
    request.stream(true);
    request.stream_options(ChatCompletionStreamOptions {
        include_usage: Some(true),
        include_obfuscation: None,
    });

    if enable_tools {
        request.tools(tool_definitions(enable_shell));
        request.parallel_tool_calls(false);
    }

    let request = request
        .build()
        .context("failed to build chat completion request")?;

    let mut stream = client
        .chat()
        .create_stream(request)
        .await
        .context("chat completion request failed")?;

    let mut content = String::new();
    let mut refusal = String::new();
    let reasoning: Option<String> = None;
    let mut usage: Option<CompletionUsage> = None;
    let mut tool_calls: Vec<PartialToolCall> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("failed to receive streaming response chunk")?;
        if let Some(chunk_usage) = chunk.usage {
            usage = Some(chunk_usage);
        }

        for choice in chunk.choices {
            if let Some(text) = choice.delta.content {
                on_text(&text);
                content.push_str(&text);
            }
            if let Some(text) = choice.delta.refusal {
                on_text(&text);
                refusal.push_str(&text);
            }
            if let Some(chunks) = choice.delta.tool_calls {
                merge_tool_call_chunks(&mut tool_calls, chunks)?;
            }
        }
    }

    let content = if content.trim().is_empty() {
        refusal.trim().to_string()
    } else {
        content.trim().to_string()
    };
    let tool_calls = finalize_tool_calls(tool_calls)?;
    if content.is_empty() && tool_calls.is_empty() {
        bail!("model returned neither content nor tool calls");
    }

    Ok(LlmReply {
        content,
        reasoning,
        tool_calls,
        input_tokens: usage.as_ref().map(|usage| u64::from(usage.prompt_tokens)),
        output_tokens: usage
            .as_ref()
            .map(|usage| u64::from(usage.completion_tokens)),
        total_tokens: usage.as_ref().map(|usage| u64::from(usage.total_tokens)),
    })
}

fn tool_definitions(enable_shell: bool) -> Vec<ChatCompletionTools> {
    let mut tools = vec![
        theorem_graph_push_tool_definition(),
        theorem_graph_list_tool_definition(),
        theorem_graph_list_deps_tool_definition(),
        theorem_graph_examine_tool_definition(),
        theorem_graph_review_tool_definition(),
        theorem_graph_revise_tool_definition(),
    ];
    if enable_shell {
        tools.push(shell_tool_definition());
    }
    tools
}

fn shell_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "shell_tool".to_string(),
            description: Some(
                "Run a shell command inside the current workspace for inspection, editing, \
                 symbolic checks, numeric experiments, builds, and tests."
                    .to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The raw shell command to execute."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Optional relative working directory inside the workspace root."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_push_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_push".to_string(),
            description: Some(
                "Add a new theorem-graph entry. Use type=context for important facts supplied by the user or obtained from files, web search, or other external resources; in that case the proof should record the source or provenance. Use type=theorem only for important lemmas or theorems you have deduced yourself.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["context", "theorem"],
                        "description": "Whether this entry is prior context or a theorem established during the current exploration."
                    },
                    "statement": {
                        "type": "string",
                        "description": "The exact mathematical statement."
                    },
                    "proof": {
                        "type": "string",
                        "description": "A rigorous proof, or a reference note when the entry is context."
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "Direct dependency ids in the theorem graph."
                    }
                },
                "required": ["type", "statement", "proof", "dependencies"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_list_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_list".to_string(),
            description: Some(
                "List theorem-graph entries in an id range, including statements, dependencies, and reviewer comments when present.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0}
                },
                "required": ["start", "end"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_list_deps_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_list_deps".to_string(),
            description: Some(
                "Show a theorem entry together with its direct dependencies, including dependency statements, dependency links, review counts, and comments.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0}
                },
                "required": ["id"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_examine_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_examine".to_string(),
            description: Some(
                "Inspect one theorem-graph entry in full detail, including proof text, dependencies, derivations, reviews, and comments.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0}
                },
                "required": ["id"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_review_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_review".to_string(),
            description: Some(
                "Run the theorem-graph review routine on a theorem and its dependencies, updating review counts and flaw markers.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0}
                },
                "required": ["id"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn theorem_graph_revise_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: "theorem_graph_revise".to_string(),
            description: Some(
                "Revise the proof and direct dependencies of an existing theorem-graph entry after a flaw has been identified.".to_string(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "minimum": 0},
                    "proof": {
                        "type": "string",
                        "description": "The corrected proof text."
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "The corrected direct dependency ids."
                    }
                },
                "required": ["id", "proof", "dependencies"],
                "additionalProperties": false
            })),
            strict: Some(true),
        },
    })
}

fn merge_tool_call_chunks(
    tool_calls: &mut Vec<PartialToolCall>,
    chunks: Vec<ChatCompletionMessageToolCallChunk>,
) -> Result<()> {
    for chunk in chunks {
        let index = usize::try_from(chunk.index)
            .with_context(|| format!("invalid tool call index: {}", chunk.index))?;
        while tool_calls.len() <= index {
            tool_calls.push(PartialToolCall::default());
        }
        let entry = &mut tool_calls[index];
        if let Some(id) = chunk.id {
            entry.id = id;
        }
        if let Some(kind) = chunk.r#type {
            match kind {
                FunctionType::Function => {}
            }
        }
        if let Some(function) = chunk.function {
            if let Some(name) = function.name {
                entry.name = name;
            }
            if let Some(arguments) = function.arguments {
                entry.arguments.push_str(&arguments);
            }
        }
    }
    Ok(())
}

fn finalize_tool_calls(tool_calls: Vec<PartialToolCall>) -> Result<Vec<ToolCall>> {
    let mut finalized = Vec::with_capacity(tool_calls.len());
    for (index, call) in tool_calls.into_iter().enumerate() {
        if call.id.is_empty() && call.name.is_empty() && call.arguments.is_empty() {
            continue;
        }
        if call.id.is_empty() {
            bail!("tool call {index} missing id in streaming response");
        }
        if call.name.is_empty() {
            bail!(
                "tool call {} missing function name in streaming response",
                call.id
            );
        }
        finalized.push(ToolCall {
            id: call.id,
            name: call.name,
            arguments: call.arguments,
        });
    }
    Ok(finalized)
}

#[allow(dead_code)]
fn extract_tool_calls(
    tool_calls: Option<Vec<ChatCompletionMessageToolCalls>>,
) -> Result<Vec<ToolCall>> {
    let mut parsed = Vec::new();
    for tool_call in tool_calls.unwrap_or_default() {
        match tool_call {
            ChatCompletionMessageToolCalls::Function(call) => parsed.push(ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
            }),
            ChatCompletionMessageToolCalls::Custom(call) => {
                bail!("unsupported custom tool call: {}", call.custom_tool.name);
            }
        }
    }
    Ok(parsed)
}

pub(crate) fn report_api_error(err: &anyhow::Error) {
    print_api_error(&format!("{err:#}"));
    println!(
        "{} request failed; inspect configuration, model name, and API compatibility",
        style(COLOR_YELLOW, "warning>")
    );
}
