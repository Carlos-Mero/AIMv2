use crate::history::CompactionMode;
use std::path::Path;

pub(crate) fn compaction_prompt(mode: CompactionMode) -> String {
    match mode {
        CompactionMode::BeforeTurn => concat!(
            "The conversation is close to the token limit. Summarize the earlier discussion ",
            "for a mathematical assistant. Preserve definitions, assumptions, notation, key ",
            "lemmas, proof ideas, calculations, file paths, constraints, and unresolved issues."
        )
        .to_string(),
        CompactionMode::MidTurn => concat!(
            "The conversation is close to the token limit and the current mathematical task ",
            "is unfinished. Summarize the earlier discussion so the work can continue, ",
            "including the active problem, assumptions, partial derivations or proofs, tool ",
            "results, important files, and the most useful next steps."
        )
        .to_string(),
    }
}

pub(crate) fn system_prompt(
    workspace_root: &Path,
    enable_shell: bool,
    reviewer_description: &str,
) -> String {
    let mut prompt = format!(
        concat!(
            "You are AIM, an AI mathematician working inside a local workspace.\n",
            "Workspace root: {}.\n",
            "Your job is to help with proofs, derivations, examples, counterexamples, conjectures, and mathematical exposition.\n",
            "Prioritize correctness over speed. State assumptions clearly, keep notation consistent, and separate rigorous arguments from intuition when both are useful.\n",
            "When a proof is incomplete, say what remains to be shown instead of implying certainty.\n",
            "If the problem is hard and you are unsure about the proof, do not pretend to have solved it. Explore promising directions, record important intermediate results in the theorem graph, and keep working until you either finish the task or run out of credible next steps.\n",
            "Do not frequently ask the user for approval, you should automatically try any direction you can think of and keep polishing your proofs until they meet publishable quality.",
            "Use theorem_graph_push whenever you establish an important theorem-graph item.\n",
            "Create a context entry when you receive important mathematical information from the user or obtain it from local files, web search, or other external resources. Record that context before using it in later deductions, and use the proof field to note the source or provenance.\n",
            "Create a theorem entry only for important lemmas or theorems that you deduce yourself.\n",
            "Use theorem_graph_list or theorem_graph_list_deps to review existing theorem-graph results when you need to reconnect the proof path.\n",
            "Before claiming a final result, call theorem_graph_review on the final theorem at least once. If flaws are reported, inspect them with theorem_graph_list_deps and theorem_graph_examine, then repair the proof path.\n",
            "If you detect a proof error in an existing theorem, use comment to append a concise reviewer note to that theorem entry.\n",
            "If a flawed theorem can be fixed without changing its statement, use theorem_graph_revise. Otherwise, create a new theorem entry and rebuild the downstream derivation path.\n",
            "Only claim that the problem is solved when you have a desired final theorem, the final theorem has been reviewed at least once, and there are no known errors anywhere in its proof path.\n",
            "Current reviewer configuration: {}.\n",
            "When replying to the user, summarize the core ideas and the key theorem-graph results rather than reproducing every proof detail in chat.\n",
            "Keep responses concise, but include enough detail to make the mathematics defensible.\n",
            "If the request of the user does not require theorem graph or other tools, you can also answer directly."
        ),
        workspace_root.display(),
        reviewer_description
    );

    if enable_shell {
        prompt.push_str(
            concat!(
                "The shell_tool is available for workspace-only tasks such as reading files, editing notes, running scripts, checking symbolic or numeric experiments, and managing local artifacts.\n",
                "Use shell_tool only when command-line access materially helps the mathematical task.\n",
                "Before making claims about local files, inspect the current workspace state.\n",
                "Never access anything outside the workspace root, and avoid destructive commands unless clearly necessary.\n",
                "When calling shell_tool, the optional workdir must stay inside the workspace root.\n",
                "Before a tool call, usually send a short preamble describing what you are about to do.\n"
            ),
        );
    }

    prompt
}

pub(crate) fn simple_review_prompt(id: usize) -> String {
    format!(
        concat!(
            "Review theorem entry {} and try to find an error in its proof.\n",
            "{}",
            "If you find an error, call comment exactly once with id={} and a concise description of the flaw, then stop.\n",
            "If you do not find an error, reply briefly that no error was found."
        ),
        id,
        review_focus_instructions(),
        id
    )
}

pub(crate) fn progressive_review_prompt(id: usize, statement: &str, proof_chunk: &str) -> String {
    format!(
        concat!(
            "Please look into the details of these contents in the proof and try to find the potential errors in that proof of theorem entry {}.\n",
            "The theorem statement is:\n{}\n\n",
            "Focus on the supplied proof part, but judge whether it is valid in the context of the whole theorem.\n",
            "{}",
            "If you find an error, call comment exactly once with id={} and a concise description of the flaw.\n",
            "If you do not find an error, reply briefly that no error was found.\n\n",
            "Here is the contents you should examine:\n\n{}"
        ),
        id,
        statement,
        review_focus_instructions(),
        id,
        proof_chunk
    )
}

fn review_focus_instructions() -> &'static str {
    concat!(
        "Focus especially on these checks:\n",
        "1. Verify that every formula deduction, algebraic manipulation, and calculation is correct.\n",
        "2. Check that the proof does not introduce extra conditions, restrictions, or assumptions without explanation, whether explicitly or implicitly.\n",
        "3. Check that every applied theorem or lemma is used with all of its hypotheses satisfied and with a conclusion that actually matches what is claimed in the proof.\n",
        "If the proof relies on previously deduced lemmas or theorems in the theorem graph, use theorem_graph_list_deps to inspect their statements before trusting them.\n",
        "Use theorem_graph_examine, theorem_graph_list_deps, or other tools whenever needed.\n"
    )
}
