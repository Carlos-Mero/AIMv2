use anyhow::{Context, Result};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SKILLS_INSTRUCTIONS_OPEN_TAG: &str = "<skills_instructions>";
const SKILLS_INSTRUCTIONS_CLOSE_TAG: &str = "</skills_instructions>";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SkillMetadata {
    pub(crate) dir_path: PathBuf,
    pub(crate) name: String,
    pub(crate) description: String,
}

impl SkillMetadata {
    pub(crate) fn skill_md_path(&self) -> PathBuf {
        self.dir_path.join("SKILL.md")
    }
}

pub(crate) fn discover_skills(
    workspace_root: &Path,
    external_skill_roots: &[PathBuf],
) -> Result<Vec<SkillMetadata>> {
    let mut roots = Vec::new();
    roots.push(workspace_root.join(".aim/skills"));
    roots.push(workspace_root.join(".agents/skills"));
    if let Some(home_dir) = home_dir() {
        roots.push(home_dir.join(".aim/skills"));
        roots.push(home_dir.join(".agents/skills"));
    }
    roots.extend(external_skill_roots.iter().cloned());

    let mut seen_dirs = HashSet::new();
    let mut skills = Vec::new();

    for root in roots {
        for skill_dir in list_skill_dirs(&root)? {
            let canonical_dir = skill_dir
                .canonicalize()
                .with_context(|| format!("failed to canonicalize {}", skill_dir.display()))?;
            if !seen_dirs.insert(canonical_dir.clone()) {
                continue;
            }
            let (name, description) = parse_skill_metadata(&canonical_dir.join("SKILL.md"))?;
            skills.push(SkillMetadata {
                dir_path: canonical_dir,
                name,
                description,
            });
        }
    }

    skills.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(skills)
}

pub(crate) fn render_skills_section(skills: &[SkillMetadata]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    lines.push("## Skills".to_string());
    lines.push("A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.".to_string());
    lines.push("### Available skills".to_string());

    for skill in skills {
        let path_str = skill.skill_md_path().to_string_lossy().replace('\\', "/");
        lines.push(format!(
            "- {}: {} (file: {})",
            skill.name, skill.description, path_str
        ));
    }

    lines.push("### How to use skills".to_string());
    lines.push(
        r###"- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, use your shell_tool to read its `SKILL.md`. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."###
            .to_string(),
    );

    let body = lines.join("\n");
    Some(format!(
        "{SKILLS_INSTRUCTIONS_OPEN_TAG}\n{body}\n{SKILLS_INSTRUCTIONS_CLOSE_TAG}"
    ))
}

fn list_skill_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let entries =
        fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?;
    let mut dirs = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() && path.join("SKILL.md").is_file() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn parse_skill_metadata(path: &Path) -> Result<(String, String)> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read skill file {}", path.display()))?;
    let dir_name = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("unnamed-skill")
        .to_string();

    let lines: Vec<&str> = text.lines().collect();
    let mut name = None;
    let mut description_lines = Vec::new();
    let mut saw_heading = false;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if saw_heading && !description_lines.is_empty() {
                break;
            }
            continue;
        }

        if !saw_heading && trimmed.starts_with('#') {
            let heading = trimmed.trim_start_matches('#').trim();
            if !heading.is_empty() {
                name = Some(heading.to_string());
                saw_heading = true;
                continue;
            }
        }

        if saw_heading {
            if trimmed.starts_with('#') {
                break;
            }
            description_lines.push(trimmed);
        }
    }

    let name = name.unwrap_or(dir_name);
    let description = if description_lines.is_empty() {
        "No description provided.".to_string()
    } else {
        description_lines.join(" ")
    };

    Ok((name, description))
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::{discover_skills, render_skills_section};
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn discover_skills_loads_heading_and_description() {
        let temp_dir = std::env::temp_dir().join(format!(
            "aimv2-skills-test-{}-{}",
            std::process::id(),
            crate::now_millis()
        ));
        let workspace_root = temp_dir.join("workspace");
        let skill_dir = workspace_root.join(".aim/skills/example");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "# Example Skill\n\nShort description.\nSecond sentence.\n",
        )
        .unwrap();

        let skills = discover_skills(&workspace_root, &[]).unwrap();

        let skill = skills
            .iter()
            .find(|skill| skill.name == "Example Skill")
            .expect("expected test skill to be discovered");
        assert_eq!(skill.description, "Short description. Second sentence.");
        assert!(skill.dir_path.is_absolute());

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn discover_skills_includes_external_roots_and_deduplicates() {
        let temp_dir = std::env::temp_dir().join(format!(
            "aimv2-skills-test-{}-{}",
            std::process::id(),
            crate::now_millis()
        ));
        let workspace_root = temp_dir.join("workspace");
        let external_root = temp_dir.join("external");
        let skill_dir = external_root.join("shared-skill");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), "# Shared\n\nReusable skill.\n").unwrap();

        let duplicate_root = external_root.join("..").join("external");
        let skills =
            discover_skills(&workspace_root, &[external_root.clone(), duplicate_root]).unwrap();

        let shared_count = skills.iter().filter(|skill| skill.name == "Shared").count();
        assert_eq!(shared_count, 1);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn render_skills_section_matches_expected_structure() {
        let rendered = render_skills_section(&[super::SkillMetadata {
            dir_path: PathBuf::from("/tmp/example-skill"),
            name: "Example".to_string(),
            description: "Example description.".to_string(),
        }])
        .unwrap();

        assert!(rendered.starts_with("<skills_instructions>\n## Skills\n"));
        assert!(
            rendered
                .contains("- Example: Example description. (file: /tmp/example-skill/SKILL.md)")
        );
        assert!(rendered.ends_with("</skills_instructions>"));
    }
}
