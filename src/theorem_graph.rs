#![allow(dead_code)]

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TheoremEntryType {
    Context,
    Theorem,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct TheoremEntry {
    pub(crate) id: usize,
    #[serde(rename = "type")]
    pub(crate) entry_type: TheoremEntryType,
    pub(crate) statement: String,
    pub(crate) proof: String,
    pub(crate) dependencies: Vec<usize>,
    pub(crate) derivations: Vec<usize>,
    pub(crate) reviews: u32,
    pub(crate) comments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct TheoremGraph {
    #[serde(default)]
    pub(crate) entries: Vec<TheoremEntry>,
}

impl TheoremGraph {
    pub(crate) fn push(
        &mut self,
        entry_type: TheoremEntryType,
        statement: impl Into<String>,
        proof: impl Into<String>,
        dependencies: Vec<usize>,
    ) -> Result<String> {
        let statement = statement.into().trim().to_string();
        let proof = proof.into().trim().to_string();
        self.validate_new_dependencies(None, &dependencies)?;
        if statement.is_empty() {
            bail!("statement cannot be empty");
        }
        if proof.is_empty() {
            bail!("proof cannot be empty");
        }

        let id = self.entries.len();
        let dependencies = normalize_ids(dependencies);
        let entry = TheoremEntry {
            id,
            entry_type,
            statement,
            proof,
            dependencies: dependencies.clone(),
            derivations: Vec::new(),
            reviews: 0,
            comments: None,
        };
        self.entries.push(entry);
        for dependency in dependencies {
            insert_sorted_unique(&mut self.entries[dependency].derivations, id);
        }
        Ok(format!("created theorem entry {id}"))
    }

    pub(crate) fn list(&self, start: usize, end: usize) -> Result<String> {
        if self.entries.is_empty() {
            return Ok("theorem graph is empty".to_string());
        }
        if start > end {
            bail!("invalid range: start must be <= end");
        }
        if start >= self.entries.len() {
            bail!("range start {start} is out of bounds");
        }
        let end = end.min(self.entries.len() - 1);

        let mut out = Vec::new();
        for entry in &self.entries[start..=end] {
            out.push(format_entry_summary(entry));
        }
        Ok(out.join("\n\n"))
    }

    pub(crate) fn list_deps(&self, id: usize) -> Result<String> {
        let entry = self.entry(id)?;
        let mut ids = Vec::with_capacity(entry.dependencies.len() + 1);
        ids.push(id);
        ids.extend(entry.dependencies.iter().copied());

        let mut out = Vec::with_capacity(ids.len());
        for dependency in ids {
            let item = self.entry(dependency)?;
            out.push(format_dependency_summary(item));
        }
        Ok(out.join("\n\n"))
    }

    pub(crate) fn examine(&self, id: usize) -> Result<String> {
        Ok(format_entry_full(self.entry(id)?))
    }

    pub(crate) fn path_to(&self, id: usize) -> Result<String> {
        let ids = self.path_scope(id)?;
        let mut out = Vec::with_capacity(ids.len());
        for theorem_id in ids {
            out.push(format_entry_full(self.entry(theorem_id)?));
        }
        Ok(out.join("\n\n"))
    }

    pub(crate) fn entry_snapshot(&self, id: usize) -> Result<TheoremEntry> {
        Ok(self.entry(id)?.clone())
    }

    pub(crate) fn add_reviews(&mut self, id: usize, review_count: u32) -> Result<u32> {
        let entry = self.entry_mut(id)?;
        entry.reviews = entry.reviews.saturating_add(review_count);
        Ok(entry.reviews)
    }

    pub(crate) fn append_comment(&mut self, id: usize, comment: impl AsRef<str>) -> Result<String> {
        let entry = self.entry_mut(id)?;
        let comment = comment.as_ref().trim();
        if comment.is_empty() {
            bail!("comment cannot be empty");
        }

        match &mut entry.comments {
            Some(existing) if !existing.trim().is_empty() => {
                existing.push_str("\n");
                existing.push_str(comment);
            }
            slot => *slot = Some(comment.to_string()),
        }

        Ok(format!("comment added to theorem entry {id}"))
    }

    pub(crate) fn revise(
        &mut self,
        id: usize,
        proof: impl Into<String>,
        dependencies: Vec<usize>,
    ) -> Result<String> {
        self.entry(id)?;
        let proof = proof.into().trim().to_string();
        if proof.is_empty() {
            bail!("proof cannot be empty");
        }

        self.validate_new_dependencies(Some(id), &dependencies)?;
        let dependencies = normalize_ids(dependencies);
        let old_dependencies = self.entries[id].dependencies.clone();

        for dependency in &old_dependencies {
            if !dependencies.contains(dependency) {
                remove_value(&mut self.entries[*dependency].derivations, id);
            }
        }
        for dependency in &dependencies {
            if !old_dependencies.contains(dependency) {
                insert_sorted_unique(&mut self.entries[*dependency].derivations, id);
            }
        }

        let entry = self.entry_mut(id)?;
        entry.proof = proof;
        entry.dependencies = dependencies;
        entry.comments = None;
        entry.reviews = 0;

        Ok(format!("revised theorem entry {id}"))
    }

    fn path_scope(&self, id: usize) -> Result<Vec<usize>> {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([id]);
        while let Some(current) = queue.pop_front() {
            if !seen.insert(current) {
                continue;
            }
            let entry = self.entry(current)?;
            for dependency in &entry.dependencies {
                queue.push_back(*dependency);
            }
        }
        Ok(seen.into_iter().collect())
    }

    fn validate_new_dependencies(
        &self,
        revised_id: Option<usize>,
        dependencies: &[usize],
    ) -> Result<()> {
        let mut seen = BTreeSet::new();
        for dependency in dependencies {
            if !seen.insert(*dependency) {
                bail!("duplicate dependency id: {dependency}");
            }
            if Some(*dependency) == revised_id {
                bail!("theorem cannot depend on itself");
            }
            if *dependency >= self.entries.len() {
                bail!("dependency id {dependency} does not exist");
            }
        }
        Ok(())
    }

    fn entry(&self, id: usize) -> Result<&TheoremEntry> {
        self.entries
            .get(id)
            .ok_or_else(|| anyhow!("theorem entry {id} does not exist"))
    }

    fn entry_mut(&mut self, id: usize) -> Result<&mut TheoremEntry> {
        self.entries
            .get_mut(id)
            .ok_or_else(|| anyhow!("theorem entry {id} does not exist"))
    }
}

fn format_entry_summary(entry: &TheoremEntry) -> String {
    let mut lines = vec![
        format!("id: {}", entry.id),
        format!("type: {}", format_entry_type(&entry.entry_type)),
        format!("statement: {}", entry.statement),
        format!("dependencies: {:?}", entry.dependencies),
    ];
    if let Some(comments) = &entry.comments {
        lines.push(format!("reviewer comments: {comments}"));
    }
    lines.join("\n")
}

fn format_dependency_summary(entry: &TheoremEntry) -> String {
    let mut lines = vec![
        format!("id: {}", entry.id),
        format!("type: {}", format_entry_type(&entry.entry_type)),
        format!("statement: {}", entry.statement),
        format!("dependencies: {:?}", entry.dependencies),
        format!("reviews: {}", entry.reviews),
    ];
    if let Some(comments) = &entry.comments {
        lines.push(format!("reviewer comments: {comments}"));
    }
    lines.join("\n")
}

fn format_entry_full(entry: &TheoremEntry) -> String {
    let mut lines = vec![
        format!("id: {}", entry.id),
        format!("type: {}", format_entry_type(&entry.entry_type)),
        format!("statement: {}", entry.statement),
        format!("proof: {}", entry.proof),
        format!("dependencies: {:?}", entry.dependencies),
        format!("derivations: {:?}", entry.derivations),
        format!("reviews: {}", entry.reviews),
    ];
    lines.push(format!(
        "comments: {}",
        entry.comments.as_deref().unwrap_or("None")
    ));
    lines.join("\n")
}

fn format_entry_type(entry_type: &TheoremEntryType) -> &'static str {
    match entry_type {
        TheoremEntryType::Context => "context",
        TheoremEntryType::Theorem => "theorem",
    }
}

fn normalize_ids(mut ids: Vec<usize>) -> Vec<usize> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn insert_sorted_unique(values: &mut Vec<usize>, value: usize) {
    match values.binary_search(&value) {
        Ok(_) => {}
        Err(index) => values.insert(index, value),
    }
}

fn remove_value(values: &mut Vec<usize>, value: usize) {
    if let Ok(index) = values.binary_search(&value) {
        values.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::{TheoremEntryType, TheoremGraph};

    #[test]
    fn push_updates_derivations() {
        let mut graph = TheoremGraph::default();
        graph
            .push(TheoremEntryType::Context, "A", "ref", vec![])
            .unwrap();
        graph
            .push(TheoremEntryType::Theorem, "B", "proof", vec![0])
            .unwrap();

        assert_eq!(graph.entries[1].id, 1);
        assert_eq!(graph.entries[1].dependencies, vec![0]);
        assert_eq!(graph.entries[0].derivations, vec![1]);
    }

    #[test]
    fn revise_rewrites_dependency_edges() {
        let mut graph = TheoremGraph::default();
        graph
            .push(TheoremEntryType::Context, "A", "ref", vec![])
            .unwrap();
        graph
            .push(TheoremEntryType::Context, "B", "ref", vec![])
            .unwrap();
        graph
            .push(TheoremEntryType::Theorem, "C", "proof", vec![0])
            .unwrap();

        graph.revise(2, "new proof", vec![1]).unwrap();

        assert!(graph.entries[0].derivations.is_empty());
        assert_eq!(graph.entries[1].derivations, vec![2]);
        assert_eq!(graph.entries[2].dependencies, vec![1]);
        assert_eq!(graph.entries[2].reviews, 0);
        assert!(graph.entries[2].comments.is_none());
    }

    #[test]
    fn append_comment_preserves_existing_comments() {
        let mut graph = TheoremGraph::default();
        graph
            .push(TheoremEntryType::Theorem, "A", "proof", vec![])
            .unwrap();

        graph.append_comment(0, "first issue").unwrap();
        graph.append_comment(0, "second issue").unwrap();

        assert_eq!(
            graph.entries[0].comments.as_deref(),
            Some("first issue\nsecond issue")
        );
    }

    #[test]
    fn add_reviews_accumulates_review_count() {
        let mut graph = TheoremGraph::default();
        graph
            .push(TheoremEntryType::Theorem, "A", "proof", vec![])
            .unwrap();

        let total = graph.add_reviews(0, 4).unwrap();

        assert_eq!(total, 4);
        assert_eq!(graph.entries[0].reviews, 4);
    }
}
