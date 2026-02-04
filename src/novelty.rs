//! “Novelty” helpers for deterministic exploration.

use crate::stable_hash64;

/// Deterministically pick up to `k` arms that are “unseen” under the provided `observed_calls`.
///
/// If `enabled` is false, this returns an empty list.
#[must_use]
pub fn novelty_pick_unseen<F>(seed: u64, arms: &[String], k: usize, enabled: bool, mut observed_calls: F) -> Vec<String>
where
    F: FnMut(&str) -> u64,
{
    if !enabled || k == 0 {
        return Vec::new();
    }
    let mut unseen: Vec<String> = arms
        .iter()
        .filter(|b| observed_calls(b.as_str()) == 0)
        .cloned()
        .collect();
    if unseen.is_empty() {
        return Vec::new();
    }
    unseen.sort_by_key(|b| stable_hash64(seed ^ 0x4E4F_5645, b)); // "NOVE"
    unseen.truncate(k);
    unseen
}

/// Deterministic “random subset” helper.
///
/// This is useful when you want bounded sampling without RNG dependencies.
#[must_use]
pub fn pick_random_subset(seed: u64, items: &[String], k: usize) -> Vec<String> {
    if k == 0 || items.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(u64, &String)> = items.iter().map(|s| (stable_hash64(seed, s), s)).collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(k.min(items.len())).map(|(_, s)| s.clone()).collect()
}

