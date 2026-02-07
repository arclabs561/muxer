//! “Novelty” helpers for deterministic exploration.

use crate::stable_hash64;

/// Deterministically pick up to `k` arms that are “unseen” under the provided `observed_calls`.
///
/// If `enabled` is false, this returns an empty list.
#[must_use]
pub fn novelty_pick_unseen<F>(
    seed: u64,
    arms: &[String],
    k: usize,
    enabled: bool,
    mut observed_calls: F,
) -> Vec<String>
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
    let mut scored: Vec<(u64, &String)> =
        items.iter().map(|s| (stable_hash64(seed, s), s)).collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(k.min(items.len()))
        .map(|(_, s)| s.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into()]
    }

    #[test]
    fn picks_unseen_only() {
        let result =
            novelty_pick_unseen(42, &arms(), 5, true, |name| if name == "b" { 1 } else { 0 });
        assert!(!result.contains(&"b".to_string()), "should skip seen arm");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn returns_empty_when_disabled() {
        let result = novelty_pick_unseen(42, &arms(), 5, false, |_| 0);
        assert!(result.is_empty());
    }

    #[test]
    fn returns_empty_when_all_seen() {
        let result = novelty_pick_unseen(42, &arms(), 5, true, |_| 1);
        assert!(result.is_empty());
    }

    #[test]
    fn truncates_to_k() {
        let result = novelty_pick_unseen(42, &arms(), 1, true, |_| 0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn deterministic_given_seed() {
        let a = novelty_pick_unseen(42, &arms(), 3, true, |_| 0);
        let b = novelty_pick_unseen(42, &arms(), 3, true, |_| 0);
        assert_eq!(a, b);
    }

    #[test]
    fn pick_random_subset_basic() {
        let items: Vec<String> = (0..10).map(|i| format!("item{i}")).collect();
        let result = pick_random_subset(42, &items, 3);
        assert_eq!(result.len(), 3);
        // All unique.
        let mut sorted = result.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn pick_random_subset_deterministic() {
        let items: Vec<String> = (0..10).map(|i| format!("item{i}")).collect();
        let a = pick_random_subset(42, &items, 5);
        let b = pick_random_subset(42, &items, 5);
        assert_eq!(a, b);
    }

    #[test]
    fn pick_random_subset_empty() {
        assert!(pick_random_subset(42, &[], 5).is_empty());
        let items = vec!["x".to_string()];
        assert!(pick_random_subset(42, &items, 0).is_empty());
    }
}
