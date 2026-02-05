//! Worst-first selection helpers (for regression hunting / triage).

use crate::stable_hash64;

/// Configuration for worst-first scoring.
#[derive(Debug, Clone, Copy)]
pub struct WorstFirstConfig {
    /// Exploration coefficient for the worst-first score.
    pub exploration_c: f64,
    /// Weight applied to hard-junk (instability) rate.
    pub hard_weight: f64,
    /// Weight applied to soft-junk rate.
    pub soft_weight: f64,
}

/// Pick one arm for "worst-first" regression hunting.
///
/// Policy:
/// - Prefer an *unseen* arm first (`observed_calls == 0`) to saturate coverage.
/// - Otherwise pick the arm with the highest "worse" score:
///   \(score = hard_weight * hard_junk + soft_weight * soft_junk + exploration_c * sqrt(ln(total_calls) / calls)\).
///
/// This helper is domain-agnostic: callers supply observed calls and summary rates/calls.
pub fn worst_first_pick_one<FObs, FSum>(
    seed: u64,
    remaining: &[String],
    cfg: WorstFirstConfig,
    mut observed_calls: FObs,
    mut summary: FSum,
) -> Option<(String, bool)>
where
    FObs: FnMut(&str) -> u64,
    FSum: FnMut(&str) -> (u64, f64, f64), // (calls, hard_junk_rate, soft_junk_rate)
{
    if remaining.is_empty() {
        return None;
    }

    // Explore unseen arms first (stable order by deterministic hash).
    let mut unseen: Vec<String> = remaining
        .iter()
        .filter(|b| observed_calls(b.as_str()) == 0)
        .cloned()
        .collect();
    if !unseen.is_empty() {
        unseen.sort_by_key(|b| stable_hash64(seed ^ 0x574F_5253, b)); // "WORS"
        return Some((unseen[0].clone(), true));
    }

    // Otherwise score by "worse" objective + exploration.
    let mut total_calls_f: f64 = 0.0;
    let mut scored: Vec<(f64, String, u64, f64, f64)> = Vec::new();
    for b in remaining {
        let (calls_u64, hard, soft) = summary(b.as_str());
        let calls = (calls_u64 as f64).max(1.0);
        total_calls_f += calls;
        scored.push((0.0, b.clone(), calls_u64, hard, soft));
    }
    let total_calls_f = total_calls_f.max(1.0);

    for row in &mut scored {
        let calls = (row.2 as f64).max(1.0);
        let exploration = cfg.exploration_c * ((total_calls_f.ln() / calls).sqrt());
        let score = cfg.hard_weight * row.3 + cfg.soft_weight * row.4 + exploration;
        row.0 = score;
    }

    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| {
                stable_hash64(seed ^ 0x574F_5253, &a.1)
                    .cmp(&stable_hash64(seed ^ 0x574F_5253, &b.1))
            })
            .then_with(|| a.1.cmp(&b.1))
    });
    let pick = scored.first().map(|r| r.1.clone())?;
    Some((pick, false))
}

/// Pick up to `k` arms for worst-first regression hunting (without replacement).
///
/// Returns a list of `(arm, explore_first)` pairs in selection order.
#[must_use]
pub fn worst_first_pick_k<FObs, FSum>(
    seed: u64,
    arms: &[String],
    k: usize,
    cfg: WorstFirstConfig,
    mut observed_calls: FObs,
    mut summary: FSum,
) -> Vec<(String, bool)>
where
    FObs: FnMut(&str) -> u64,
    FSum: FnMut(&str) -> (u64, f64, f64),
{
    if k == 0 || arms.is_empty() {
        return Vec::new();
    }
    let mut chosen: Vec<(String, bool)> = Vec::new();
    let mut remaining: Vec<String> = arms.to_vec();

    while chosen.len() < k && !remaining.is_empty() {
        let (pick, explore_first) = match worst_first_pick_one(
            seed ^ ((chosen.len() as u64) + 1),
            &remaining,
            cfg,
            |b| observed_calls(b),
            |b| summary(b),
        ) {
            None => break,
            Some(x) => x,
        };

        remaining.retain(|b| b != &pick);
        chosen.push((pick, explore_first));
    }

    chosen
}
