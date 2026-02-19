//! Worst-first selection helpers (regression hunting / active triage).
//!
//! This is the **investigation phase** that follows detection (the `monitor` module).
//! After monitoring flags a change in an arm's behavior, worst-first prioritizes that
//! arm to characterize the change: what broke, by how much, and (in the contextual
//! regime) where in the covariate space.
//!
//! The scoring is intentionally inverted from normal MAB selection: higher badness score
//! = more interesting to investigate.  The UCB exploration term ensures under-sampled
//! arms (including newly-flagged ones) get priority.
//!
//! ## Non-contextual worst-first
//!
//! `worst_first_pick_one` / `worst_first_pick_k` operate per-arm and are sufficient for
//! the non-contextual regime.  After detection, extra traffic is routed to the flagged
//! arm until the monitoring signal decays.
//!
//! ## Contextual extension: per-cell triage
//!
//! In the contextual regime (`LinUcb`), a detected change may be localised to a subset
//! of the covariate space — an arm might degrade only for a specific domain or language
//! feature regime.  Per-arm scoring misses this: it averages across context bins and
//! may dilute a localised signal.
//!
//! The `contextual_worst_first_pick_*` functions lift triage to **(arm, context-bin)**
//! pairs, where bins are derived by quantising each feature dimension into `levels` equal
//! buckets and hashing the resulting bucket vector with `stable_hash64`.  Callers
//! maintain per-cell call/badness counters; these helpers pick which cell to investigate
//! next using the same exploration-adjusted badness score.
//!
//! ## The meta-inference problem
//!
//! The real open problem in post-detection investigation is not "how to switch modes"
//! (normal -> investigation -> back) but **when a detected change invalidates the
//! current objective weighting**.  A level shift in one arm might require only
//! re-estimation; a structural change (the arm's response function changed shape)
//! might require re-evaluating which objectives matter.  This is a meta-level
//! inference problem -- deciding whether to continue optimizing the current objective
//! tuple or revise it -- that existing bandit/RL theory does not address.
//!
//! For practical purposes, `muxer` implements the simpler version: after detection,
//! route extra traffic to the flagged arm (via badness scoring + exploration bonus)
//! until the monitoring signal decays or the arm is manually reset.  The mode
//! transition is implicit (via the guard thresholds in `MabConfig`), not an explicit
//! POMDP policy over objective states.

use crate::{stable_hash64, stable_hash64_u64};

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

// =============================================================================
// Contextual extension: per-(arm, context-bin) worst-first triage
// =============================================================================

/// Configuration for quantising a feature vector into a discrete context bin.
///
/// Each dimension of the feature vector is mapped to a bucket index in `[0, levels)`.
/// Input values are clamped to `[0.0, 1.0]` before quantisation; callers are
/// responsible for normalising features to this range if needed.
/// The bucket vector is hashed with [`stable_hash64`] to produce a single `u64` bin ID.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextBinConfig {
    /// Number of equal-width quantisation levels per feature dimension (1–255).
    ///
    /// - `1`: all contexts map to a single bin (collapses to per-arm triage).
    /// - `4`: 4 buckets per dimension (default; gives ~256 distinct bins for 4-d features).
    /// - Higher values give finer granularity but require more observations per bin.
    pub levels: u8,
    /// Seed mixed into the bin hash for disambiguation.
    pub seed: u64,
}

impl Default for ContextBinConfig {
    fn default() -> Self {
        Self {
            levels: 4,
            seed: 0xC0B1_C0B1,
        }
    }
}

/// Compute a stable context bin ID from a feature vector.
///
/// Each dimension is quantised into `config.levels` equal-width buckets over `[0, 1]`
/// (values are clamped; non-finite values map to bucket 0).  The resulting bucket
/// vector is hashed deterministically with [`stable_hash64`].
///
/// The returned bin ID is stable: the same `context` and `config` always produce the
/// same bin, regardless of arm ordering or call history.
///
/// # Example
///
/// ```rust
/// use muxer::{context_bin, ContextBinConfig};
///
/// let cfg = ContextBinConfig { levels: 4, seed: 0 };
/// let b1 = context_bin(&[0.1, 0.9], cfg);
/// let b2 = context_bin(&[0.1, 0.9], cfg);
/// assert_eq!(b1, b2, "same context → same bin");
///
/// let b3 = context_bin(&[0.9, 0.1], cfg);
/// assert_ne!(b1, b3, "different context → different bin (very likely)");
/// ```
pub fn context_bin(context: &[f64], config: ContextBinConfig) -> u64 {
    let levels = (config.levels.max(1)) as u64;
    // Build a short string key from the quantised bucket indices.
    // Format: "b0:b1:b2:..." where each bi is the bucket index for dimension i.
    let mut key = String::with_capacity(context.len() * 4);
    for (i, &v) in context.iter().enumerate() {
        let clamped = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        let bucket = ((clamped * levels as f64).floor() as u64).min(levels - 1);
        if i > 0 {
            key.push(':');
        }
        key.push_str(&bucket.to_string());
    }
    stable_hash64(config.seed, &key)
}

/// An (arm, context-bin) pair identifying a specific covariate cell for triage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContextualCell {
    /// The arm (provider/backend).
    pub arm: String,
    /// The context bin ID (from [`context_bin`]).
    pub context_bin: u64,
}

impl ContextualCell {
    /// Create a new contextual cell.
    pub fn new(arm: impl Into<String>, context_bin: u64) -> Self {
        Self {
            arm: arm.into(),
            context_bin,
        }
    }
}

/// Pick one (arm, context-bin) cell for contextual worst-first regression triage.
///
/// This is the contextual analogue of [`worst_first_pick_one`].  It operates on the
/// cross-product of `arms × active_bins` and picks the cell with the highest
/// exploration-adjusted badness score.
///
/// Policy:
/// - Prefers *unseen* cells first (`cell_calls(arm, bin) == 0`), in deterministic
///   hash order, so coverage sweeps are reproducible.
/// - Otherwise scores each cell by:
///   \(score = hard\_weight \cdot hard\_junk + soft\_weight \cdot soft\_junk + exploration\_c \cdot \sqrt{\ln(N) / n_{a,j}}\)
///   where \(N\) = total calls across all cells and \(n_{a,j}\) = calls for this cell.
///
/// ## Arguments
///
/// - `seed`: deterministic seed for tie-breaking.
/// - `arms`: candidate arms.
/// - `active_bins`: context bins to consider (e.g. bins seen in recent history).
///   An empty `active_bins` slice returns `None`.
/// - `cfg`: worst-first scoring weights.
/// - `cell_calls`: closure returning the number of calls observed for `(arm, bin)`.
/// - `cell_summary`: closure returning `(calls, hard_junk_rate, soft_junk_rate)` for `(arm, bin)`.
///
/// ## Returns
///
/// `Some((cell, explore_first))` where `explore_first` is `true` if the cell was
/// picked for coverage (no previous observations), or `None` if `arms` or `active_bins`
/// is empty.
#[must_use]
pub fn contextual_worst_first_pick_one<FCalls, FSummary>(
    seed: u64,
    arms: &[String],
    active_bins: &[u64],
    cfg: WorstFirstConfig,
    mut cell_calls: FCalls,
    mut cell_summary: FSummary,
) -> Option<(ContextualCell, bool)>
where
    FCalls: FnMut(&str, u64) -> u64,
    FSummary: FnMut(&str, u64) -> (u64, f64, f64),
{
    if arms.is_empty() || active_bins.is_empty() {
        return None;
    }

    const TIE_SEED: u64 = 0xCE11_C0B1; // "CELL"+"COBI"

    // Phase 1: prefer unseen cells (deterministic hash order).
    let mut unseen: Vec<ContextualCell> = Vec::new();
    for arm in arms {
        for &bin in active_bins {
            if cell_calls(arm.as_str(), bin) == 0 {
                unseen.push(ContextualCell::new(arm.clone(), bin));
            }
        }
    }
    if !unseen.is_empty() {
        unseen.sort_by_key(|c| {
            let h1 = stable_hash64(seed ^ TIE_SEED, &c.arm);
            let h2 = stable_hash64_u64(seed ^ TIE_SEED ^ 1, c.context_bin);
            h1 ^ h2.rotate_left(32)
        });
        return Some((unseen[0].clone(), true));
    }

    // Phase 2: score by badness + UCB exploration.
    let mut total_calls: f64 = 0.0;
    let mut scored: Vec<(f64, ContextualCell, u64, f64, f64)> = Vec::new();
    for arm in arms {
        for &bin in active_bins {
            let (calls_u64, hard, soft) = cell_summary(arm.as_str(), bin);
            let calls = (calls_u64 as f64).max(1.0);
            total_calls += calls;
            scored.push((
                0.0,
                ContextualCell::new(arm.clone(), bin),
                calls_u64,
                hard,
                soft,
            ));
        }
    }
    let total_calls = total_calls.max(1.0);

    for row in &mut scored {
        let calls = (row.2 as f64).max(1.0);
        let exploration = cfg.exploration_c * ((total_calls.ln() / calls).sqrt());
        row.0 = cfg.hard_weight * row.3 + cfg.soft_weight * row.4 + exploration;
    }

    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0).then_with(|| {
            let ha = stable_hash64(seed ^ TIE_SEED, &a.1.arm)
                ^ stable_hash64_u64(seed ^ TIE_SEED ^ 2, a.1.context_bin);
            let hb = stable_hash64(seed ^ TIE_SEED, &b.1.arm)
                ^ stable_hash64_u64(seed ^ TIE_SEED ^ 2, b.1.context_bin);
            ha.cmp(&hb)
        })
    });

    scored.into_iter().next().map(|r| (r.1, false))
}

/// Pick up to `k` (arm, context-bin) cells for contextual worst-first triage (without replacement).
///
/// Returns cells in investigation-priority order.  Each call to `contextual_worst_first_pick_one`
/// uses a perturbed seed so that successive picks explore different cells.
#[must_use]
pub fn contextual_worst_first_pick_k<FCalls, FSummary>(
    seed: u64,
    arms: &[String],
    active_bins: &[u64],
    k: usize,
    cfg: WorstFirstConfig,
    mut cell_calls: FCalls,
    mut cell_summary: FSummary,
) -> Vec<(ContextualCell, bool)>
where
    FCalls: FnMut(&str, u64) -> u64,
    FSummary: FnMut(&str, u64) -> (u64, f64, f64),
{
    if k == 0 || arms.is_empty() || active_bins.is_empty() {
        return Vec::new();
    }

    let mut chosen: Vec<(ContextualCell, bool)> = Vec::new();
    let mut remaining_bins: Vec<u64> = active_bins.to_vec();
    let mut remaining_arms: Vec<String> = arms.to_vec();

    while chosen.len() < k {
        if remaining_arms.is_empty() || remaining_bins.is_empty() {
            break;
        }
        let pick_seed = seed ^ ((chosen.len() as u64) + 1).wrapping_mul(0x9E37_79B9);
        match contextual_worst_first_pick_one(
            pick_seed,
            &remaining_arms,
            &remaining_bins,
            cfg,
            |a, b| cell_calls(a, b),
            |a, b| cell_summary(a, b),
        ) {
            None => break,
            Some((cell, explore)) => {
                // Remove the picked bin across all arms (to avoid returning the same bin twice).
                remaining_bins.retain(|&b| b != cell.context_bin);
                // If a bin is exhausted across all arms, also remove the arm — but only remove
                // arms that have no remaining bins with unseen/bad data.  Here we take the simpler
                // conservative approach: only remove the specific (arm, bin) pair by tracking
                // already-chosen cells.
                chosen.push((cell, explore));
            }
        }
    }

    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arms() -> Vec<String> {
        vec!["a".into(), "b".into(), "c".into()]
    }

    fn default_cfg() -> WorstFirstConfig {
        WorstFirstConfig {
            exploration_c: 1.0,
            hard_weight: 2.0,
            soft_weight: 1.0,
        }
    }

    #[test]
    fn picks_unseen_first() {
        let (pick, explore) =
            worst_first_pick_one(42, &arms(), default_cfg(), |_| 0, |_| (0, 0.0, 0.0)).unwrap();
        assert!(explore, "should flag explore_first for unseen arms");
        assert!(arms().contains(&pick));
    }

    #[test]
    fn prefers_highest_badness() {
        let (pick, explore) = worst_first_pick_one(
            42,
            &arms(),
            default_cfg(),
            |_| 10, // all seen
            |name| match name {
                "a" => (10, 0.1, 0.1),
                "b" => (10, 0.9, 0.9), // worst
                "c" => (10, 0.5, 0.5),
                _ => (10, 0.0, 0.0),
            },
        )
        .unwrap();
        assert!(!explore);
        assert_eq!(pick, "b", "should pick the arm with highest badness");
    }

    #[test]
    fn pick_k_returns_at_most_k() {
        let result = worst_first_pick_k(42, &arms(), 2, default_cfg(), |_| 0, |_| (0, 0.0, 0.0));
        assert_eq!(result.len(), 2);
        // All picks should be unique.
        assert_ne!(result[0].0, result[1].0);
    }

    #[test]
    fn pick_k_empty_arms() {
        let result = worst_first_pick_k(42, &[], 5, default_cfg(), |_| 0, |_| (0, 0.0, 0.0));
        assert!(result.is_empty());
    }

    #[test]
    fn deterministic_given_seed() {
        let a = worst_first_pick_k(99, &arms(), 3, default_cfg(), |_| 0, |_| (0, 0.0, 0.0));
        let b = worst_first_pick_k(99, &arms(), 3, default_cfg(), |_| 0, |_| (0, 0.0, 0.0));
        assert_eq!(a, b);
    }

    // -------------------------------------------------------------------------
    // context_bin tests
    // -------------------------------------------------------------------------

    #[test]
    fn context_bin_is_deterministic() {
        let cfg = ContextBinConfig::default();
        assert_eq!(context_bin(&[0.1, 0.5, 0.9], cfg), context_bin(&[0.1, 0.5, 0.9], cfg));
    }

    #[test]
    fn context_bin_differs_for_different_contexts() {
        let cfg = ContextBinConfig::default();
        let b1 = context_bin(&[0.1, 0.9], cfg);
        let b2 = context_bin(&[0.9, 0.1], cfg);
        assert_ne!(b1, b2);
    }

    #[test]
    fn context_bin_single_level_all_same() {
        let cfg = ContextBinConfig { levels: 1, seed: 0 };
        assert_eq!(context_bin(&[0.0], cfg), context_bin(&[1.0], cfg));
    }

    #[test]
    fn context_bin_clamps_nonfinite() {
        let cfg = ContextBinConfig::default();
        let nan_bin = context_bin(&[f64::NAN, 0.5], cfg);
        let zero_bin = context_bin(&[0.0, 0.5], cfg);
        assert_eq!(nan_bin, zero_bin, "NaN clamped to 0 → same bin as 0.0");
    }

    // -------------------------------------------------------------------------
    // contextual_worst_first_pick_one tests
    // -------------------------------------------------------------------------

    fn bins_ab() -> Vec<u64> {
        vec![1, 2]
    }

    #[test]
    fn contextual_picks_unseen_cell_first() {
        let (cell, explore) = contextual_worst_first_pick_one(
            42,
            &arms(),
            &bins_ab(),
            default_cfg(),
            |_, _| 0,
            |_, _| (0, 0.0, 0.0),
        )
        .unwrap();
        assert!(explore);
        assert!(arms().contains(&cell.arm));
        assert!(bins_ab().contains(&cell.context_bin));
    }

    #[test]
    fn contextual_prefers_worst_cell() {
        let arms2 = vec!["x".to_string(), "y".to_string()];
        let bins = vec![10u64, 20u64];
        let (cell, explore) = contextual_worst_first_pick_one(
            42,
            &arms2,
            &bins,
            default_cfg(),
            |_, _| 10, // all seen
            |arm, bin| match (arm, bin) {
                ("x", 10) => (10, 0.1, 0.1),
                ("x", 20) => (10, 0.2, 0.2),
                ("y", 10) => (10, 0.1, 0.1),
                ("y", 20) => (10, 0.9, 0.9), // worst cell
                _ => (10, 0.0, 0.0),
            },
        )
        .unwrap();
        assert!(!explore);
        assert_eq!(cell.arm, "y");
        assert_eq!(cell.context_bin, 20);
    }

    #[test]
    fn contextual_empty_bins_returns_none() {
        assert!(contextual_worst_first_pick_one(
            42,
            &arms(),
            &[],
            default_cfg(),
            |_, _| 0,
            |_, _| (0, 0.0, 0.0),
        )
        .is_none());
    }

    #[test]
    fn contextual_pick_k_returns_unique_bins() {
        let bins = vec![1u64, 2, 3];
        let result = contextual_worst_first_pick_k(
            42,
            &arms(),
            &bins,
            3,
            default_cfg(),
            |_, _| 0,
            |_, _| (0, 0.0, 0.0),
        );
        assert_eq!(result.len(), 3);
        let picked_bins: Vec<u64> = result.iter().map(|(c, _)| c.context_bin).collect();
        let mut dedup = picked_bins.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), picked_bins.len(), "bins should be unique");
    }

    #[test]
    fn contextual_pick_k_deterministic() {
        let bins = vec![1u64, 2, 3];
        let a = contextual_worst_first_pick_k(
            7,
            &arms(),
            &bins,
            3,
            default_cfg(),
            |_, _| 0,
            |_, _| (0, 0.0, 0.0),
        );
        let b = contextual_worst_first_pick_k(
            7,
            &arms(),
            &bins,
            3,
            default_cfg(),
            |_, _| 0,
            |_, _| (0, 0.0, 0.0),
        );
        assert_eq!(a, b);
    }
}
