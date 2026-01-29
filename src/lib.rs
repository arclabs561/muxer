//! `muxer`: deterministic, multi-objective bandit-style routing primitives.
//!
//! This crate is designed for “provider routing” style problems:
//! you have a small set of arms (providers) and repeated calls that produce
//! outcomes (ok/fail, rate limit, cost, latency, “junk”).
//!
//! Goals:
//! - **Deterministic by default**: same stats + config → same choice.
//! - **Non-stationarity friendly**: prefer **sliding-window** summaries over lifetime averages.
//! - **Multi-objective**: choose on a Pareto frontier, then deterministic scalarization.
//!
//! Non-goals:
//! - This is not a full contextual-bandit library.

#![forbid(unsafe_code)]

use pare::{Direction, ParetoFrontier};
use std::collections::{BTreeMap, VecDeque};

mod alloc;
pub use alloc::*;

mod exp3ix;
pub use exp3ix::*;

mod thompson;
pub use thompson::*;

/// A single observed outcome for an arm.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Outcome {
    /// Whether the request succeeded for this arm.
    pub ok: bool,
    /// Whether the request failed due to rate limiting (HTTP 429).
    pub http_429: bool,
    /// Whether the downstream result was judged “low value” (e.g. blocked, empty extraction).
    pub junk: bool,
    /// Whether the junk was “hard” (e.g. JS/auth wall) vs “soft” (low-signal extraction).
    pub hard_junk: bool,
    /// Provider-specific cost units for this call (caller-defined).
    pub cost_units: u64,
    /// Total elapsed time for this call, in milliseconds.
    pub elapsed_ms: u64,
}

/// Sliding-window statistics for an arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Window {
    cap: usize,
    buf: VecDeque<Outcome>,
}

impl Window {
    /// Create an empty window with capacity `cap` (minimum 1).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            buf: VecDeque::new(),
        }
    }

    /// Maximum number of outcomes retained.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Number of outcomes currently retained.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the window has no outcomes.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Push a new outcome, evicting the oldest if at capacity.
    pub fn push(&mut self, o: Outcome) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(o);
    }

    /// Best-effort: mutate the most recent outcome.
    ///
    /// This is useful when an outcome's "junk" label is only known after downstream processing
    /// (e.g. search succeeded, but extraction later looked low-signal).
    pub fn set_last_junk(&mut self, junk: bool) {
        if let Some(last) = self.buf.back_mut() {
            last.junk = junk;
        }
    }

    /// Best-effort: set “junk” and whether it is “hard junk” for the most recent outcome.
    pub fn set_last_junk_level(&mut self, junk: bool, hard_junk: bool) {
        if let Some(last) = self.buf.back_mut() {
            last.junk = junk;
            last.hard_junk = hard_junk && junk;
        }
    }

    /// Summarize the current window as counts and sums.
    pub fn summary(&self) -> Summary {
        let n = self.buf.len() as u64;
        if n == 0 {
            return Summary::default();
        }
        let mut ok = 0u64;
        let mut http_429 = 0u64;
        let mut junk = 0u64;
        let mut hard_junk = 0u64;
        let mut cost_units = 0u64;
        let mut elapsed_ms_sum = 0u64;
        for o in &self.buf {
            ok += o.ok as u64;
            http_429 += o.http_429 as u64;
            junk += o.junk as u64;
            hard_junk += o.hard_junk as u64;
            cost_units = cost_units.saturating_add(o.cost_units);
            elapsed_ms_sum = elapsed_ms_sum.saturating_add(o.elapsed_ms);
        }
        Summary {
            calls: n,
            ok,
            http_429,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms_sum,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Aggregate counts/sums over a window of outcomes.
pub struct Summary {
    /// Number of calls observed.
    pub calls: u64,
    /// Number of successful calls.
    pub ok: u64,
    /// Number of HTTP 429 (rate-limited) calls.
    pub http_429: u64,
    /// Number of calls judged “junk”.
    pub junk: u64,
    /// Number of calls judged “hard junk”.
    pub hard_junk: u64,
    /// Sum of `cost_units` over calls.
    pub cost_units: u64,
    /// Sum of `elapsed_ms` over calls.
    pub elapsed_ms_sum: u64,
}

impl Summary {
    /// Fraction of calls that succeeded.
    pub fn ok_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.ok as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were rate-limited (HTTP 429).
    pub fn http_429_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.http_429 as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “junk”.
    pub fn junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.junk as f64) / (self.calls as f64)
        }
    }

    /// Mean `cost_units` per call.
    pub fn mean_cost_units(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.cost_units as f64) / (self.calls as f64)
        }
    }

    /// Mean `elapsed_ms` per call.
    pub fn mean_elapsed_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.elapsed_ms_sum as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “hard junk”.
    pub fn hard_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            (self.hard_junk as f64) / (self.calls as f64)
        }
    }

    /// Fraction of calls that were judged “soft junk” (junk but not hard junk).
    pub fn soft_junk_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            let soft = self.junk.saturating_sub(self.hard_junk);
            (soft as f64) / (self.calls as f64)
        }
    }
}

/// Configuration knobs for deterministic MAB-style selection.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MabConfig {
    /// UCB exploration coefficient.
    pub exploration_c: f64,
    /// Penalty weight for mean cost_units (0 disables).
    pub cost_weight: f64,
    /// Penalty weight for mean latency in ms (0 disables).
    pub latency_weight: f64,
    /// Penalty weight for junk_rate (0 disables).
    pub junk_weight: f64,
    /// Penalty weight for hard_junk_rate (0 disables).
    pub hard_junk_weight: f64,
    /// Optional constraint: discard arms whose windowed junk_rate exceeds this.
    pub max_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed hard_junk_rate exceeds this.
    pub max_hard_junk_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed http_429_rate exceeds this.
    pub max_http_429_rate: Option<f64>,
    /// Optional constraint: discard arms whose windowed mean_cost_units exceeds this.
    pub max_mean_cost_units: Option<f64>,
}

impl Default for MabConfig {
    fn default() -> Self {
        Self {
            exploration_c: 0.7,
            cost_weight: 0.0,
            latency_weight: 0.0,
            junk_weight: 0.0,
            hard_junk_weight: 0.0,
            max_junk_rate: None,
            max_hard_junk_rate: None,
            max_http_429_rate: None,
            max_mean_cost_units: None,
        }
    }
}

/// Debug row for one candidate arm.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CandidateDebug {
    /// Arm name.
    pub name: String,
    /// Calls observed in the summary.
    pub calls: u64,
    /// Successes observed in the summary.
    pub ok: u64,
    /// Rate-limit events observed in the summary.
    pub http_429: u64,
    /// Junk outcomes observed in the summary.
    pub junk: u64,
    /// Hard junk outcomes observed in the summary.
    pub hard_junk: u64,
    /// Success rate.
    pub ok_rate: f64,
    /// HTTP 429 rate.
    pub http_429_rate: f64,
    /// Junk rate.
    pub junk_rate: f64,
    /// Hard junk rate.
    pub hard_junk_rate: f64,
    /// Soft junk rate.
    pub soft_junk_rate: f64,
    /// Mean cost units per call.
    pub mean_cost_units: f64,
    /// Mean latency (ms) per call.
    pub mean_elapsed_ms: f64,
    /// UCB exploration term.
    pub ucb: f64,
    /// Scalar objective used for the frontier’s success dimension (junk is a separate objective).
    pub objective_success: f64,
}

/// Output of `select_mab` (chosen arm + debugging context).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Selection {
    /// The selected arm.
    pub chosen: String,
    /// Arm names on the Pareto frontier (in the frontier’s iteration order).
    pub frontier: Vec<String>,
    /// Candidate debug rows (in the input arm order).
    pub candidates: Vec<CandidateDebug>,
    /// The config used to compute this selection.
    pub config: MabConfig,
}

/// Deterministic selection:
/// - Explore each arm at least once (in stable order).
/// - Then:
///   - build a Pareto frontier over:
///     - maximize success (ok_rate with 429 penalty, plus UCB)
///     - minimize mean cost_units
///     - minimize mean latency
///     - minimize junk_rate
///   - pick the best scalarized point (with stable tie-break)
pub fn select_mab(
    arms_in_order: &[String],
    summaries: &BTreeMap<String, Summary>,
    cfg: MabConfig,
) -> Selection {
    // Apply hard constraints (BwK-ish “anytime” gating).
    // If constraints filter everything, fall back to the original arm set (never return empty).
    let mut eligible: Vec<String> = Vec::new();
    for a in arms_in_order {
        let s = summaries.get(a).copied().unwrap_or_default();
        let ok = cfg
            .max_junk_rate
            .map(|thr| s.junk_rate() <= thr)
            .unwrap_or(true)
            && cfg
                .max_hard_junk_rate
                .map(|thr| s.hard_junk_rate() <= thr)
                .unwrap_or(true)
            && cfg
                .max_http_429_rate
                .map(|thr| s.http_429_rate() <= thr)
                .unwrap_or(true)
            && cfg
                .max_mean_cost_units
                .map(|thr| s.mean_cost_units() <= thr)
                .unwrap_or(true);
        if ok {
            eligible.push(a.clone());
        }
    }
    let arms_in_order: &[String] = if eligible.is_empty() {
        arms_in_order
    } else {
        &eligible
    };

    // Explore first.
    for a in arms_in_order {
        if summaries.get(a).copied().unwrap_or_default().calls == 0 {
            return Selection {
                chosen: a.clone(),
                frontier: vec![a.clone()],
                candidates: vec![CandidateDebug {
                    name: a.clone(),
                    calls: 0,
                    ok: 0,
                    http_429: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    http_429_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: 0.0,
                }],
                config: cfg,
            };
        }
    }

    // Only count calls for the arms actually being considered (important if `summaries`
    // contains additional entries beyond `arms_in_order`).
    let total_calls: f64 = arms_in_order
        .iter()
        .map(|a| summaries.get(a).copied().unwrap_or_default().calls as f64)
        .sum::<f64>()
        .max(1.0);

    // Pareto frontier (maximize space):
    // - objective_success: maximize
    // - costs/latency/junk: minimize => negate
    //
    // We keep the frontier computation separate from scalarization so callers can
    // inspect "who is on the frontier" deterministically.
    let mut frontier_points: Vec<Vec<f64>> = Vec::new();
    let mut frontier_names_in_order: Vec<String> = Vec::new();

    let mut candidates = Vec::new();
    for a in arms_in_order {
        let s = summaries.get(a).copied().unwrap_or_default();
        let n = (s.calls as f64).max(1.0);
        let ok_rate = s.ok_rate();
        let rl_rate = s.http_429_rate();
        let junk_rate = s.junk_rate();
        let hard_junk_rate = s.hard_junk_rate();
        let soft_junk_rate = s.soft_junk_rate();

        // Success objective discourages providers that are 429-ing right now.
        // Junk is handled as a separate minimized objective (and optionally weighted).
        let effective_ok = ok_rate * (1.0 - rl_rate);
        let ucb = cfg.exploration_c * ((total_calls.ln() / n).sqrt());
        let objective_success = effective_ok + ucb;

        let mean_cost = s.mean_cost_units();
        let mean_lat = s.mean_elapsed_ms();

        candidates.push(CandidateDebug {
            name: a.clone(),
            calls: s.calls,
            ok: s.ok,
            http_429: s.http_429,
            junk: s.junk,
            hard_junk: s.hard_junk,
            ok_rate,
            http_429_rate: rl_rate,
            junk_rate,
            hard_junk_rate,
            soft_junk_rate,
            mean_cost_units: mean_cost,
            mean_elapsed_ms: mean_lat,
            ucb,
            objective_success,
        });

        frontier_points.push(vec![
            objective_success,
            -mean_cost,
            -mean_lat,
            -hard_junk_rate,
            -soft_junk_rate,
        ]);
        frontier_names_in_order.push(a.clone());
    }

    let mut frontier = ParetoFrontier::new(vec![Direction::Maximize; 5]);
    for (i, vals) in frontier_points.iter().enumerate() {
        frontier.push(vals.clone(), i);
    }
    let mut frontier_indices: Vec<usize> = if frontier.is_empty() {
        (0..frontier_points.len()).collect()
    } else {
        frontier.points().iter().map(|p| p.data).collect()
    };
    // Ensure stable ordering regardless of frontier internals.
    frontier_indices.sort_unstable();

    let frontier_names: Vec<String> = frontier_indices
        .iter()
        .filter_map(|&i| frontier_names_in_order.get(i).cloned())
        .collect();

    // Deterministic scalarization among frontier points.
    let weights: [f64; 5] = [
        1.0,
        cfg.cost_weight.max(0.0),
        cfg.latency_weight.max(0.0),
        cfg.hard_junk_weight.max(0.0),
        cfg.junk_weight.max(0.0),
    ];
    let mut best_name = frontier_names
        .first()
        .cloned()
        .unwrap_or_else(|| arms_in_order.first().cloned().unwrap_or_default());
    let mut best_score = f64::NEG_INFINITY;
    for &idx in &frontier_indices {
        let Some(c) = candidates.get(idx) else {
            continue;
        };
        // Maximize:
        //   objective_success - w_cost * cost - w_lat * latency - w_hard * hard_junk - w_soft * soft_junk
        let s = c.objective_success
            - weights[1] * c.mean_cost_units
            - weights[2] * c.mean_elapsed_ms
            - weights[3] * c.hard_junk_rate
            - weights[4] * c.soft_junk_rate;
        if s > best_score || ((s - best_score).abs() <= 1e-12 && c.name < best_name) {
            best_score = s;
            best_name = c.name.clone();
        }
    }

    Selection {
        chosen: best_name,
        frontier: frontier_names,
        candidates,
        config: cfg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn s(
        calls: u64,
        ok: u64,
        http_429: u64,
        junk: u64,
        hard_junk: u64,
        cost_units: u64,
        elapsed_ms_sum: u64,
    ) -> Summary {
        Summary {
            calls,
            ok,
            http_429,
            junk,
            hard_junk,
            cost_units,
            elapsed_ms_sum,
        }
    }

    #[test]
    fn select_mab_is_deterministic_and_prefers_lower_junk_all_else_equal() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        // Same ok/http_429/cost/lat, but different junk.
        m.insert("a".to_string(), s(10, 9, 0, 5, 0, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 0, 0, 0, 10, 1000));

        let sel1 = select_mab(&arms, &m, MabConfig::default());
        let sel2 = select_mab(&arms, &m, MabConfig::default());
        assert_eq!(sel1.chosen, "b");
        assert_eq!(sel1.chosen, sel2.chosen);
    }

    #[test]
    fn constraints_filter_arms_but_never_return_empty() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(10, 9, 0, 9, 0, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 0, 9, 0, 10, 1000));

        let cfg = MabConfig {
            max_junk_rate: Some(0.1),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &m, cfg);
        assert!(!sel.chosen.is_empty());
        assert!(sel.frontier.iter().any(|x| x == &sel.chosen));
    }

    #[test]
    fn constraints_can_exclude_high_429_arm() {
        let arms = vec!["brave".to_string(), "tavily".to_string()];
        let mut m = BTreeMap::new();
        m.insert("brave".to_string(), s(10, 10, 8, 0, 0, 10, 1000));
        m.insert("tavily".to_string(), s(10, 9, 0, 0, 0, 10, 1000));

        let cfg = MabConfig {
            max_http_429_rate: Some(0.5),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &m, cfg);
        assert_eq!(sel.chosen, "tavily");
    }

    #[test]
    fn constraints_can_exclude_high_hard_junk_arm() {
        let arms = vec!["a".to_string(), "b".to_string()];
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), s(10, 9, 0, 1, 1, 10, 1000));
        m.insert("b".to_string(), s(10, 9, 0, 1, 0, 10, 1000));

        let cfg = MabConfig {
            max_hard_junk_rate: Some(0.05),
            ..MabConfig::default()
        };
        let sel = select_mab(&arms, &m, cfg);
        assert_eq!(sel.chosen, "b");
    }

    proptest! {
        #[test]
        fn select_mab_never_panics_and_returns_member_of_arms(
            // Keep this intentionally small/bounded to avoid slow tests.
            calls_a in 0u64..50,
            calls_b in 0u64..50,
            ok_a in 0u64..50,
            ok_b in 0u64..50,
            http_a in 0u64..50,
            http_b in 0u64..50,
            junk_a in 0u64..50,
            junk_b in 0u64..50,
            hard_a in 0u64..50,
            hard_b in 0u64..50,
            cost_a in 0u64..500,
            cost_b in 0u64..500,
            lat_a in 0u64..50_000,
            lat_b in 0u64..50_000,
        ) {
            let arms = vec!["a".to_string(), "b".to_string()];
            let mut m = BTreeMap::new();

            // Sanitize counts so they never exceed calls.
            let sa = s(
                calls_a,
                ok_a.min(calls_a),
                http_a.min(calls_a),
                junk_a.min(calls_a),
                hard_a.min(junk_a.min(calls_a)),
                cost_a,
                lat_a,
            );
            let sb = s(
                calls_b,
                ok_b.min(calls_b),
                http_b.min(calls_b),
                junk_b.min(calls_b),
                hard_b.min(junk_b.min(calls_b)),
                cost_b,
                lat_b,
            );
            m.insert("a".to_string(), sa);
            m.insert("b".to_string(), sb);

            let cfg = MabConfig {
                exploration_c: 0.7,
                cost_weight: 0.0,
                latency_weight: 0.0,
                junk_weight: 0.0,
                hard_junk_weight: 0.0,
                max_junk_rate: None,
                max_hard_junk_rate: None,
                max_http_429_rate: None,
                max_mean_cost_units: None,
            };

            let sel = select_mab(&arms, &m, cfg);
            prop_assert!(sel.chosen == "a" || sel.chosen == "b");
            prop_assert!(sel.frontier.iter().any(|x| x == &sel.chosen));

            // Determinism: same input -> same output.
            let sel2 = select_mab(&arms, &m, cfg);
            prop_assert_eq!(sel.chosen, sel2.chosen);
        }
    }
}
