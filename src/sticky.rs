//! Stickiness / switching-cost wrappers.
//!
//! Production routers often need to avoid “flapping” between arms due to:
//! - cache warmup costs
//! - connection pooling / cold-start penalties
//! - rate-limit recovery dynamics
//! - general operational stability requirements
//!
//! This module provides a small stateful wrapper that can be used on top of selection
//! functions/policies by comparing a scalar score margin before switching.

use std::collections::BTreeMap;

use crate::{
    CandidateDebug, Decision, DecisionNote, DecisionPolicy, MabConfig, MabSelectionDecision,
    Selection,
};

/// Stickiness configuration.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StickyConfig {
    /// Minimum number of consecutive decisions to stay on an arm before allowing a switch.
    ///
    /// - `0` disables dwell control.
    pub min_dwell: u64,
    /// Minimum required score advantage to switch away from the previous arm.
    ///
    /// - `0.0` disables margin control.
    pub min_switch_margin: f64,
}

impl Default for StickyConfig {
    fn default() -> Self {
        Self {
            min_dwell: 0,
            min_switch_margin: 0.0,
        }
    }
}

fn f64_or0(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Compute the scalar score used for selection tie-breaking (higher is better),
/// derived from `CandidateDebug` and `MabConfig`.
///
/// This covers the core multi-objective weights.  Monitoring penalty weights
/// (drift, catKL, CUSUM) live on [`crate::MonitoredMabConfig`] and are not
/// included here -- those scores are still present on [`CandidateDebug`] for
/// inspection, but the sticky wrapper operates on the base selection config.
pub fn mab_scalar_score(c: &CandidateDebug, cfg: MabConfig) -> f64 {
    let cost_w = f64_or0(cfg.cost_weight);
    let lat_w = f64_or0(cfg.latency_weight);
    let junk_w = f64_or0(cfg.junk_weight);
    let hard_w = f64_or0(cfg.hard_junk_weight);

    f64_or0(c.objective_success)
        - cost_w * f64_or0(c.mean_cost_units)
        - lat_w * f64_or0(c.mean_elapsed_ms)
        // Match `select_mab` semantics: soft junk weight should not double-count hard junk.
        - junk_w * f64_or0(c.soft_junk_rate)
        - hard_w * f64_or0(c.hard_junk_rate)
}

/// Stateful stickiness wrapper for deterministic `select_mab`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StickyMab {
    pub cfg: StickyConfig,
    previous: Option<String>,
    dwell: u64,
}

impl StickyMab {
    pub fn new(cfg: StickyConfig) -> Self {
        Self {
            cfg,
            previous: None,
            dwell: 0,
        }
    }

    /// Current previous choice, if any.
    pub fn previous(&self) -> Option<&str> {
        self.previous.as_deref()
    }

    /// Number of consecutive decisions on the current `previous` arm.
    pub fn dwell(&self) -> u64 {
        self.dwell
    }

    /// Reset stickiness state (for tests or epoch resets).
    pub fn reset(&mut self) {
        self.previous = None;
        self.dwell = 0;
    }

    fn scores_by_arm(sel: &Selection) -> BTreeMap<&str, f64> {
        let mut out = BTreeMap::new();
        for c in &sel.candidates {
            out.insert(c.name.as_str(), mab_scalar_score(c, sel.config));
        }
        out
    }

    /// Shared stickiness logic: given a selection, explore_first flag, apply dwell + margin gates.
    /// Returns the (possibly overridden) selection and any sticky-specific notes.
    fn apply_inner(
        &mut self,
        mut sel: Selection,
        explore_first: bool,
    ) -> (Selection, Vec<DecisionNote>) {
        if explore_first {
            self.previous = Some(sel.chosen.clone());
            self.dwell = 1;
            return (sel, Vec::new());
        }

        let candidate = sel.chosen.clone();
        let Some(prev) = self.previous.clone() else {
            self.previous = Some(candidate);
            self.dwell = 1;
            return (sel, Vec::new());
        };

        // If previous is not among the considered candidates, follow the base choice.
        let scores = Self::scores_by_arm(&sel);
        let Some(prev_score) = scores.get(prev.as_str()).copied() else {
            self.previous = Some(candidate);
            self.dwell = 1;
            return (sel, Vec::new());
        };

        // If base choice is the previous arm, just bump dwell.
        if candidate == prev {
            self.dwell = self.dwell.saturating_add(1);
            return (sel, Vec::new());
        }

        // Dwell gate.
        if self.cfg.min_dwell > 0 && self.dwell < self.cfg.min_dwell {
            let note = DecisionNote::StickyKeptPreviousDwell {
                previous: prev.clone(),
                candidate,
                dwell: self.dwell,
                min_dwell: self.cfg.min_dwell,
            };
            sel.chosen = prev;
            self.dwell = self.dwell.saturating_add(1);
            return (sel, vec![note]);
        }

        // Margin gate.
        let cand_score = scores
            .get(candidate.as_str())
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let margin = cand_score - prev_score;
        let min_margin = f64_or0(self.cfg.min_switch_margin);
        if min_margin > 0.0 && !(margin.is_finite() && margin >= min_margin) {
            let note = DecisionNote::StickyKeptPreviousMargin {
                previous: prev.clone(),
                candidate,
                previous_score: prev_score,
                candidate_score: cand_score,
                margin,
                min_margin,
            };
            sel.chosen = prev;
            self.dwell = self.dwell.saturating_add(1);
            return (sel, vec![note]);
        }

        // Allow switch.
        let note = DecisionNote::StickySwitched {
            previous: prev,
            candidate: candidate.clone(),
            previous_score: prev_score,
            candidate_score: cand_score,
            margin,
        };
        self.previous = Some(candidate);
        self.dwell = 1;
        (sel, vec![note])
    }

    /// Apply stickiness to a bare `Selection`.
    ///
    /// Uses a heuristic for explore-first detection: `candidates.len() == 1 && calls == 0`.
    /// Prefer [`apply_mab`](Self::apply_mab) when `MabSelectionDecision` is available.
    pub fn apply(&mut self, sel: Selection) -> Selection {
        let explore_first = sel.candidates.len() == 1 && sel.candidates[0].calls == 0;
        let (sel, _notes) = self.apply_inner(sel, explore_first);
        sel
    }

    /// Apply stickiness to a `MabSelectionDecision`, returning the (possibly overridden) selection.
    pub fn apply_mab(&mut self, decision: MabSelectionDecision) -> Selection {
        let (sel, _notes) = self.apply_inner(decision.selection, decision.explore_first);
        sel
    }

    /// Apply stickiness and return a unified `Decision` (recommended for logging/replay).
    ///
    /// Includes constraint gating metadata and stickiness actions (kept previous / switched).
    pub fn apply_mab_decide(&mut self, decision: MabSelectionDecision) -> Decision {
        let constraints = DecisionNote::Constraints {
            eligible_arms: decision.eligible_arms.clone(),
            fallback_used: decision.constraints_fallback_used,
        };

        let explore_first = decision.explore_first;
        let (sel, sticky_notes) = self.apply_inner(decision.selection, explore_first);

        let mut notes = vec![constraints];
        if explore_first {
            notes.push(DecisionNote::ExploreFirst);
        } else {
            notes.push(DecisionNote::DeterministicChoice);
        }
        notes.extend(sticky_notes);

        Decision {
            policy: DecisionPolicy::Mab,
            chosen: sel.chosen,
            probs: None,
            notes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CandidateDebug, MabConfig};

    fn mk_sel(previous: &str, candidate: &str) -> Selection {
        let cfg = MabConfig::default();
        Selection {
            chosen: candidate.to_string(),
            frontier: vec!["a".to_string(), "b".to_string()],
            candidates: vec![
                CandidateDebug {
                    name: "a".to_string(),
                    calls: 10,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: if previous == "a" { 1.0 } else { 2.0 },
                    drift_score: None,
                    catkl_score: None,
                    cusum_score: None,
                    ok_half_width: None,
                    junk_half_width: None,
                    hard_junk_half_width: None,
                    mean_quality_score: None,
                },
                CandidateDebug {
                    name: "b".to_string(),
                    calls: 10,
                    ok: 0,
                    junk: 0,
                    hard_junk: 0,
                    ok_rate: 0.0,
                    junk_rate: 0.0,
                    hard_junk_rate: 0.0,
                    soft_junk_rate: 0.0,
                    mean_cost_units: 0.0,
                    mean_elapsed_ms: 0.0,
                    ucb: 0.0,
                    objective_success: if previous == "a" { 2.0 } else { 1.0 },
                    drift_score: None,
                    catkl_score: None,
                    cusum_score: None,
                    ok_half_width: None,
                    junk_half_width: None,
                    hard_junk_half_width: None,
                    mean_quality_score: None,
                },
            ],
            config: cfg,
        }
    }

    #[test]
    fn sticky_never_returns_arm_not_in_candidates() {
        let mut sticky = StickyMab::new(StickyConfig {
            min_dwell: 100,
            min_switch_margin: 100.0,
        });

        // Seed previous as "a".
        let _ = sticky.apply(mk_sel("a", "a"));

        // Now provide a selection that does not include "a" at all.
        let cfg = MabConfig::default();
        let sel = Selection {
            chosen: "x".to_string(),
            frontier: vec!["x".to_string()],
            candidates: vec![CandidateDebug {
                name: "x".to_string(),
                calls: 10,
                ok: 0,
                junk: 0,
                hard_junk: 0,
                ok_rate: 0.0,
                junk_rate: 0.0,
                hard_junk_rate: 0.0,
                soft_junk_rate: 0.0,
                mean_cost_units: 0.0,
                mean_elapsed_ms: 0.0,
                ucb: 0.0,
                objective_success: 0.0,
                drift_score: None,
                catkl_score: None,
                cusum_score: None,
                ok_half_width: None,
                junk_half_width: None,
                hard_junk_half_width: None,
                mean_quality_score: None,
            }],
            config: cfg,
        };
        let out = sticky.apply(sel);
        assert_eq!(out.chosen, "x");
    }
}
