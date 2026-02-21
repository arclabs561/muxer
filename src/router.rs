//! Stateful routing session: the "front door" for most production deployments.
//!
//! [`Router`] owns all per-arm state (sliding windows, monitoring, triage) and
//! exposes a simple three-method interface:
//!
//! ```text
//! let d = router.select(k, seed);   // pick k arms
//! run_requests(d.chosen());          // your code
//! router.observe(arm, outcome);      // record what happened
//! ```
//!
//! The router handles the full routing lifecycle:
//!
//! 1. **Normal** — MAB selection with novelty/coverage pre-picks + latency guardrail.
//! 2. **Triage** — after monitoring fires on an arm, worst-first routing
//!    investigates the alarmed arm while the rest continues normally.
//! 3. **Acknowledgment** — [`Router::acknowledge_change`] resets the CUSUM bank
//!    and promotes the arm's recent window into its baseline.
//!
//! ## Large K
//!
//! For large arm counts (K > 20), set `k > 1` in [`Router::select`] to batch
//! the initial explore-first phase: with k=3, initial coverage of K=30 arms
//! takes ~10 rounds instead of 30.  [`RouterConfig`]'s `coverage` field
//! (`CoverageConfig::min_fraction`) ensures no arm is permanently starved.

use crate::monitor::{DriftConfig, MonitoredWindow};
use crate::{
    policy_fill_generic, select_mab_explain,
    select_mab_monitored_explain_with_summaries, worst_first_pick_k, ControlConfig,
    ContextualCell, CoverageConfig, DriftMetric, LatencyGuardrailConfig, MabConfig, Outcome,
    OutcomeIdx, PipelineOrder, Summary, TriageSession, TriageSessionConfig, Window,
    WorstFirstConfig, split_control_budget,
};
use std::collections::BTreeMap;

// ============================================================================
// Configuration
// ============================================================================

/// Full configuration for a [`Router`].
///
/// Start with [`RouterConfig::default()`] and enable features via the builder
/// methods or by setting fields directly.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RouterConfig {
    // --- Selection ---
    /// MAB selection configuration (UCB weights, constraints, etc.).
    pub mab: MabConfig,
    /// Drift config for monitored selection.
    pub drift: DriftConfig,
    /// Capacity of the sliding selection window per arm.
    ///
    /// Use [`crate::suggested_window_cap`] to pick this based on expected throughput
    /// and changepoint rate.  For K > 20 arms, consider smaller values (50–100)
    /// to bound memory.
    pub window_cap: usize,

    // --- Monitoring (optional) ---
    /// Enable drift/catKL/CUSUM monitoring (`MonitoredWindow`s per arm).
    ///
    /// When `false`, only plain `Window`s are maintained; drift guards and
    /// monitored selection are unavailable.
    pub enable_monitoring: bool,
    /// Baseline window capacity (for `MonitoredWindow`).
    pub baseline_cap: usize,
    /// Recent window capacity (for `MonitoredWindow`).
    pub recent_cap: usize,

    // --- Triage (optional) ---
    /// Triage session config.  `None` disables detection and triage.
    pub triage_cfg: Option<TriageSessionConfig>,
    /// Worst-first scoring for the triage investigation phase.
    pub triage_wf: WorstFirstConfig,
    /// Fraction of k picks to route to alarmed arms during triage (clamped 0..1).
    ///
    /// Example: `0.5` means half the picks go to worst-first investigation,
    /// half go to normal MAB selection.  Default: 0.5.
    pub triage_fraction: f64,

    // --- Pipeline ---
    /// Coverage / maintenance sampling.
    pub coverage: CoverageConfig,
    /// Latency guardrail (hard pre-filter by mean elapsed time).
    pub guardrail: LatencyGuardrailConfig,
    /// Pipeline ordering: whether novelty/coverage runs before or after the guardrail.
    pub pipeline_order: PipelineOrder,
    /// Enable novelty pre-picks (explore unseen arms first).
    pub novelty_enabled: bool,

    // --- Control ---
    /// Control arm budget (deterministic-random picks as a bias anchor).
    pub control: ControlConfig,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            mab: MabConfig::default(),
            drift: DriftConfig {
                metric: DriftMetric::Hellinger,
                tol: 1e-9,
                min_baseline: 20,
                min_recent: 10,
            },
            window_cap: 100,
            enable_monitoring: false,
            baseline_cap: 500,
            recent_cap: 50,
            triage_cfg: None,
            triage_wf: WorstFirstConfig {
                exploration_c: 1.0,
                hard_weight: 3.0,
                soft_weight: 1.0,
            },
            triage_fraction: 0.5,
            coverage: CoverageConfig::default(),
            guardrail: LatencyGuardrailConfig::default(),
            pipeline_order: PipelineOrder::NoveltyFirst,
            novelty_enabled: true,
            control: ControlConfig::default(),
        }
    }
}

impl RouterConfig {
    /// Enable monitoring with given window capacities.
    pub fn with_monitoring(mut self, baseline_cap: usize, recent_cap: usize) -> Self {
        self.enable_monitoring = true;
        self.baseline_cap = baseline_cap;
        self.recent_cap = recent_cap;
        self
    }

    /// Enable triage with default `TriageSessionConfig`.
    pub fn with_triage(mut self) -> Self {
        self.triage_cfg = Some(TriageSessionConfig::default());
        self
    }

    /// Enable triage with a custom config.
    pub fn with_triage_cfg(mut self, cfg: TriageSessionConfig) -> Self {
        self.triage_cfg = Some(cfg);
        self
    }

    /// Set the selection window capacity.
    pub fn window_cap(mut self, cap: usize) -> Self {
        self.window_cap = cap;
        self
    }

    /// Enable coverage sampling with the given minimum fraction.
    pub fn with_coverage(mut self, min_fraction: f64, min_floor: u64) -> Self {
        self.coverage = CoverageConfig { enabled: true, min_fraction, min_calls_floor: min_floor };
        self
    }

    /// Enable latency guardrail.
    pub fn with_guardrail(mut self, max_mean_ms: f64) -> Self {
        self.guardrail = LatencyGuardrailConfig {
            max_mean_ms: Some(max_mean_ms),
            require_measured: false,
            allow_fewer: true,
        };
        self
    }

    /// Reserve control picks.
    pub fn with_control(mut self, control_k: usize) -> Self {
        self.control = ControlConfig::with_k(control_k);
        self
    }
}

// ============================================================================
// RouterMode and RouterDecision
// ============================================================================

/// Current operating mode of the [`Router`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RouterMode {
    /// Normal operation: MAB selection with coverage/novelty/guardrail.
    Normal,
    /// Regression detected on one or more arms.
    ///
    /// The router is routing extra investigation traffic to the flagged arms
    /// via worst-first selection.
    Triage {
        /// Arms whose CUSUM bank has alarmed.
        alarmed_arms: Vec<String>,
    },
}

impl RouterMode {
    /// Whether the router is currently in triage mode.
    pub fn is_triage(&self) -> bool {
        matches!(self, RouterMode::Triage { .. })
    }

    /// Arms that have alarmed, if any.
    pub fn alarmed_arms(&self) -> &[String] {
        match self {
            RouterMode::Triage { alarmed_arms } => alarmed_arms,
            RouterMode::Normal => &[],
        }
    }
}

/// Output of a single [`Router::select`] call.
#[derive(Debug, Clone)]
pub struct RouterDecision {
    /// Chosen arms in order (primary picks first, then MAB fills).
    pub chosen: Vec<String>,
    /// Current routing mode at decision time.
    pub mode: RouterMode,
    /// Arms chosen as novelty/coverage pre-picks (subset of `chosen`).
    pub prechosen: Vec<String>,
    /// Arms that were control (random) picks (subset of `chosen`).
    pub control_picks: Vec<String>,
    /// Arms eligible for MAB selection after pre-picks and guardrail.
    pub mab_eligible: Vec<String>,
    /// Triage cells (arm × context-bin) to route investigation traffic to.
    ///
    /// Non-empty only in [`RouterMode::Triage`].
    pub triage_cells: Vec<ContextualCell>,
}

impl RouterDecision {
    fn empty(mode: RouterMode) -> Self {
        Self {
            chosen: Vec::new(),
            mode,
            prechosen: Vec::new(),
            control_picks: Vec::new(),
            mab_eligible: Vec::new(),
            triage_cells: Vec::new(),
        }
    }

    /// The primary chosen arm (first in `chosen`), if any.
    pub fn primary(&self) -> Option<&str> {
        self.chosen.first().map(|s| s.as_str())
    }
}

// ============================================================================
// Router
// ============================================================================

/// Stateful routing session.
///
/// Owns all per-arm window state and orchestrates the full routing lifecycle:
/// normal selection → monitoring → triage → acknowledgment.
///
/// ## Lifecycle
///
/// ```rust
/// use muxer::{Router, RouterConfig, Outcome};
///
/// let arms = vec!["a".to_string(), "b".to_string()];
/// let mut router = Router::new(arms, RouterConfig::default()).unwrap();
///
/// // Each round:
/// let d = router.select(1, 0);         // pick 1 arm
/// let arm = d.primary().unwrap().to_string();
/// // ... make the call ...
/// let outcome = Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 50, quality_score: None };
/// router.observe(&arm, outcome);       // record result
///
/// // After detecting a regression:
/// if router.mode().is_triage() {
///     // route extra traffic to investigation arms via d.triage_cells
///     router.acknowledge_change("a"); // reset CUSUM when regression is resolved
/// }
/// ```
pub struct Router {
    arms: Vec<String>,
    windows: BTreeMap<String, Window>,
    monitored: Option<BTreeMap<String, MonitoredWindow>>,
    triage: Option<TriageSession>,
    cfg: RouterConfig,
    total_observations: u64,
}

impl Router {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new router for the given arms and configuration.
    ///
    /// Returns an error if triage is enabled but the CUSUM bank cannot be
    /// initialised (e.g. invalid simplex in `TriageSessionConfig`).
    pub fn new(arms: Vec<String>, cfg: RouterConfig) -> Result<Self, logp::Error> {
        let windows: BTreeMap<String, Window> = arms
            .iter()
            .map(|a| (a.clone(), Window::new(cfg.window_cap.max(1))))
            .collect();

        let monitored = if cfg.enable_monitoring {
            Some(
                arms.iter()
                    .map(|a| {
                        (
                            a.clone(),
                            MonitoredWindow::new(
                                cfg.baseline_cap.max(1),
                                cfg.recent_cap.max(1),
                            ),
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };

        let triage = if let Some(ref tcfg) = cfg.triage_cfg {
            Some(TriageSession::new(&arms, tcfg.clone())?)
        } else {
            None
        };

        Ok(Self {
            arms,
            windows,
            monitored,
            triage,
            cfg,
            total_observations: 0,
        })
    }

    /// Add a new arm at runtime.
    ///
    /// The arm starts with empty windows and will be explored by novelty
    /// pre-picks on the next `select` call.
    ///
    /// Returns an error if triage is enabled and the CUSUM bank for the new
    /// arm cannot be initialised.
    pub fn add_arm(&mut self, arm: String) -> Result<(), logp::Error> {
        if self.arms.contains(&arm) {
            return Ok(());
        }
        self.windows.insert(arm.clone(), Window::new(self.cfg.window_cap.max(1)));
        if let Some(ref mut m) = self.monitored {
            m.insert(
                arm.clone(),
                MonitoredWindow::new(self.cfg.baseline_cap.max(1), self.cfg.recent_cap.max(1)),
            );
        }
        if let Some(ref mut t) = self.triage {
            let tcfg = self.cfg.triage_cfg.clone().unwrap_or_default();
            // Rebuild triage with the new arm included.
            let mut all_arms = self.arms.clone();
            all_arms.push(arm.clone());
            *t = TriageSession::new(&all_arms, tcfg)?;
        }
        self.arms.push(arm);
        Ok(())
    }

    /// Remove an arm. Its windows and triage state are discarded.
    pub fn remove_arm(&mut self, arm: &str) {
        self.arms.retain(|a| a != arm);
        self.windows.remove(arm);
        if let Some(ref mut m) = self.monitored {
            m.remove(arm);
        }
        // TriageSession doesn't support removal; rebuild if needed.
        if self.triage.is_some() {
            let tcfg = self.cfg.triage_cfg.clone().unwrap_or_default();
            self.triage = TriageSession::new(&self.arms, tcfg).ok();
        }
    }

    // -----------------------------------------------------------------------
    // Core interface
    // -----------------------------------------------------------------------

    /// Select up to `k` arms for this decision round.
    ///
    /// Does not mutate any window state.  Call [`observe`] afterwards to record
    /// outcomes.
    ///
    /// ### Large K
    ///
    /// With many arms (K > 20), use `k > 1` to batch the initial explore-first
    /// phase.  With k=3, K=30 arms reach initial coverage in ~10 rounds instead
    /// of 30.  Enable `CoverageConfig` to ensure no arm is permanently starved.
    pub fn select(&self, k: usize, seed: u64) -> RouterDecision {
        if k == 0 || self.arms.is_empty() {
            return RouterDecision::empty(self.mode());
        }

        let mode = self.mode();
        let alarmed = mode.alarmed_arms().to_vec();
        let mut chosen: Vec<String> = Vec::new();
        let mut triage_cells: Vec<ContextualCell> = Vec::new();

        // -- Step 1: Control picks (bias anchor) --
        let (control_picks, remaining_k) =
            split_control_budget(seed ^ 0xC0E1_1A11, &self.arms, k, self.cfg.control);
        for a in &control_picks {
            chosen.push(a.clone());
        }

        let remaining_arms: Vec<String> = self
            .arms
            .iter()
            .filter(|a| !chosen.contains(*a))
            .cloned()
            .collect();

        // -- Step 2: Triage picks (investigation traffic) --
        let triage_k = if !alarmed.is_empty() {
            let frac = self.cfg.triage_fraction.clamp(0.0, 1.0);
            ((remaining_k as f64 * frac).ceil() as usize)
                .max(1)
                .min(alarmed.len())
                .min(remaining_k.saturating_sub(1).max(1))
        } else {
            0
        };

        if triage_k > 0 {
            let triage_arms: Vec<String> =
                alarmed.iter().filter(|a| !chosen.contains(*a)).cloned().collect();
            let picks = worst_first_pick_k(
                seed ^ 0x5452_4947,
                &triage_arms,
                triage_k,
                self.cfg.triage_wf,
                |arm| self.windows.get(arm).map(|w| w.len() as u64).unwrap_or(0),
                |arm| {
                    let s = self.windows.get(arm).map(|w| w.summary()).unwrap_or_default();
                    (s.calls, s.hard_junk_rate(), s.soft_junk_rate())
                },
            );
            for (arm, _) in picks {
                if chosen.len() < k {
                    chosen.push(arm);
                }
            }
            // Triage cells for deeper investigation.
            if let Some(triage) = &self.triage {
                let bins = triage.tracker().active_bins();
                if !bins.is_empty() {
                    triage_cells = triage
                        .top_alarmed_cells(&bins, triage_k)
                        .into_iter()
                        .map(|(cell, _explore)| cell)
                        .collect();
                }
            }
        }

        let mab_k = k.saturating_sub(chosen.len());
        let mab_arms: Vec<String> = remaining_arms
            .iter()
            .filter(|a| !chosen.contains(*a))
            .cloned()
            .collect();

        if mab_k == 0 || mab_arms.is_empty() {
            return RouterDecision {
                chosen,
                mode,
                prechosen: Vec::new(),
                control_picks,
                mab_eligible: Vec::new(),
                triage_cells,
            };
        }

        // -- Step 3: Build observation snapshot for pipeline --
        let obs_snap: BTreeMap<String, (u64, u64)> = mab_arms
            .iter()
            .map(|a| {
                let w = self.windows.get(a.as_str());
                let calls = w.map(|w| w.len() as u64).unwrap_or(0);
                let elapsed = w.map(|w| w.summary().elapsed_ms_sum).unwrap_or(0);
                (a.clone(), (calls, elapsed))
            })
            .collect();

        let sum_snap: BTreeMap<String, Summary> = mab_arms
            .iter()
            .map(|a| {
                let s = self.windows.get(a.as_str()).map(|w| w.summary()).unwrap_or_default();
                (a.clone(), s)
            })
            .collect();

        // -- Step 4: Policy plan (novelty + coverage + guardrail) --
        let plan = policy_fill_generic(
            seed ^ 0x504C_414E,
            &mab_arms,
            mab_k,
            self.cfg.novelty_enabled,
            self.cfg.coverage,
            self.cfg.guardrail,
            self.cfg.pipeline_order,
            |arm| obs_snap.get(arm).copied().unwrap_or((0, 0)),
            |eligible, need| {
                // MAB selection over eligible arms.
                self.select_mab_round(eligible, need, &sum_snap, seed ^ 0x4D41_4200)
            },
        );

        let prechosen = plan.plan.prechosen.clone();
        let mab_eligible = plan.plan.eligible.clone();

        for arm in &plan.chosen {
            if chosen.len() < k {
                chosen.push(arm.clone());
            }
        }

        RouterDecision {
            chosen,
            mode,
            prechosen,
            control_picks,
            mab_eligible,
            triage_cells,
        }
    }

    /// Record an outcome for an arm (no feature context).
    ///
    /// Updates all windows and the triage detector.
    pub fn observe(&mut self, arm: &str, outcome: Outcome) {
        self.observe_with_context(arm, outcome, &[]);
    }

    /// Record an outcome with a feature context vector.
    ///
    /// The context is used for per-cell triage if [`RouterConfig::triage_cfg`]
    /// is set.  Pass `&[]` if you have no features.
    pub fn observe_with_context(&mut self, arm: &str, outcome: Outcome, context: &[f64]) {
        if let Some(w) = self.windows.get_mut(arm) {
            w.push(outcome);
        }
        if let Some(ref mut m) = self.monitored {
            if let Some(mw) = m.get_mut(arm) {
                mw.push(outcome);
            }
        }
        if let Some(ref mut t) = self.triage {
            let idx = OutcomeIdx::from_outcome(outcome.ok, outcome.junk, outcome.hard_junk);
            t.observe(arm, idx, context);
        }
        self.total_observations += 1;
    }

    /// Update the most recent outcome's junk label (delayed quality assessment).
    ///
    /// Mirrors [`Window::set_last_junk_level`] across all maintained windows.
    pub fn set_last_junk_level(&mut self, arm: &str, junk: bool, hard_junk: bool) {
        if let Some(w) = self.windows.get_mut(arm) {
            w.set_last_junk_level(junk, hard_junk);
        }
        if let Some(ref mut m) = self.monitored {
            if let Some(mw) = m.get_mut(arm) {
                mw.set_last_junk_level(junk, hard_junk);
            }
        }
    }

    /// Acknowledge a confirmed regression on `arm`.
    ///
    /// This performs the full post-detection protocol:
    /// 1. Resets the arm's CUSUM bank (so it can detect the next change).
    /// 2. Promotes the arm's recent window into its baseline (so the new
    ///    post-change distribution becomes the reference for future monitoring).
    ///
    /// Call this after you've investigated the regression and confirmed the arm
    /// has stabilized at a new operating point.
    pub fn acknowledge_change(&mut self, arm: &str) {
        if let Some(ref mut t) = self.triage {
            t.reset_arm(arm);
        }
        if let Some(ref mut m) = self.monitored {
            if let Some(mw) = m.get_mut(arm) {
                mw.acknowledge_change();
            }
        }
    }

    /// Acknowledge all alarmed arms (convenience wrapper).
    pub fn acknowledge_all_changes(&mut self) {
        let alarmed = self.mode().alarmed_arms().to_vec();
        for arm in &alarmed {
            self.acknowledge_change(arm);
        }
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Current routing mode.
    pub fn mode(&self) -> RouterMode {
        let alarmed = self
            .triage
            .as_ref()
            .map(|t| t.alarmed_arms())
            .unwrap_or_default();
        if alarmed.is_empty() {
            RouterMode::Normal
        } else {
            RouterMode::Triage { alarmed_arms: alarmed }
        }
    }

    /// Arms registered with this router.
    pub fn arms(&self) -> &[String] {
        &self.arms
    }

    /// Total observations recorded via [`observe`].
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Current `Summary` for an arm (from the selection window).
    pub fn summary(&self, arm: &str) -> Summary {
        self.windows.get(arm).map(|w| w.summary()).unwrap_or_default()
    }

    /// Read-only access to the selection window for an arm.
    pub fn window(&self, arm: &str) -> Option<&Window> {
        self.windows.get(arm)
    }

    /// Read-only access to the monitored window for an arm (if monitoring enabled).
    pub fn monitored_window(&self, arm: &str) -> Option<&MonitoredWindow> {
        self.monitored.as_ref()?.get(arm)
    }

    /// Triage session (if enabled).
    pub fn triage_session(&self) -> Option<&TriageSession> {
        self.triage.as_ref()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Run one round of MAB selection over `eligible` arms, returning up to `need` picks.
    fn select_mab_round(
        &self,
        eligible: &[String],
        need: usize,
        sum_snap: &BTreeMap<String, Summary>,
        seed: u64,
    ) -> Vec<String> {
        if eligible.is_empty() || need == 0 {
            return Vec::new();
        }
        let monitored = self.monitored.as_ref();
        let mut remaining: Vec<String> = eligible.to_vec();
        let mut picks: Vec<String> = Vec::new();

        for round in 0..need {
            if remaining.is_empty() {
                break;
            }
            let round_seed = seed ^ (round as u64).wrapping_mul(0x9E37_79B9);

            // Pass the full monitored map directly — select_mab_monitored_explain_with_summaries
            // iterates over `remaining` and ignores arms not present in the map.
            // This avoids the O(K × window_cap) clone that a sub-map would incur.
            let d = if let Some(mon) = monitored {
                select_mab_monitored_explain_with_summaries(
                    &remaining,
                    sum_snap,
                    mon,
                    self.cfg.drift,
                    self.cfg.mab,
                )
            } else {
                select_mab_explain(&remaining, sum_snap, self.cfg.mab)
            };

            let _ = round_seed; // suppress unused warning
            let pick = d.selection.chosen.clone();
            remaining.retain(|a| a != &pick);
            picks.push(pick);
        }
        picks
    }
}

// ============================================================================
// Snapshot / persistence
// ============================================================================

/// A serializable snapshot of [`Router`] state.
///
/// The snapshot captures all per-arm window data and config, but **not** the
/// live triage CUSUM scores (which are rebuilt from config on restore).
/// This is intentional: CUSUM scores are noise-sensitive and should start
/// fresh after a process restart to avoid false alarms from stale history.
///
/// # Persistence pattern
///
/// ```rust,no_run
/// # #[cfg(feature = "serde")]
/// # {
/// use muxer::{Router, RouterConfig};
///
/// let arms = vec!["a".to_string()];
/// let mut router = Router::new(arms, RouterConfig::default()).unwrap();
/// // ... run routing ...
///
/// // Save (requires serde feature):
/// let snap = router.snapshot();
/// let json = serde_json::to_string(&snap).unwrap();
///
/// // Restore:
/// let snap2: muxer::RouterSnapshot = serde_json::from_str(&json).unwrap();
/// let router2 = Router::from_snapshot(snap2).unwrap();
/// # }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RouterSnapshot {
    /// Ordered arm list.
    pub arms: Vec<String>,
    /// Per-arm selection windows.
    pub windows: BTreeMap<String, Window>,
    /// Per-arm monitored windows (if monitoring was enabled).
    pub monitored: Option<BTreeMap<String, MonitoredWindow>>,
    /// Router configuration.
    pub cfg: RouterConfig,
    /// Total observations recorded before the snapshot.
    pub total_observations: u64,
}

impl Router {
    /// Capture a serializable snapshot of the current state.
    ///
    /// The snapshot excludes live CUSUM scores; they are reset on restore.
    pub fn snapshot(&self) -> RouterSnapshot {
        RouterSnapshot {
            arms: self.arms.clone(),
            windows: self.windows.clone(),
            monitored: self.monitored.clone(),
            cfg: self.cfg.clone(),
            total_observations: self.total_observations,
        }
    }

    /// Restore a [`Router`] from a snapshot.
    ///
    /// All window state is restored; CUSUM banks are rebuilt fresh from the
    /// config (detection history is reset).
    pub fn from_snapshot(snap: RouterSnapshot) -> Result<Self, logp::Error> {
        let triage = if let Some(ref tcfg) = snap.cfg.triage_cfg {
            Some(TriageSession::new(&snap.arms, tcfg.clone())?)
        } else {
            None
        };
        Ok(Self {
            arms: snap.arms,
            windows: snap.windows,
            monitored: snap.monitored,
            triage,
            cfg: snap.cfg,
            total_observations: snap.total_observations,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Outcome;

    fn arms(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("arm{i}")).collect()
    }

    fn clean() -> Outcome {
        Outcome { ok: true, junk: false, hard_junk: false, cost_units: 1, elapsed_ms: 50, quality_score: None }
    }

    fn bad() -> Outcome {
        Outcome { ok: false, junk: true, hard_junk: true, cost_units: 1, elapsed_ms: 50, quality_score: None }
    }

    // --- Basic invariants ---

    #[test]
    fn router_select_returns_member_of_arms() {
        let r = Router::new(arms(3), RouterConfig::default()).unwrap();
        let d = r.select(1, 42);
        assert_eq!(d.chosen.len(), 1);
        assert!(r.arms().contains(&d.chosen[0]));
    }

    #[test]
    fn router_select_multi_pick_unique() {
        let r = Router::new(arms(5), RouterConfig::default()).unwrap();
        let d = r.select(3, 7);
        assert!(d.chosen.len() <= 3);
        let mut s = d.chosen.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), d.chosen.len(), "picks must be unique");
    }

    #[test]
    fn router_observe_increments_total() {
        let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
        assert_eq!(r.total_observations(), 0);
        r.observe("arm0", clean());
        r.observe("arm1", clean());
        assert_eq!(r.total_observations(), 2);
    }

    #[test]
    fn router_select_never_returns_more_than_k() {
        let r = Router::new(arms(2), RouterConfig::default()).unwrap();
        let d = r.select(5, 0); // k=5 > K=2
        assert!(d.chosen.len() <= 2);
    }

    // --- Explore-first ---

    #[test]
    fn router_explores_unseen_arm_before_exploitation() {
        let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
        // Flood arm0 with good outcomes so arm1 (unseen) should be explored first.
        for _ in 0..50 {
            r.observe("arm0", clean());
        }
        let d = r.select(1, 0);
        assert_eq!(d.chosen[0], "arm1", "unseen arm should be explored first");
    }

    // --- Select quality preference ---

    #[test]
    fn router_prefers_better_arm_after_enough_data() {
        let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
        for _ in 0..50 {
            r.observe("arm0", clean());
        }
        for _ in 0..50 {
            r.observe("arm1", Outcome { ok: true, junk: true, hard_junk: false, ..clean() });
        }
        let d = r.select(1, 0);
        assert_eq!(d.chosen[0], "arm0", "arm0 has lower junk rate");
    }

    // --- Triage lifecycle ---

    #[test]
    fn router_triage_detects_hard_failure_arm() {
        let tcfg = TriageSessionConfig {
            min_n: 10,
            threshold: 3.0,
            ..TriageSessionConfig::default()
        };
        let cfg = RouterConfig::default().with_triage_cfg(tcfg);
        let mut r = Router::new(vec!["good".to_string(), "bad".to_string()], cfg).unwrap();

        // Seed baseline for both arms.
        for _ in 0..20 {
            r.observe("good", clean());
            r.observe("bad", clean());
        }
        // Inject hard failures on "bad".
        for _ in 0..30 {
            r.observe("bad", bad());
        }

        assert!(
            r.mode().is_triage(),
            "should alarm after sustained hard failures"
        );
        assert!(
            r.mode().alarmed_arms().contains(&"bad".to_string()),
            "'bad' arm should be alarmed"
        );
    }

    #[test]
    fn router_acknowledge_change_resets_triage() {
        let tcfg = TriageSessionConfig {
            min_n: 5,
            threshold: 2.0,
            ..TriageSessionConfig::default()
        };
        let cfg = RouterConfig::default().with_triage_cfg(tcfg);
        let mut r = Router::new(vec!["a".to_string()], cfg).unwrap();

        for _ in 0..10 {
            r.observe("a", clean());
        }
        for _ in 0..20 {
            r.observe("a", bad());
        }
        assert!(r.mode().is_triage());

        r.acknowledge_change("a");
        assert!(!r.mode().is_triage(), "mode should return to Normal after acknowledge");
    }

    // --- Monitoring windows ---

    #[test]
    fn router_monitoring_windows_exist_when_enabled() {
        let cfg = RouterConfig::default().with_monitoring(200, 50);
        let r = Router::new(arms(3), cfg).unwrap();
        for a in r.arms() {
            assert!(r.monitored_window(a).is_some(), "monitored window should exist for {a}");
        }
    }

    #[test]
    fn router_acknowledge_promotes_recent_to_baseline() {
        let cfg = RouterConfig::default().with_monitoring(200, 50);
        let mut r = Router::new(vec!["a".to_string()], cfg).unwrap();
        for _ in 0..20 {
            r.observe("a", clean());
        }
        let before_recent = r.monitored_window("a").unwrap().recent_len();
        assert!(before_recent > 0);
        r.acknowledge_change("a");
        let after_recent = r.monitored_window("a").unwrap().recent_len();
        assert_eq!(after_recent, 0, "recent window should be cleared after acknowledge");
    }

    // --- Dynamic arm management ---

    #[test]
    fn router_add_arm_is_explored_next() {
        let mut r = Router::new(arms(2), RouterConfig::default()).unwrap();
        for _ in 0..50 {
            r.observe("arm0", clean());
            r.observe("arm1", clean());
        }
        r.add_arm("arm2".to_string()).unwrap();

        let d = r.select(1, 0);
        assert_eq!(d.chosen[0], "arm2", "newly added arm should be explored first");
    }

    #[test]
    fn router_remove_arm_not_selected() {
        let mut r = Router::new(arms(3), RouterConfig::default()).unwrap();
        r.remove_arm("arm1");
        for _ in 0..100 {
            let d = r.select(1, 0);
            assert_ne!(d.chosen[0], "arm1", "removed arm must not be selected");
        }
    }

    // --- Large K ---

    #[test]
    fn router_large_k_covers_all_arms_with_multi_pick() {
        let n = 30;
        let cfg = RouterConfig::default();
        let mut r = Router::new(arms(n), cfg).unwrap();

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        // With k=3 and K=30, expect initial coverage in ~10 rounds.
        for round in 0..15 {
            let d = r.select(3, round as u64);
            for a in &d.chosen {
                seen.insert(a.clone());
            }
            for a in &d.chosen {
                r.observe(a, clean());
            }
        }
        assert_eq!(seen.len(), n, "all {n} arms should be explored within 15 rounds (k=3)");
    }

    #[test]
    fn router_large_k_with_coverage_prevents_starvation() {
        let n = 20;
        let cfg = RouterConfig::default().with_coverage(0.02, 1);
        let mut r = Router::new(arms(n), cfg).unwrap();

        // Run 200 rounds with k=1.
        for i in 0..200 {
            let d = r.select(1, i as u64);
            if let Some(arm) = d.primary() {
                r.observe(arm, clean());
            }
        }

        // With coverage floor 2%, each arm should have been called at least a few times.
        for a in r.arms() {
            let s = r.summary(a);
            assert!(
                s.calls > 0,
                "arm {a} should have at least 1 observation with coverage enabled"
            );
        }
    }

    // --- Control picks ---

    #[test]
    fn router_control_picks_are_subset_of_chosen() {
        let cfg = RouterConfig::default().with_control(1);
        let r = Router::new(arms(5), cfg).unwrap();
        let d = r.select(3, 42);
        for p in &d.control_picks {
            assert!(d.chosen.contains(p), "control pick {p} must be in chosen");
        }
    }

    // --- Determinism ---

    #[test]
    fn router_select_is_deterministic() {
        let mut r = Router::new(arms(4), RouterConfig::default()).unwrap();
        for _ in 0..20 {
            r.observe("arm0", clean());
            r.observe("arm1", bad());
        }
        let d1 = r.select(2, 99);
        let d2 = r.select(2, 99);
        assert_eq!(d1.chosen, d2.chosen, "same seed → same picks");
    }
}
