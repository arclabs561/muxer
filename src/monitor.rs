//! Monitoring primitives: drift detection + uncertainty summaries.
//!
//! This module implements **Objective 3 (detection/triage)** of the routing framework.
//! It provides numerical signals (drift scores, catKL, CUSUM) that selection policies
//! can use as constraints or additional objectives.
//!
//! ## Relationship to the objective manifold
//!
//! For **static (non-adaptive) schedules**, **average** detection delay D_avg and
//! estimation MSE are exactly proportional:
//!
//! ```text
//!   D_avg = (2 * C * sigma^2 * ln(1/alpha) / delta^2) * MSE
//! ```
//!
//! Both functionals have sensitivity `s(a,x) ~ -1/p_a(x)^2` with respect to the
//! design measure -- they both care about "how many observations at this cell" and
//! are indifferent to *which* cells those observations come from.  Their gradients
//! are proportional everywhere, so they trace the same direction in design space.
//! Detection monitoring "comes for free" with any exploration schedule that reduces
//! estimation error.  The three named objectives span a rank-2 sensitivity space;
//! the Pareto front is a one-dimensional curve.
//!
//! **This proportionality holds even in contextual settings** when detection is
//! measured as an average over cells.  The collapse breaks only when detection is
//! measured as a **worst-case** (minimax) over cells: worst-case delay concentrates
//! sensitivity on the bottleneck (arm, cell) pair, producing a point-mass sensitivity
//! function that is linearly independent from both regret and estimation sensitivity.
//!
//! The practical consequence for `muxer`: since monitoring is currently per-arm
//! (not per-arm-per-cell), the detectors (catKL, CUSUM, drift) operate in the
//! average-case regime and are structurally coupled with estimation quality.
//! To get genuinely independent monitoring in the contextual regime, one would need
//! per-cell monitoring that flags the weakest link, not just aggregate per-arm scores.
//!
//! **For adaptive policies** (which `muxer`'s selection functions produce), the clean
//! proportionality is approximate, not exact.  Adaptive policies allocate the
//! suboptimal arm in bursts (exploration phases) rather than uniformly; this temporal
//! clustering worsens worst-case detection delay without changing total n.  The
//! practical consequence: `muxer`'s CUSUM and catKL detectors are sensitive to
//! observation ordering, not just counts.  This is why `MonitoredWindow` maintains
//! a time-ordered sequence (not just aggregate counts) and why `CusumCatDetector`
//! processes observations sequentially rather than from summary statistics.
//!
//! ## CUSUM detection delay and sensitivity
//!
//! For a CUSUM monitoring arm `a` at covariate cell `j` with allocation probability
//! `p_a(x_j)`, the expected detection delay for a level shift of magnitude `delta`
//! with false alarm rate `alpha` is:
//!
//! ```text
//!   D_{a,j} ~ ln(1/alpha) / ((T/M) * p_a(x_j) * delta^2 / (2*sigma^2))
//! ```
//!
//! The sensitivity with respect to the design variable `p_a(x_j)` is:
//!
//! ```text
//!   s_D(a,j) = -ln(1/alpha) / ((T/M) * p_a(x_j)^2 * delta^2 / (2*sigma^2))
//! ```
//!
//! - **Average delay** `D_avg = (1/KM) * sum D_{a,j}`: sensitivity `~ 1/p^2`
//!   everywhere, proportional to nonparametric IMSE.  Redundant with estimation.
//! - **Worst-case delay** `D_max = max D_{a,j}`: sensitivity is a point mass at the
//!   binding constraint `(a*, j*) = argmax D_{a,j}`.  Independent from estimation.
//!
//! This is the precise mechanism controlling the collapse-vs-revival transition.
//!
//! ## Connection to profile monitoring / SPC
//!
//! Detecting that an arm's response function f_a has changed shape (not just level)
//! is essentially the **profile monitoring** problem from statistical process control.
//! SPC practitioners have long understood that Phase I (model estimation) and Phase II
//! (monitoring) are coupled: poor Phase I estimation inflates Phase II false alarm
//! rates.  The `muxer` monitoring primitives (catKL, CUSUM) implement Phase II;
//! the quality of Phase I depends on the exploration budget the selection policy
//! allocates.  This coupling is what the objective manifold framework formalizes.
//!
//! In the **contextual** regime, detection requires sampling where the regret-optimal
//! policy does not go (uniformly across covariate space, rather than near decision
//! boundaries).  Monitoring has a real cost that the routing policy must explicitly
//! budget for.
//!
//! ## Post-detection investigation
//!
//! The `worst_first` module is the **triage/investigation** complement to detection:
//! after this module flags a change (high drift score), worst-first prioritizes the
//! flagged arm to characterize the change.  The real open problem is not "how to
//! switch modes" but **when a detected change invalidates the current objective
//! weighting** -- a meta-level inference problem about whether the system's goals
//! need to shift (e.g., from exploitation to characterization).

use crate::{Outcome, Summary, Window};

/// Categorical drift metric used for comparing two outcome distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DriftMetric {
    /// Fisher–Rao (Rao) distance on the simplex (radians, in `[0, π]`).
    #[default]
    Rao,
    /// Jensen–Shannon divergence (nats, in `[0, ln 2]`).
    JensenShannon,
    /// Hellinger distance (in `[0, 1]`).
    Hellinger,
}

/// How to adjust an observed Bernoulli rate using a confidence bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RateBoundMode {
    /// Do not adjust.
    #[default]
    None,
    /// Use the upper confidence bound (conservative).
    Upper,
    /// Use the lower confidence bound (optimistic).
    Lower,
}

/// Configuration for computing Wilson score bounds for rates.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UncertaintyConfig {
    /// Z-score used for Wilson bounds.
    ///
    /// - `1.96` ≈ 95% (two-sided)
    /// - `2.58` ≈ 99% (two-sided)
    pub z: f64,
    /// Apply a Wilson bound to `ok_rate`.
    pub ok_mode: RateBoundMode,
    /// Apply a Wilson bound to `junk_rate`.
    pub junk_mode: RateBoundMode,
    /// Apply a Wilson bound to `hard_junk_rate`.
    pub hard_junk_mode: RateBoundMode,
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            z: 1.96,
            ok_mode: RateBoundMode::None,
            // Default: conservative for junk metrics (avoid small-n optimism).
            junk_mode: RateBoundMode::Upper,
            hard_junk_mode: RateBoundMode::Upper,
        }
    }
}

/// Drift computation configuration.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DriftConfig {
    pub metric: DriftMetric,
    /// Simplex validation tolerance.
    pub tol: f64,
    /// Minimum baseline samples required before reporting a drift score.
    pub min_baseline: u64,
    /// Minimum recent samples required before reporting a drift score.
    pub min_recent: u64,
}

impl Default for DriftConfig {
    fn default() -> Self {
        Self {
            metric: DriftMetric::default(),
            tol: 1e-12,
            min_baseline: 20,
            min_recent: 10,
        }
    }
}

/// A baseline/recent pair of windows for drift monitoring.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MonitoredWindow {
    baseline: Window,
    recent: Window,
}

impl MonitoredWindow {
    /// Create a new monitored window with baseline and recent capacities.
    pub fn new(baseline_cap: usize, recent_cap: usize) -> Self {
        Self {
            baseline: Window::new(baseline_cap),
            recent: Window::new(recent_cap),
        }
    }

    /// Access the baseline window.
    pub fn baseline(&self) -> &Window {
        &self.baseline
    }

    /// Access the recent window.
    pub fn recent(&self) -> &Window {
        &self.recent
    }

    /// Push a new outcome to both baseline and recent windows.
    pub fn push(&mut self, o: Outcome) {
        self.baseline.push(o);
        self.recent.push(o);
    }

    /// Best-effort: mutate the most recent outcome in both windows.
    pub fn set_last_junk(&mut self, junk: bool) {
        self.baseline.set_last_junk(junk);
        self.recent.set_last_junk(junk);
    }

    /// Best-effort: mutate the most recent outcome in both windows.
    pub fn set_last_junk_level(&mut self, junk: bool, hard_junk: bool) {
        self.baseline.set_last_junk_level(junk, hard_junk);
        self.recent.set_last_junk_level(junk, hard_junk);
    }

    /// Best-effort: set the continuous quality score for the most recent outcome in both windows.
    pub fn set_last_quality_score(&mut self, score: f64) {
        self.baseline.set_last_quality_score(score);
        self.recent.set_last_quality_score(score);
    }

    /// Summary of the recent window (what selection should use by default).
    pub fn recent_summary(&self) -> Summary {
        self.recent.summary()
    }

    /// Number of outcomes currently in the baseline window.
    pub fn baseline_len(&self) -> usize {
        self.baseline.len()
    }

    /// Number of outcomes currently in the recent window.
    pub fn recent_len(&self) -> usize {
        self.recent.len()
    }

    /// Acknowledge a confirmed change: promote recent observations into the baseline,
    /// then clear the recent window.
    ///
    /// Call this after investigating and confirming a regression on this arm.
    /// The recent window becomes part of the new baseline; subsequent observations
    /// build a fresh recent window so drift detection can track further changes.
    ///
    /// This is the standard post-detection protocol:
    /// 1. [`crate::TriageSession::reset_arm`] — resets the CUSUM bank.
    /// 2. This method — resets the monitoring windows.
    pub fn acknowledge_change(&mut self) {
        for o in self.recent.iter().copied() {
            self.baseline.push(o);
        }
        self.recent = Window::new(self.recent.cap());
    }

    /// Merge recent observations into the baseline without clearing the recent window.
    ///
    /// Use when you want the baseline to absorb new data while keeping recent
    /// observations for ongoing drift comparison (e.g., a soft acknowledgment
    /// that does not restart the drift sensor).
    pub fn promote_recent_to_baseline(&mut self) {
        for o in self.recent.iter().copied() {
            self.baseline.push(o);
        }
    }
}

/// Drift computation output.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DriftDecision {
    /// Drift score (meaning depends on `metric`).
    pub score: f64,
    pub metric: DriftMetric,
    /// Baseline sample count.
    pub baseline_n: u64,
    /// Recent sample count.
    pub recent_n: u64,
    /// Categorical baseline distribution used for drift calculation.
    pub baseline_p: Vec<f64>,
    /// Categorical recent distribution used for drift calculation.
    pub recent_p: Vec<f64>,
}

/// Compute drift between two simplex distributions directly (domain-agnostic).
///
/// This is useful for downstream domains that already have categorical probabilities
/// (or any other simplex-valued representation) without going through `muxer::Outcome`.
pub fn drift_simplex(
    p: &[f64],
    q: &[f64],
    metric: DriftMetric,
    tol: f64,
) -> Result<f64, logp::Error> {
    match metric {
        DriftMetric::Rao => rao_distance_categorical(p, q, tol),
        DriftMetric::JensenShannon => logp::jensen_shannon_divergence(p, q, tol),
        DriftMetric::Hellinger => hellinger_categorical(p, q, tol),
    }
}

fn bhattacharyya_coefficient(p: &[f64], q: &[f64], tol: f64) -> Result<f64, logp::Error> {
    if p.len() != q.len() {
        return Err(logp::Error::LengthMismatch(p.len(), q.len()));
    }
    logp::validate_simplex(p, tol)?;
    logp::validate_simplex(q, tol)?;

    let mut bc = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        bc += (pi * qi).sqrt();
    }

    // Clamp to guard against tiny floating error (e.g. 1 + 1e-16).
    Ok(bc.clamp(0.0, 1.0))
}

fn hellinger_categorical(p: &[f64], q: &[f64], tol: f64) -> Result<f64, logp::Error> {
    // Standard Hellinger distance: H(p,q) ∈ [0,1].
    let bc = bhattacharyya_coefficient(p, q, tol)?;
    Ok((1.0 - bc).max(0.0).sqrt())
}

fn rao_distance_categorical(p: &[f64], q: &[f64], tol: f64) -> Result<f64, logp::Error> {
    // Fisher–Rao distance on the categorical simplex: d(p,q) = 2 arccos( Σ_i √(p_i q_i) ) ∈ [0,π].
    let bc = bhattacharyya_coefficient(p, q, tol)?;
    Ok(2.0 * bc.acos())
}

// ============================================================================
// Domain-agnostic categorical change monitoring (KL-to-baseline)
// ============================================================================

/// Online categorical change detector based on the statistic:
///
/// \( S_n = n \cdot KL(\hat q_n \| p_0)\),
///
/// where \(\hat q_n\) is the empirical categorical distribution of observed outcomes (with optional
/// Dirichlet smoothing) and \(p_0\) is a fixed baseline distribution.
///
/// This aligns with classical large-deviation heuristics: under i.i.d. baseline sampling,
/// \(S_n\) stays small most of the time; under a shift, it grows roughly linearly with \(n\).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CatKlDetector {
    k: usize,
    /// Baseline distribution \(p_0\) (simplex; must be strictly positive if `alpha==0`).
    p0: Vec<f64>,
    /// Dirichlet smoothing pseudo-count (alpha > 0 guarantees \(\hat q_n\) has full support).
    alpha: f64,
    /// Minimum number of observations before computing score.
    min_n: u64,
    /// Alarm threshold on \(S_n\).
    threshold: f64,
    /// Counts per category.
    counts: Vec<u64>,
    /// Total observations.
    n: u64,
    tol: f64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CatKlAlarm {
    pub n: u64,
    pub score: f64,
    pub threshold: f64,
}

impl CatKlDetector {
    /// Create a detector with a baseline distribution `p0` (simplex).
    ///
    /// Note: `threshold` may be `f64::INFINITY` to represent “never alarm” (useful when you only
    /// want a drift *score* but not a stop rule).
    pub fn new(
        p0: &[f64],
        alpha: f64,
        min_n: u64,
        threshold: f64,
        tol: f64,
    ) -> Result<Self, logp::Error> {
        logp::validate_simplex(p0, tol)?;
        let k = p0.len();
        if k == 0 {
            return Err(logp::Error::Empty);
        }
        let alpha = if alpha.is_finite() && alpha >= 0.0 {
            alpha
        } else {
            0.0
        };
        // Treat `+∞` as a valid “never alarm” threshold.
        let threshold = if threshold.is_nan() || threshold < 0.0 {
            0.0
        } else {
            threshold
        };

        // If alpha==0, we need p0_i>0 whenever q_i might be >0. We can't guarantee that, so we
        // enforce strictly-positive p0 when no smoothing is requested.
        if alpha == 0.0 && p0.iter().any(|&x| x <= 0.0) {
            return Err(logp::Error::Domain(
                "CatKlDetector: p0 must be strictly positive when alpha==0",
            ));
        }

        Ok(Self {
            k,
            p0: p0.to_vec(),
            alpha,
            min_n,
            threshold,
            counts: vec![0; k],
            n: 0,
            tol,
        })
    }

    /// Reset internal counts.
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.n = 0;
    }

    /// Update with a categorical observation `idx` in `[0, k)`.
    ///
    /// Returns `Some(alarm)` if the score meets/exceeds threshold and `n>=min_n`.
    pub fn update(&mut self, idx: usize) -> Option<CatKlAlarm> {
        if idx >= self.k {
            return None;
        }
        self.counts[idx] = self.counts[idx].saturating_add(1);
        self.n = self.n.saturating_add(1);
        let n = self.n;
        if n < self.min_n {
            return None;
        }
        let score = self.score()?;
        if score >= self.threshold {
            Some(CatKlAlarm {
                n,
                score,
                threshold: self.threshold,
            })
        } else {
            None
        }
    }

    /// Current total observations.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Current score \(S_n = n * KL(\hat q_n \| p_0)\).
    pub fn score(&self) -> Option<f64> {
        if self.n == 0 || self.n < self.min_n {
            return None;
        }
        let q = self.empirical();
        let kl = logp::kl_divergence(&q, &self.p0, self.tol).ok()?;
        Some((self.n as f64) * kl)
    }

    fn empirical(&self) -> Vec<f64> {
        let kf = self.k as f64;
        let alpha = self.alpha;
        let denom = (self.n as f64) + alpha * kf;
        if denom <= 0.0 {
            // Shouldn't happen, but keep it safe.
            return vec![1.0 / kf; self.k];
        }
        self.counts
            .iter()
            .map(|&c| ((c as f64) + alpha) / denom)
            .collect()
    }
}

// ============================================================================
// Domain-agnostic categorical CUSUM (log-likelihood ratio)
// ============================================================================

/// Online CUSUM change detector for categorical observations with a fixed null `p0` and alternative `p1`.
///
/// This maintains:
/// \(S_t = \max(0, S_{t-1} + \log(p_1(X_t)) - \log(p_0(X_t)))\),
/// and triggers an alarm when `S_t >= threshold` after `min_n` samples.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumCatDetector {
    k: usize,
    p0: Vec<f64>,
    p1: Vec<f64>,
    min_n: u64,
    threshold: f64,
    s: f64,
    n: u64,
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumCatAlarm {
    pub n: u64,
    pub score: f64,
    pub threshold: f64,
}

impl CusumCatDetector {
    /// Create a detector with a baseline distribution `p0` and alternative `p1`.
    ///
    /// Note: `threshold` may be `f64::INFINITY` to represent “never alarm” (useful when you only
    /// want a CUSUM *score* but not a stop rule).
    pub fn new(
        p0: &[f64],
        p1: &[f64],
        alpha: f64,
        min_n: u64,
        threshold: f64,
        tol: f64,
    ) -> Result<Self, logp::Error> {
        logp::validate_simplex(p0, tol)?;
        logp::validate_simplex(p1, tol)?;
        if p0.len() != p1.len() {
            return Err(logp::Error::LengthMismatch(p0.len(), p1.len()));
        }
        let k = p0.len();
        if k == 0 {
            return Err(logp::Error::Empty);
        }
        let alpha = if alpha.is_finite() && alpha >= 0.0 {
            alpha
        } else {
            0.0
        };
        // Treat `+∞` as a valid “never alarm” threshold.
        let threshold = if threshold.is_nan() || threshold < 0.0 {
            0.0
        } else {
            threshold
        };

        // If alpha==0, require strictly-positive probabilities so log() is safe.
        if alpha == 0.0 && (p0.iter().any(|&x| x <= 0.0) || p1.iter().any(|&x| x <= 0.0)) {
            return Err(logp::Error::Domain(
                "CusumCatDetector: p0 and p1 must be strictly positive when alpha==0",
            ));
        }

        // Apply smoothing (if any) to guarantee positivity.
        let denom = 1.0 + alpha * (k as f64);
        let p0s: Vec<f64> = p0.iter().map(|&x| (x + alpha) / denom).collect();
        let p1s: Vec<f64> = p1.iter().map(|&x| (x + alpha) / denom).collect();
        // With alpha>0, all entries are positive; with alpha==0 we already checked.
        Ok(Self {
            k,
            p0: p0s,
            p1: p1s,
            min_n,
            threshold,
            s: 0.0,
            n: 0,
        })
    }

    pub fn reset(&mut self) {
        self.s = 0.0;
        self.n = 0;
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn score(&self) -> f64 {
        self.s
    }

    /// Update with a categorical observation `idx` in `[0, k)`.
    ///
    /// Returns `Some(alarm)` if the CUSUM score meets/exceeds threshold and `n>=min_n`.
    pub fn update(&mut self, idx: usize) -> Option<CusumCatAlarm> {
        if idx >= self.k {
            return None;
        }
        let llr = self.p1[idx].ln() - self.p0[idx].ln();
        self.s = (self.s + llr).max(0.0);
        self.n = self.n.saturating_add(1);
        if self.n < self.min_n {
            return None;
        }
        if self.s >= self.threshold {
            Some(CusumCatAlarm {
                n: self.n,
                score: self.s,
                threshold: self.threshold,
            })
        } else {
            None
        }
    }
}

// ============================================================================
// GLR-lite: CUSUM banks (multiple alternatives)
// ============================================================================

/// Output of updating a [`CusumCatBank`].
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumCatBankUpdate {
    /// Whether any detector in the bank alarmed on this update.
    ///
    /// This is equivalent to “some alternative exceeded `threshold` with `n >= min_n`”.
    pub alarmed: bool,
    /// Maximum CUSUM score across alternatives after the update.
    ///
    /// Note: this score can be positive even when `n < min_n` (i.e. before the bank is eligible to alarm).
    pub score_max: f64,
    /// Number of observations processed by the bank (shared across alternatives).
    pub n: u64,
}

/// A small bank of categorical CUSUM detectors sharing a common null `p0`.
///
/// This is a “GLR-lite” robustification:
/// - maintain one [`CusumCatDetector`] per candidate alternative, and
/// - use the max score across alternatives for suspicion / alarm gating.
///
/// This is useful when the post-change distribution is unknown but you can propose a small set
/// of plausible alternatives (“an alt bank”).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CusumCatBank {
    dets: Vec<CusumCatDetector>,
    min_n: u64,
}

impl CusumCatBank {
    /// Create a CUSUM bank from a baseline distribution `p0` and a list of alternatives `alts`.
    ///
    /// Each alternative must:
    /// - be a valid simplex (within `tol`), and
    /// - have the same length as `p0`.
    ///
    /// `alts` must be non-empty.
    ///
    /// Note: `threshold` may be `f64::INFINITY` to represent “never alarm” (useful when you only
    /// want a bank *score* but not a stop rule).
    pub fn new(
        p0: &[f64],
        alts: &[Vec<f64>],
        alpha: f64,
        min_n: u64,
        threshold: f64,
        tol: f64,
    ) -> Result<Self, logp::Error> {
        if alts.is_empty() {
            return Err(logp::Error::Domain("CusumCatBank: alts must be non-empty"));
        }
        let dets: Vec<CusumCatDetector> = alts
            .iter()
            .map(|alt| CusumCatDetector::new(p0, alt, alpha, min_n, threshold, tol))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { dets, min_n })
    }

    /// Number of alternatives in the bank.
    pub fn len(&self) -> usize {
        self.dets.len()
    }

    /// Whether the bank contains zero alternatives.
    pub fn is_empty(&self) -> bool {
        self.dets.is_empty()
    }

    /// Minimum observations required before the bank can alarm.
    pub fn min_n(&self) -> u64 {
        self.min_n
    }

    /// Number of observations processed by the bank.
    pub fn n(&self) -> u64 {
        self.dets.first().map(|d| d.n()).unwrap_or(0)
    }

    /// Maximum CUSUM score across alternatives.
    pub fn score_max(&self) -> f64 {
        self.dets.iter().map(|d| d.score()).fold(0.0_f64, f64::max)
    }

    /// Reset all alternatives to score=0 and n=0.
    pub fn reset(&mut self) {
        for d in &mut self.dets {
            d.reset();
        }
    }

    /// Update all alternatives with the categorical observation `idx`.
    ///
    /// Returns whether any alternative alarmed, plus the max score and shared `n`.
    pub fn update(&mut self, idx: usize) -> CusumCatBankUpdate {
        let mut alarmed = false;
        let mut score_max = 0.0_f64;
        let mut n = 0_u64;
        for (i, d) in self.dets.iter_mut().enumerate() {
            alarmed |= d.update(idx).is_some();
            score_max = score_max.max(d.score());
            if i == 0 {
                n = d.n();
            }
        }
        CusumCatBankUpdate {
            alarmed,
            score_max,
            n,
        }
    }
}

/// Compute drift between two windows (returns `None` if either side is under-sampled).
pub fn drift_between_windows(
    baseline: &Window,
    recent: &Window,
    cfg: DriftConfig,
) -> Option<DriftDecision> {
    let bn = baseline.len() as u64;
    let rn = recent.len() as u64;
    if bn < cfg.min_baseline || rn < cfg.min_recent {
        return None;
    }

    let p = categorical_probs(baseline);
    let q = categorical_probs(recent);

    let score = match cfg.metric {
        DriftMetric::Rao => rao_distance_categorical(&p, &q, cfg.tol).ok()?,
        DriftMetric::JensenShannon => logp::jensen_shannon_divergence(&p, &q, cfg.tol).ok()?,
        DriftMetric::Hellinger => hellinger_categorical(&p, &q, cfg.tol).ok()?,
    };

    Some(DriftDecision {
        score,
        metric: cfg.metric,
        baseline_n: bn,
        recent_n: rn,
        baseline_p: p,
        recent_p: q,
    })
}

/// A small categorical representation of outcomes, intended for drift monitoring.
///
/// Categories (mutually exclusive by construction):
/// - ok & not junk
/// - ok & soft junk
/// - ok & hard junk
/// - fail (not ok)
fn categorical_counts(window: &Window) -> [u64; 4] {
    let mut ok_clean = 0u64;
    let mut soft = 0u64;
    let mut hard = 0u64;
    let mut fail = 0u64;

    for o in window.iter() {
        if !o.ok {
            fail += 1;
            continue;
        }
        if o.hard_junk {
            hard += 1;
        } else if o.junk {
            soft += 1;
        } else {
            ok_clean += 1;
        }
    }

    [ok_clean, soft, hard, fail]
}

fn categorical_index(o: &Outcome) -> usize {
    // Categories (mutually exclusive by construction):
    // - ok & not junk
    // - ok & soft junk
    // - ok & hard junk
    // - fail (not ok)
    if !o.ok {
        return 3;
    }
    if o.hard_junk {
        2
    } else if o.junk {
        1
    } else {
        0
    }
}

fn categorical_probs(window: &Window) -> Vec<f64> {
    let c = categorical_counts(window);
    let n = c.iter().copied().sum::<u64>().max(1) as f64;
    c.iter().map(|&x| (x as f64) / n).collect()
}

fn smooth_simplex_from_counts(counts: &[u64], alpha: f64) -> Vec<f64> {
    let k = counts.len().max(1) as f64;
    let alpha = if alpha.is_finite() && alpha > 0.0 {
        alpha
    } else {
        0.0
    };
    let n = counts.iter().copied().sum::<u64>() as f64;
    let denom = n + alpha * k;
    if denom <= 0.0 {
        return vec![1.0 / k; counts.len().max(1)];
    }
    counts
        .iter()
        .map(|&c| ((c as f64) + alpha) / denom)
        .collect()
}

/// Categorical KL change score between baseline and recent windows:
///
/// \(S = n_{recent} \cdot KL(q_{recent} \| p_{baseline})\),
/// where `p_baseline` and `q_recent` are derived from the window’s categorical counts with optional
/// Dirichlet smoothing (`alpha`).
pub fn catkl_score_between_windows(
    baseline: &Window,
    recent: &Window,
    alpha: f64,
    tol: f64,
    min_baseline: u64,
    min_recent: u64,
) -> Option<f64> {
    let bn = baseline.len() as u64;
    let rn = recent.len() as u64;
    if bn < min_baseline || rn < min_recent || rn == 0 {
        return None;
    }
    let b = categorical_counts(baseline);
    let r = categorical_counts(recent);
    let p0 = smooth_simplex_from_counts(&b, alpha);
    let q = smooth_simplex_from_counts(&r, alpha);
    let kl = logp::kl_divergence(&q, &p0, tol).ok()?;
    Some((rn as f64) * kl)
}

fn default_cusum_alt_p() -> [f64; 4] {
    // Conservative alternative that emphasizes "something is wrong":
    // shift probability mass towards hard_junk/fail categories.
    [0.05, 0.05, 0.45, 0.45]
}

/// Categorical CUSUM score over the recent window using:
/// - `p0`: baseline empirical distribution (with optional smoothing)
/// - `p1`: alternative distribution (fixed, with optional smoothing)
///
/// ## Detection sensitivity (concrete)
///
/// For this categorical CUSUM, the expected detection delay given a shift from
/// p0 to the true post-change distribution q is approximately:
///
/// ```text
/// E[delay] ~ threshold / KL(q || p0)
/// ```
///
/// where threshold is the alarm boundary (here: infinity, so we report the score
/// rather than alarming) and KL is the per-observation log-likelihood ratio.
///
/// The sensitivity function (how detection delay changes with sampling at a
/// context point x in the contextual regime) is:
///
/// ```text
/// s_detection(a, x) = -1 / (n_recent(a, x) * KL(q_a(x) || p0_a(x)))
/// ```
///
/// This is generically different from `s_regret(a, x) = gap_a(x)` and
/// `s_estimation(a, x) = -leverage_a(x)`, because KL depends on the distribution
/// shape rather than the mean.  In the non-contextual case, `n_recent(a, x) = n_a`
/// for all x, so `s_detection` is proportional to `s_estimation` (both ~ 1/n_a).
/// In the contextual case, they diverge.
///
/// Returns `None` if either side is under-sampled.
pub fn cusum_score_between_windows(
    baseline: &Window,
    recent: &Window,
    alpha: f64,
    tol: f64,
    min_baseline: u64,
    min_recent: u64,
    alt_p: Option<[f64; 4]>,
) -> Option<f64> {
    let bn = baseline.len() as u64;
    let rn = recent.len() as u64;
    if bn < min_baseline || rn < min_recent {
        return None;
    }

    let b = categorical_counts(baseline);
    let p0 = smooth_simplex_from_counts(&b, alpha);

    let p1_raw = alt_p.unwrap_or_else(default_cusum_alt_p);
    let p1_vec = p1_raw.to_vec();

    // Validate / renormalize p1 via simplex validation + smoothing inside detector.
    logp::validate_simplex(&p1_vec, tol).ok()?;

    let mut d = CusumCatDetector::new(&p0, &p1_vec, alpha, 1, f64::INFINITY, tol).ok()?;
    for o in recent.iter() {
        let idx = categorical_index(o);
        let _ = d.update(idx);
    }
    Some(d.score())
}

/// Wilson score interval for a Bernoulli proportion.
///
/// Returns `(lower, upper, half_width)`, with bounds clamped into `[0,1]`.
pub fn wilson_bounds(successes: u64, trials: u64, z: f64) -> (f64, f64, f64) {
    if trials == 0 {
        return (0.0, 1.0, 0.5);
    }
    let n = trials as f64;
    let k = successes.min(trials) as f64;
    let p_hat = k / n;
    let z = if z.is_finite() && z > 0.0 { z } else { 1.96 };
    let z2 = z * z;

    // Wilson interval:
    // center = (p + z^2/(2n)) / (1 + z^2/n)
    // radius = z * sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let rad = (z * ((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n))).sqrt()) / denom;
    let lo = (center - rad).clamp(0.0, 1.0);
    let hi = (center + rad).clamp(0.0, 1.0);
    ((lo), (hi), (hi - lo) / 2.0)
}

/// Result of calibrating a score threshold against a max-score distribution.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThresholdCalibration {
    /// Chosen threshold (from the provided grid).
    pub threshold: f64,
    /// Empirical false-alarm estimate: \(\hat p = \#\{M \ge h\}/n\).
    pub fa_hat: f64,
    /// Wilson upper bound for the false-alarm probability at the same threshold.
    pub fa_wilson_hi: f64,
    /// Number of Monte Carlo trials used.
    pub trials: u64,
    /// Whether some grid point satisfied the requested constraint.
    pub grid_satisfied: bool,
}

/// Calibrate a threshold \(h\) from a list of null max-scores \(M\).
///
/// Given:
/// - a sorted (ascending) grid of thresholds, and
/// - `max_scores[i] = max_t score_i(t)` for independent null trials,
///
/// this picks the *smallest* grid threshold satisfying:
/// - empirical mode: \(\hat P[M \ge h] \le \alpha\), or
/// - Wilson-conservative mode (when `require_wilson=true`): `wilson_hi <= alpha`.
///
/// Notes:
/// - `max_scores` is sorted in place (using a total order).
/// - If the grid is empty, the returned threshold is `0.0` and `grid_satisfied=false`.
/// - The grid is assumed to be sorted in ascending order; the function returns the first grid point
///   that satisfies the requested constraint.
#[must_use]
pub fn calibrate_threshold_from_max_scores(
    max_scores: &mut [f64],
    grid: &[f64],
    alpha: f64,
    z: f64,
    require_wilson: bool,
) -> ThresholdCalibration {
    let trials = max_scores.len() as u64;

    if grid.is_empty() {
        return ThresholdCalibration {
            threshold: 0.0,
            fa_hat: 1.0,
            fa_wilson_hi: 1.0,
            trials,
            grid_satisfied: false,
        };
    }
    if trials == 0 {
        return ThresholdCalibration {
            threshold: *grid.last().unwrap_or(&0.0),
            fa_hat: 1.0,
            fa_wilson_hi: 1.0,
            trials,
            grid_satisfied: false,
        };
    }

    debug_assert!(
        grid.windows(2).all(|w| w[0] <= w[1]),
        "calibrate_threshold_from_max_scores: grid must be nondecreasing"
    );

    let alpha = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let z = if z.is_finite() && z > 0.0 { z } else { 1.96 };

    max_scores.sort_by(|a, b| a.total_cmp(b));

    let n = max_scores.len() as f64;
    let mut found = false;
    let mut best = *grid.last().unwrap();
    let mut best_fa = 1.0;
    let mut best_hi = 1.0;

    for &thr in grid {
        let idx = max_scores.partition_point(|&x| x < thr);
        let fa_count = (max_scores.len() - idx) as u64;
        let fa = (fa_count as f64) / n;
        let (_lo, hi, _half) = wilson_bounds(fa_count, trials, z);
        let ok = if require_wilson {
            hi <= alpha
        } else {
            fa <= alpha
        };
        if ok {
            found = true;
            best = thr;
            best_fa = fa;
            best_hi = hi;
            break;
        }
    }

    if !found {
        let thr = *grid.last().unwrap();
        let idx = max_scores.partition_point(|&x| x < thr);
        let fa_count = (max_scores.len() - idx) as u64;
        let fa = (fa_count as f64) / n;
        let (_lo, hi, _half) = wilson_bounds(fa_count, trials, z);
        best = thr;
        best_fa = fa;
        best_hi = hi;
    }

    ThresholdCalibration {
        threshold: best,
        fa_hat: best_fa,
        fa_wilson_hi: best_hi,
        trials,
        grid_satisfied: found,
    }
}

/// Adjust a raw rate using a Wilson bound mode.
pub fn apply_rate_bound(successes: u64, trials: u64, z: f64, mode: RateBoundMode) -> (f64, f64) {
    let (lo, hi, half) = wilson_bounds(successes, trials, z);
    let used = match mode {
        RateBoundMode::None => {
            if trials == 0 {
                0.0
            } else {
                (successes.min(trials) as f64) / (trials as f64)
            }
        }
        RateBoundMode::Upper => hi,
        RateBoundMode::Lower => lo,
    };
    (used, half)
}

/// Uncertainty-adjusted rates for a summary (returns the used rates + their half-widths).
#[derive(Debug, Clone, Copy)]
pub struct AdjustedRates {
    pub ok_rate: f64,
    pub ok_half: f64,
    pub junk_rate: f64,
    pub junk_half: f64,
    pub hard_junk_rate: f64,
    pub hard_junk_half: f64,
}

pub fn adjusted_rates(summary: Summary, cfg: UncertaintyConfig) -> AdjustedRates {
    let calls = summary.calls;
    let (ok_rate, ok_half) = apply_rate_bound(summary.ok, calls, cfg.z, cfg.ok_mode);
    let (junk_rate, junk_half) = apply_rate_bound(summary.junk, calls, cfg.z, cfg.junk_mode);
    let (hard_junk_rate, hard_junk_half) =
        apply_rate_bound(summary.hard_junk, calls, cfg.z, cfg.hard_junk_mode);

    AdjustedRates {
        ok_rate,
        ok_half,
        junk_rate,
        junk_half,
        hard_junk_rate,
        hard_junk_half,
    }
}

// ---------------------------------------------------------------------------
// Convenience CUSUM threshold calibration via Monte Carlo null simulation
// ---------------------------------------------------------------------------

/// Simulate null max-scores for a CUSUM bank over `m` rounds.
///
/// Feeds `n_trials` independent null sequences (each of length `m`) through a
/// `CusumCatBank`, recording the maximum CUSUM score per trial.  The resulting
/// distribution can be fed directly to [`calibrate_threshold_from_max_scores`].
///
/// # Arguments
///
/// - `p0`: null (pre-change) categorical distribution (must be a valid simplex).
/// - `alts`: alternative distributions for the CUSUM bank.
/// - `m`: horizon in sample-time rounds (number of outcomes per null trial).
/// - `cusum_alpha`: Dirichlet smoothing for the CUSUM detector.
/// - `min_n`: minimum observations before the detector can alarm.
/// - `n_trials`: number of Monte Carlo null trials.
/// - `seed`: RNG seed for reproducibility.
///
/// Returns `Err` if the bank cannot be initialised (e.g. invalid simplex).
#[cfg(feature = "stochastic")]
pub fn simulate_cusum_null_max_scores(
    p0: &[f64],
    alts: &[Vec<f64>],
    m: u64,
    cusum_alpha: f64,
    min_n: u64,
    n_trials: usize,
    seed: u64,
) -> Result<Vec<f64>, logp::Error> {
    use rand::Rng;
    use rand::SeedableRng;

    if n_trials == 0 || m == 0 {
        return Ok(Vec::new());
    }

    // Build cumulative CDF for sampling from p0.
    let sum: f64 = p0.iter().sum();
    let cdf: Vec<f64> = {
        let mut c = Vec::with_capacity(p0.len());
        let mut acc = 0.0;
        for &v in p0 {
            acc += v / sum;
            c.push(acc.min(1.0));
        }
        c
    };

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut max_scores = Vec::with_capacity(n_trials);

    for _ in 0..n_trials {
        let mut bank = CusumCatBank::new(p0, alts, cusum_alpha, min_n, f64::INFINITY, 1e-9)?;
        let mut max_score = 0.0_f64;
        for _ in 0..m {
            // Sample one outcome from p0.
            let u: f64 = rng.random();
            let idx = cdf.partition_point(|&c| c < u).min(p0.len() - 1);
            bank.update(idx);
            max_score = max_score.max(bank.score_max());
        }
        max_scores.push(max_score);
    }
    Ok(max_scores)
}

/// Calibrate a CUSUM threshold to satisfy `P[τ < m] ≤ alpha` under the null.
///
/// Combines [`simulate_cusum_null_max_scores`] with
/// [`calibrate_threshold_from_max_scores`] into a single convenience call.
///
/// # Arguments
///
/// - `p0`: null categorical distribution.
/// - `alts`: CUSUM alternative distributions.
/// - `alpha`: target false-alarm rate (e.g. `0.05` for 5%).
/// - `m`: survival horizon (rounds).  The returned threshold satisfies
///   approximately `P[alarm within m null rounds] ≤ alpha`.
/// - `n_trials`: Monte Carlo sample count (1000–10000 typical).
/// - `cusum_alpha`: Dirichlet smoothing.
/// - `min_n`: minimum observations before detector can alarm.
/// - `seed`: RNG seed.
/// - `require_wilson`: if true, require the Wilson upper bound ≤ alpha
///   (more conservative; needs larger `n_trials`).
///
/// Returns the [`ThresholdCalibration`] result, or `Err` if the CUSUM bank
/// cannot be initialised.
///
/// # Example
///
/// ```rust,no_run
/// # #[cfg(feature = "stochastic")]
/// # {
/// use muxer::monitor::calibrate_cusum_threshold;
///
/// let p0 = vec![0.85, 0.05, 0.05, 0.05];
/// let alts = vec![vec![0.40, 0.10, 0.40, 0.10]];
/// let cal = calibrate_cusum_threshold(&p0, &[alts[0].clone()], 0.05, 500, 2000, 1e-3, 20, 42, false)
///     .expect("calibration");
/// println!("threshold = {:.2}, fa_hat = {:.3}", cal.threshold, cal.fa_hat);
/// # }
/// ```
#[cfg(feature = "stochastic")]
#[allow(clippy::too_many_arguments)]
pub fn calibrate_cusum_threshold(
    p0: &[f64],
    alts: &[Vec<f64>],
    alpha: f64,
    m: u64,
    n_trials: usize,
    cusum_alpha: f64,
    min_n: u64,
    seed: u64,
    require_wilson: bool,
) -> Result<ThresholdCalibration, logp::Error> {
    let mut max_scores =
        simulate_cusum_null_max_scores(p0, alts, m, cusum_alpha, min_n, n_trials, seed)?;
    if max_scores.is_empty() {
        return Ok(ThresholdCalibration {
            threshold: 0.0,
            fa_hat: 1.0,
            fa_wilson_hi: 1.0,
            trials: 0,
            grid_satisfied: false,
        });
    }

    // Build a grid from quantiles of the null max-score distribution.
    let mut sorted = max_scores.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let mut grid: Vec<f64> = (0..=200)
        .map(|i| {
            let idx = ((i as f64 / 200.0) * n as f64) as usize;
            sorted[idx.min(n - 1)]
        })
        .collect();
    grid.sort_by(|a, b| a.total_cmp(b));
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    Ok(calibrate_threshold_from_max_scores(
        &mut max_scores,
        &grid,
        alpha,
        1.96,
        require_wilson,
    ))
}

#[cfg(test)]
mod threshold_tests {
    use super::*;

    #[test]
    fn cusum_infinite_threshold_never_alarms() {
        let p0 = [0.90, 0.03, 0.02, 0.05];
        let p1 = [0.05, 0.05, 0.45, 0.45];
        let mut d = CusumCatDetector::new(&p0, &p1, 1e-3, 1, f64::INFINITY, 1e-12).expect("new");
        for _ in 0..200 {
            assert!(d.update(3).is_none()); // drive score upward
        }
        assert!(d.score().is_finite());
    }

    #[test]
    fn catkl_infinite_threshold_never_alarms() {
        let p0 = [0.90, 0.03, 0.02, 0.05];
        let mut d = CatKlDetector::new(&p0, 1e-3, 1, f64::INFINITY, 1e-12).expect("new");
        for _ in 0..200 {
            assert!(d.update(3).is_none());
        }
        let s = d.score().expect("score");
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn simplex_vec(len: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(0.0f64..10.0, len).prop_map(|mut v| {
            let s: f64 = v.iter().sum();
            if s == 0.0 {
                v[0] = 1.0;
                return v;
            }
            for x in v.iter_mut() {
                *x /= s;
            }
            v
        })
    }

    #[test]
    fn calibrate_threshold_from_max_scores_picks_expected_grid_point() {
        let mut maxes = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let grid = [0.0, 2.0, 4.0, 6.0];
        let cal = calibrate_threshold_from_max_scores(&mut maxes, &grid, 0.5, 1.96, false);
        assert!(cal.grid_satisfied);
        assert_eq!(cal.threshold, 4.0);
        assert!(
            (cal.fa_hat - (2.0 / 6.0)).abs() <= 1e-12,
            "fa_hat={}",
            cal.fa_hat
        );
    }

    #[test]
    fn calibrate_threshold_from_max_scores_wilson_is_more_conservative_than_empirical() {
        // With small `trials`, Wilson upper bounds can be much larger than `fa_hat`,
        // forcing a more conservative (larger) threshold when `require_wilson=true`.
        let mut maxes_emp = vec![0.0, 3.0, 1.0, 9.0, 5.0, 6.0, 2.0, 8.0, 7.0, 4.0];
        let mut maxes_wil = maxes_emp.clone();
        let grid = [9.0, 10.0];
        let alpha = 0.3;
        let z = 1.96;

        let emp = calibrate_threshold_from_max_scores(&mut maxes_emp, &grid, alpha, z, false);
        assert!(emp.grid_satisfied);
        assert_eq!(emp.threshold, 9.0);
        assert!((emp.fa_hat - 0.1).abs() <= 1e-12);
        assert!(emp.fa_wilson_hi > alpha, "expected Wilson hi > alpha");

        let wil = calibrate_threshold_from_max_scores(&mut maxes_wil, &grid, alpha, z, true);
        assert!(wil.grid_satisfied);
        assert_eq!(wil.threshold, 10.0);
        assert!((wil.fa_hat - 0.0).abs() <= 1e-12);
        assert!(
            wil.fa_wilson_hi <= alpha + 1e-12,
            "wilson_hi={}",
            wil.fa_wilson_hi
        );
    }

    #[test]
    fn calibrate_threshold_from_max_scores_grid_satisfied_implies_constraint_holds() {
        let maxes = vec![0.0, 1.0, 2.0, 2.1, 2.2, 10.0, 10.0, 10.0];
        let grid = [0.0, 2.0, 5.0, 11.0];
        let alpha = 0.4;
        let z = 1.96;

        for require_wilson in [false, true] {
            let mut m = maxes.clone();
            let cal = calibrate_threshold_from_max_scores(&mut m, &grid, alpha, z, require_wilson);
            assert!(cal.trials as usize == maxes.len());
            if cal.grid_satisfied {
                if require_wilson {
                    assert!(cal.fa_wilson_hi <= alpha + 1e-12, "hi={}", cal.fa_wilson_hi);
                } else {
                    assert!(cal.fa_hat <= alpha + 1e-12, "fa={}", cal.fa_hat);
                }
            }
        }
    }

    #[test]
    fn calibrate_threshold_from_max_scores_reports_unsatisfied_grid() {
        let mut maxes = vec![10.0; 10];
        let grid = [1.0, 2.0, 3.0];
        let cal = calibrate_threshold_from_max_scores(&mut maxes, &grid, 0.1, 1.96, false);
        assert!(!cal.grid_satisfied);
        assert_eq!(cal.threshold, 3.0);
        assert!((cal.fa_hat - 1.0).abs() <= 1e-12, "fa_hat={}", cal.fa_hat);
    }

    #[test]
    fn cusum_cat_bank_matches_single_detector() {
        let p0: Vec<f64> = vec![0.90, 0.03, 0.02, 0.05];
        let alt: Vec<f64> = vec![0.05, 0.05, 0.45, 0.45];
        let alts = vec![alt.clone()];

        let alpha = 1e-3;
        let min_n = 3;
        let thr = 1e9;
        let tol = 1e-12;

        let mut bank = CusumCatBank::new(&p0, &alts, alpha, min_n, thr, tol).expect("bank");
        let mut det = CusumCatDetector::new(&p0, &alt, alpha, min_n, thr, tol).expect("det");

        let xs = [0usize, 0, 3, 3, 3, 0, 3];
        for &x in &xs {
            let u = bank.update(x);
            let a = det.update(x);
            assert_eq!(u.alarmed, a.is_some());
            assert_eq!(u.n, det.n());
            assert!((u.score_max - det.score()).abs() <= 1e-12);
        }
    }

    #[test]
    fn cusum_cat_bank_matches_max_of_two_detectors() {
        let p0: Vec<f64> = vec![0.90, 0.03, 0.02, 0.05];
        let alt_a: Vec<f64> = vec![0.05, 0.05, 0.45, 0.45];
        let alt_b: Vec<f64> = vec![0.05, 0.45, 0.05, 0.45];
        let alts = vec![alt_a.clone(), alt_b.clone()];

        let alpha = 1e-3;
        let min_n = 3;
        let thr = 1e9;
        let tol = 1e-12;

        let mut bank = CusumCatBank::new(&p0, &alts, alpha, min_n, thr, tol).expect("bank");
        let mut a = CusumCatDetector::new(&p0, &alt_a, alpha, min_n, thr, tol).expect("a");
        let mut b = CusumCatDetector::new(&p0, &alt_b, alpha, min_n, thr, tol).expect("b");

        // Mix categories so each alternative gets some positive LLR segments.
        let xs = [0usize, 1, 3, 3, 1, 3, 0, 3, 1, 3];
        for &x in &xs {
            let u = bank.update(x);
            let aa = a.update(x);
            let bb = b.update(x);

            let expected_alarm = aa.is_some() || bb.is_some();
            let expected_max = a.score().max(b.score());
            assert_eq!(u.alarmed, expected_alarm);
            assert_eq!(u.n, a.n());
            assert_eq!(a.n(), b.n());
            assert!((u.score_max - expected_max).abs() <= 1e-12);
        }
    }

    #[test]
    fn cusum_cat_bank_reset_clears_n_and_score() {
        let p0: Vec<f64> = vec![0.90, 0.03, 0.02, 0.05];
        let alt: Vec<f64> = vec![0.05, 0.05, 0.45, 0.45];
        let alts = vec![alt];

        let mut bank = CusumCatBank::new(&p0, &alts, 1e-3, 1, 1e9, 1e-12).expect("bank");
        for _ in 0..10 {
            let _ = bank.update(3);
        }
        assert!(bank.n() > 0);
        assert!(bank.score_max() >= 0.0);

        bank.reset();
        assert_eq!(bank.n(), 0);
        assert!((bank.score_max() - 0.0).abs() <= 1e-12);
    }

    #[test]
    fn cusum_cat_bank_rejects_empty_alts() {
        let p0: Vec<f64> = vec![0.90, 0.03, 0.02, 0.05];
        let alts: Vec<Vec<f64>> = Vec::new();
        let err = CusumCatBank::new(&p0, &alts, 1e-3, 1, 1.0, 1e-12).unwrap_err();
        match err {
            logp::Error::Domain(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn wilson_bounds_are_ordered_and_bounded() {
        let (lo, hi, half) = wilson_bounds(8, 10, 1.96);
        assert!(lo.is_finite() && hi.is_finite() && half.is_finite());
        assert!(0.0 <= lo && lo <= hi && hi <= 1.0);
        assert!(half >= 0.0);
    }

    #[test]
    fn drift_between_windows_is_zero_for_identical() {
        let mut b = Window::new(50);
        let mut r = Window::new(50);
        for _ in 0..30 {
            let o = Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            };
            b.push(o);
            r.push(o);
        }
        let cfg = DriftConfig {
            metric: DriftMetric::Rao,
            tol: 1e-12,
            min_baseline: 10,
            min_recent: 10,
        };
        let d = drift_between_windows(&b, &r, cfg).expect("drift decision");
        assert!(d.score.abs() < 1e-12, "score={}", d.score);
    }

    #[test]
    fn drift_between_windows_increases_when_distributions_differ() {
        let mut b = Window::new(200);
        let mut r = Window::new(200);
        // baseline: mostly ok_clean
        for _ in 0..80 {
            b.push(Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            });
        }
        // recent: mostly hard junk
        for _ in 0..80 {
            r.push(Outcome {
                ok: true,
                junk: true,
                hard_junk: true,
                cost_units: 0,
                elapsed_ms: 0,
                quality_score: None,
            });
        }

        let cfg = DriftConfig {
            metric: DriftMetric::Hellinger,
            tol: 1e-12,
            min_baseline: 10,
            min_recent: 10,
        };
        let d = drift_between_windows(&b, &r, cfg).expect("drift decision");
        assert!(d.score > 0.1, "score={}", d.score);
    }

    proptest! {
        #[test]
        fn drift_metrics_are_symmetric(p in simplex_vec(7), q in simplex_vec(7)) {
            let tol = 1e-9;
            for metric in [DriftMetric::Rao, DriftMetric::Hellinger, DriftMetric::JensenShannon] {
                let d1 = drift_simplex(&p, &q, metric, tol).unwrap();
                let d2 = drift_simplex(&q, &p, metric, tol).unwrap();
                prop_assert!((d1 - d2).abs() < 1e-12, "metric={metric:?} d1={d1} d2={d2}");
            }
        }

        #[test]
        fn drift_bounds_hold(p in simplex_vec(9), q in simplex_vec(9)) {
            let tol = 1e-9;
            let rao = drift_simplex(&p, &q, DriftMetric::Rao, tol).unwrap();
            prop_assert!(rao >= -1e-12);
            prop_assert!(rao <= core::f64::consts::PI + 1e-12);

            let h = drift_simplex(&p, &q, DriftMetric::Hellinger, tol).unwrap();
            prop_assert!(h >= -1e-12);
            prop_assert!(h <= 1.0 + 1e-12);

            let js = drift_simplex(&p, &q, DriftMetric::JensenShannon, tol).unwrap();
            prop_assert!(js >= -1e-12);
            prop_assert!(js <= logp::LN_2 + 1e-9);
        }

        #[test]
        fn triangle_inequality_holds_for_metrics(p in simplex_vec(8), q in simplex_vec(8), r in simplex_vec(8)) {
            let tol = 1e-9;
            let eps = 1e-9;

            // Hellinger is a metric.
            let pq = drift_simplex(&p, &q, DriftMetric::Hellinger, tol).unwrap();
            let qr = drift_simplex(&q, &r, DriftMetric::Hellinger, tol).unwrap();
            let pr = drift_simplex(&p, &r, DriftMetric::Hellinger, tol).unwrap();
            prop_assert!(pr <= pq + qr + eps, "H: pr={pr} pq+qr={}", pq+qr);

            // Rao distance on the categorical simplex is a metric.
            let pq = drift_simplex(&p, &q, DriftMetric::Rao, tol).unwrap();
            let qr = drift_simplex(&q, &r, DriftMetric::Rao, tol).unwrap();
            let pr = drift_simplex(&p, &r, DriftMetric::Rao, tol).unwrap();
            prop_assert!(pr <= pq + qr + 1e-6, "Rao: pr={pr} pq+qr={}", pq+qr);
        }

        #[test]
        fn wilson_bounds_contain_empirical_rate(
            trials in 1u64..500,
            successes in 0u64..500,
            z in 0.5f64..5.0f64,
        ) {
            let s = successes.min(trials);
            let p_hat = (s as f64) / (trials as f64);
            let (lo, hi, _half) = wilson_bounds(s, trials, z);
            prop_assert!(lo <= p_hat + 1e-12);
            prop_assert!(p_hat <= hi + 1e-12);
        }

        #[test]
        fn apply_rate_bound_upper_ge_lower(
            trials in 1u64..500,
            successes in 0u64..500,
            z in 0.5f64..5.0f64,
        ) {
            let s = successes.min(trials);
            let (u, _) = apply_rate_bound(s, trials, z, RateBoundMode::Upper);
            let (l, _) = apply_rate_bound(s, trials, z, RateBoundMode::Lower);
            prop_assert!(u >= l - 1e-12);
        }

        #[test]
        fn wilson_half_width_shrinks_with_more_trials_at_half_success(
            n1 in 5u64..200,
            n2 in 201u64..2_000,
            z in 0.5f64..5.0f64,
        ) {
            // Use ~50% success to avoid edge pathologies at 0 or 1.
            let s1 = n1 / 2;
            let s2 = n2 / 2;
            let (_lo1, _hi1, h1) = wilson_bounds(s1, n1, z);
            let (_lo2, _hi2, h2) = wilson_bounds(s2, n2, z);
            prop_assert!(h2 <= h1 + 1e-12, "h1={h1} h2={h2}");
        }

        #[test]
        fn calibrate_threshold_from_max_scores_is_monotone_in_alpha(
            max_scores in prop::collection::vec(-1000.0f64..1000.0, 1..80),
            mut grid in prop::collection::vec(-1000.0f64..1000.0, 1..60),
            a1 in 0.0f64..1.0f64,
            a2 in 0.0f64..1.0f64,
            z in 0.5f64..5.0f64,
            require_wilson in any::<bool>(),
        ) {
            grid.sort_by(|a, b| a.total_cmp(b));
            let (alpha_small, alpha_large) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };

            let mut m_small = max_scores.clone();
            let mut m_large = max_scores;
            let cal_small =
                calibrate_threshold_from_max_scores(&mut m_small, &grid, alpha_small, z, require_wilson);
            let cal_large =
                calibrate_threshold_from_max_scores(&mut m_large, &grid, alpha_large, z, require_wilson);

            // Stricter alpha implies a threshold that is >= (more conservative).
            prop_assert!(cal_small.threshold >= cal_large.threshold);
        }
    }

    #[test]
    fn catkl_detector_stays_near_zero_when_empirical_matches_baseline_uniform() {
        // Deterministic sequence that exactly matches uniform over 4 categories.
        let p0 = [0.25, 0.25, 0.25, 0.25];
        let mut d = CatKlDetector::new(&p0, 1e-9, 20, 10.0, 1e-12).unwrap();
        for t in 0..200 {
            let _ = d.update((t as usize) % 4);
        }
        let s = d.score().unwrap();
        assert!(s < 1e-6, "score={}", s);
    }

    #[test]
    fn catkl_detector_triggers_on_strong_shift() {
        // Baseline uniform, observations all go to category 0 -> large KL.
        let p0 = [0.25, 0.25, 0.25, 0.25];
        let mut d = CatKlDetector::new(&p0, 1e-3, 10, 5.0, 1e-12).unwrap();
        let mut alarmed = false;
        for _ in 0..200 {
            if d.update(0).is_some() {
                alarmed = true;
                break;
            }
        }
        assert!(alarmed, "expected alarm");
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 96, .. ProptestConfig::default() })]
        #[test]
        fn cusum_never_alarms_when_p1_equals_p0(
            p in simplex_vec(6),
            idxs in prop::collection::vec(0usize..6, 1..300),
        ) {
            let mut d = CusumCatDetector::new(&p, &p, 1e-6, 1, 1e-9, 1e-12).unwrap();
            for idx in idxs {
                let alarm = d.update(idx);
                prop_assert!(alarm.is_none());
                prop_assert!(d.score().abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn cusum_triggers_on_persistent_positive_llr_stream() {
        let p0 = [0.25, 0.25, 0.25, 0.25];
        let p1 = [0.70, 0.10, 0.10, 0.10];
        let mut d = CusumCatDetector::new(&p0, &p1, 1e-6, 5, 2.0, 1e-12).unwrap();
        let mut alarmed = false;
        for _ in 0..200 {
            if d.update(0).is_some() {
                alarmed = true;
                break;
            }
        }
        assert!(alarmed, "expected alarm");
    }

    // ========================================================================
    // CUSUM detection delay / sensitivity formula verification
    //
    // These tests connect muxer's CusumCatDetector to the theoretical detection
    // delay formulas from the objective manifold research program.
    //
    // Background: Page's CUSUM (1954) maintains a running sum
    //   S_t = max(0, S_{t-1} + log(p1(X_t) / p0(X_t)))
    // and alarms when S_t >= threshold h.
    //
    // Under the alternative (all observations from p1), the expected per-step
    // increment is KL(p1 || p0) > 0, so the expected number of samples to
    // alarm is approximately h / KL(p1 || p0).  This is Lorden's (1971) bound.
    //
    // When observations arrive at rate r (fraction of time steps where the arm
    // is sampled), the wall-clock delay is:
    //   E[wall_delay] ~ E[sample_delay] / r = h / (r * KL(p1 || p0))
    //
    // For a deterministic observation sequence (always the same category),
    // the per-step LLR is fixed at log(p1[cat] / p0[cat]), and the delay
    // is exactly ceil(h / per_step_llr).
    // ========================================================================

    /// Verify E[delay_samples] ~ h / per_step_llr for a deterministic post-change
    /// observation stream.
    ///
    /// We feed the CUSUM the category with the highest likelihood ratio (p1 >> p0),
    /// so each observation contributes a large positive LLR increment.  The CUSUM
    /// should alarm after approximately h / llr observations.
    ///
    /// The tolerance is intentionally wide (0.5x-2.5x) because the CusumCatDetector
    /// constructor applies Dirichlet smoothing (alpha=1e-6), which slightly shifts
    /// the effective p0/p1 from the nominal values.
    #[test]
    fn cusum_detection_delay_matches_formula() {
        let p0 = [0.90, 0.03, 0.02, 0.05];
        let p1 = [0.05, 0.05, 0.45, 0.45];
        let alpha = 1e-6;
        let tol = 1e-12;

        // Category 2 has the largest positive LLR:
        //   ln(p1[2]/p0[2]) = ln(0.45/0.02) = ln(22.5) ~ 3.11
        // Categories 0 and 1 have negative LLR (p1 < p0 for those).
        let best_cat = 2;

        for &h in &[2.0, 5.0, 10.0] {
            let mut d = CusumCatDetector::new(&p0, &p1, alpha, 1, h, tol).unwrap();

            let per_step_llr = (p1[best_cat] / p0[best_cat]).ln();
            assert!(per_step_llr > 0.0);

            let predicted_delay = (h / per_step_llr).ceil() as u64;

            let mut actual_delay = 0u64;
            for step in 0..10_000u64 {
                if d.update(best_cat).is_some() {
                    actual_delay = step + 1;
                    break;
                }
            }

            assert!(actual_delay > 0, "CUSUM should eventually alarm at h={h}");
            let ratio = actual_delay as f64 / predicted_delay as f64;
            assert!(
                (0.5..2.5).contains(&ratio),
                "h={h}: actual={actual_delay}, predicted={predicted_delay}, ratio={ratio:.3}"
            );
        }
    }

    /// CUSUM score is monotone non-decreasing when fed only positive-LLR observations.
    ///
    /// This is a basic sanity check: if every observation has p1[cat] > p0[cat],
    /// then the log-likelihood ratio is always positive, and the CUSUM sum
    /// S_t = max(0, S_{t-1} + llr) is strictly increasing (since S_{t-1} >= 0
    /// and llr > 0).
    #[test]
    fn cusum_score_monotone_under_positive_llr() {
        let p0 = [0.90, 0.03, 0.02, 0.05];
        let p1 = [0.05, 0.05, 0.45, 0.45];
        let alpha = 1e-6;
        let tol = 1e-12;

        // Use "never alarm" threshold so we can observe the full score trajectory.
        let mut d = CusumCatDetector::new(&p0, &p1, alpha, 1, f64::INFINITY, tol).unwrap();

        let best_cat = 2; // positive LLR
        let mut prev_score = 0.0;
        for _ in 0..100 {
            let _ = d.update(best_cat);
            let s = d.score();
            assert!(
                s >= prev_score - 1e-12,
                "score decreased: {prev_score} -> {s}"
            );
            prev_score = s;
        }
        assert!(
            prev_score > 0.0,
            "score should be positive after 100 positive-LLR obs"
        );
    }

    /// Wall-clock delay scales inversely with sampling rate.
    ///
    /// If the CUSUM needs N post-change samples to alarm, and arm observations
    /// arrive at rate r (fraction of wall-clock steps), then:
    ///   wall_delay = N / r
    ///
    /// Doubling the sampling rate halves the wall delay (for fixed sample delay).
    /// This is the "two clocks" concept from EXPERIMENTS.md: wall time vs sample time.
    #[test]
    fn cusum_delay_scales_inversely_with_sampling_rate() {
        let p0 = [0.90, 0.03, 0.02, 0.05];
        let p1 = [0.30, 0.10, 0.30, 0.30];
        let h = 5.0;
        let alpha = 1e-6;
        let tol = 1e-12;

        let mut d = CusumCatDetector::new(&p0, &p1, alpha, 1, h, tol).unwrap();
        let mut sample_delay = 0u64;
        let best_cat = 2; // p1[2]/p0[2] = 0.30/0.02 = 15 (highest ratio)
        for step in 0..10_000 {
            if d.update(best_cat).is_some() {
                sample_delay = step + 1;
                break;
            }
        }
        assert!(sample_delay > 0, "should alarm");

        // Two sampling rates: fast (50%) and slow (5%).
        let rate_fast = 0.5;
        let rate_slow = 0.05;
        let wall_fast = sample_delay as f64 / rate_fast;
        let wall_slow = sample_delay as f64 / rate_slow;

        assert!(wall_fast < wall_slow);
        // Ratio of wall delays should equal ratio of rates (inverted).
        let ratio = wall_slow / wall_fast;
        assert!(
            (ratio - (rate_fast / rate_slow)).abs() < 1e-9,
            "wall delay ratio = {ratio}, expected {}",
            rate_fast / rate_slow
        );
    }

    // ========================================================================
    // Product bound and proportionality tests (analytical)
    //
    // These verify the core identities from the objective manifold framework
    // using closed-form formulas (no simulation, no approximation).
    // ========================================================================

    /// R_T * D_avg = 2*b*Delta*T / delta^2 for all n_2.
    ///
    /// This is Theorem 1 from the research program.  The product depends only on
    /// the problem parameters (gap Delta, change magnitude delta, threshold b,
    /// horizon T), not on the allocation n_2.  This means the regret-detection
    /// tradeoff is a hyperbola: you slide along it by changing n_2, but you
    /// cannot escape it.
    #[test]
    fn product_bound_regret_times_delay_is_constant() {
        let delta = 0.5_f64;
        let delta_det = 0.3_f64;
        let t = 1000.0_f64;
        let b = 1.0_f64;

        let expected = 2.0 * b * delta * t / (delta_det * delta_det);

        for n2 in [5, 10, 50, 100, 250, 500, 800] {
            let n2f = n2 as f64;
            let r = delta * n2f;
            let d = 2.0 * b * t / (delta_det * delta_det * n2f);
            let product = r * d;
            let err = ((product - expected) / expected).abs();
            assert!(
                err < 1e-12,
                "n2={n2}: product={product}, expected={expected}"
            );
        }
    }

    /// D_avg / MSE = constant (independent of n).
    ///
    /// Both D_avg and MSE are O(1/n), so their ratio is a constant that depends
    /// on the change parameters, the false alarm constraint, and the horizon:
    ///
    ///   D_avg / MSE = 2 * ln(1/alpha) * T / delta^2
    ///                 (when sigma^2 = 1)
    ///
    /// This is the proportionality that makes average detection structurally
    /// redundant with estimation in the non-contextual case.
    #[test]
    fn detection_mse_ratio_is_constant() {
        let sigma2 = 1.0_f64;
        let alpha = 0.05_f64;
        let delta = 0.3_f64;
        let t = 500.0_f64;

        let c = 2.0 * sigma2 * (1.0 / alpha).ln() / (delta * delta);

        for n in [10, 30, 60, 120, 300] {
            let n = n as f64;
            let mse = sigma2 / n;
            let delay = c * t / n;
            let ratio = delay / mse;
            let expected = c * t / sigma2;
            let err = ((ratio - expected) / expected).abs();
            assert!(err < 1e-12, "n={n}: ratio={ratio}, expected={expected}");
        }
    }
}
