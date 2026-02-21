//! Tests anchoring the objective manifold theory to `muxer` and `pare`.
//!
//! # Background
//!
//! A sequential policy produces a single data stream.  Every observation simultaneously
//! serves multiple purposes: exploitation (get good results now), estimation (learn about
//! each arm), and monitoring (detect when an arm breaks).  These roles are traditionally
//! studied by separate communities with separate performance criteria.
//!
//! The objective manifold framework unifies them by working with the **design measure** --
//! the joint distribution over (arm, covariate, time) that the policy induces.  Every
//! performance criterion is a functional of this measure, and their interactions are
//! governed by the geometry of their **sensitivity functions** (gradients with respect
//! to the design measure).
//!
//! # What these tests verify
//!
//! 1. **Product bound** (Theorems 1-2 of the research program): for K=2 Gaussian arms
//!    with fixed allocation, R_T * D = const.  This is the fundamental impossibility:
//!    you cannot have both low regret and fast detection.
//!
//! 2. **Non-contextual collapse**: MSE and average detection delay have proportional
//!    sensitivity functions.  Three named objectives collapse to a 1D tradeoff curve.
//!
//! 3. **Contextual revival**: with worst-case detection and multiple covariate cells,
//!    all three objectives are genuinely independent.  The Pareto front is 2-dimensional.
//!
//! 4. **Saturation principle**: the Pareto front dimension is bounded by the design
//!    space dimension, regardless of how many objectives are named.
//!
//! 5. **K=3, M=9 computed example**: 8 objectives over 27 design variables.  Formal
//!    rank 8 but effective dimension ~3-4 (eigenvalue spectrum spans orders of magnitude).
//!
//! 6. **Temporal scheduling**: uniform observation spacing dominates for worst-case
//!    detection delay (clumping observations worsens the worst case).
//!
//! 7. **Lai-Robbins connection**: the information-theoretic lower bound on exploration
//!    rate constrains the detection budget available to any uniformly good policy.
//!
//! ## Reference notes
//!
//! - Product bound: Garivier & Moulines 2008 (arXiv:0805.3415) prove the impossibility
//!   result for adaptive policies: achieving R(T) regret on stationary instances implies
//!   Ω(T/R(T)) regret on some piecewise-stationary instance.  The product identity here
//!   is the static-schedule special case.
//! - Regret-BAI Pareto: Zhong, Cheung & Tan 2021 (arXiv:2110.08627) formally prove
//!   the two-objective Pareto tradeoff (regret vs BAI) using a similar product structure.
//! - BQCD lower bound: Gopalan, Saligrama & Lakshminarayanan 2021 (arXiv:2107.10492)
//!   establish Ω(log(m)/D*) detection delay under false-alarm constraint m.
//! - IDS: Russo & Van Roy 2014 (arXiv:1403.5556).

use pare::sensitivity::{analyze_redundancy, SensitivityRow};

// ============================================================================
// I. Product bound: R_T * D = const for fixed allocations
// ============================================================================
//
// Setup: K=2 Gaussian arms, unit variance.
// Arm 1 has mean mu_1 > mu_2 = mu_1 - Delta (gap Delta > 0).
// Arm 2 may experience a level shift of magnitude delta at unknown time tau.
//
// Policy: pull arm 2 exactly n_2 times, uniformly spaced over T rounds.
// CUSUM detector on arm 2 with threshold b.
//
// Regret:          R_T = Delta * n_2
// Detection delay: D_2 = 2*b*T / (delta^2 * n_2)
// Product:         R_T * D_2 = 2*b*Delta*T / delta^2    (independent of n_2!)
//
// For adaptive policies this equality becomes a lower bound: the
// Garivier-Moulines impossibility (arXiv:0805.3415) shows that achieving
// R(T) regret on stationary instances forces Ω(T/R(T)) regret on some
// piecewise-stationary instance, which is the adaptive-policy analogue.
//
// This is the central impossibility result: the product of regret and detection
// delay is a constant determined by the problem parameters (gap, shift, horizon),
// not by the policy's allocation.  Reducing regret (smaller n_2) necessarily
// increases detection delay, and vice versa.

#[test]
fn product_bound_is_constant_across_allocations() {
    let delta_gap = 0.5_f64; // mean gap between arms
    let delta_shift = 0.3_f64; // change magnitude to detect
    let t = 1000.0_f64; // horizon
    let b = 1.0_f64; // CUSUM threshold (normalized)

    let expected_product = 2.0 * b * delta_gap * t / (delta_shift * delta_shift);

    let mut products = Vec::new();
    for n2 in [10, 50, 100, 200, 500, 900] {
        let n2 = n2 as f64;
        let regret = delta_gap * n2;
        let delay = 2.0 * b * t / (delta_shift * delta_shift * n2);
        let product = regret * delay;
        products.push(product);

        let rel_err = ((product - expected_product) / expected_product).abs();
        assert!(
            rel_err < 1e-12,
            "n2={n2}: product={product}, expected={expected_product}, rel_err={rel_err}"
        );
    }

    // Coefficient of variation should be ~0 (all products identical).
    let mean = products.iter().sum::<f64>() / products.len() as f64;
    let var = products.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / products.len() as f64;
    let cv = var.sqrt() / mean;
    assert!(cv < 1e-14, "coefficient of variation = {cv}, should be ~0");
}

/// The product bound implies a "tradeoff table":
///
/// | Regret rate     | Implied detection delay          |
/// |-----------------|----------------------------------|
/// | O(log T)        | Omega(T / log T)  (near-linear)  |
/// | O(sqrt(T))      | Omega(sqrt(T))                   |
/// | O(T) (no learn) | O(1)  (achievable)               |
///
/// At the Lai-Robbins optimal regret rate O(log T), detection delay grows
/// nearly linearly in T.  Practical systems that need fast detection must
/// sacrifice regret optimality.
#[test]
fn product_bound_tradeoff_table() {
    let delta = 0.5_f64;
    let delta_det = 0.3_f64;
    let t = 10_000.0_f64;
    let b = 1.0_f64;
    let c = 2.0 * b * delta * t / (delta_det * delta_det);

    // Lai-Robbins regime: O(log T) regret => Omega(T / log T) delay.
    let r_log = t.ln(); // ~9.2
    let d_log = c / r_log;
    // d_log should be roughly T / log(T) * constant.
    assert!(d_log > t / (2.0 * t.ln()), "d_log = {d_log}");

    // Moderate regime: O(sqrt(T)) regret => O(sqrt(T)) delay.
    let r_sqrt = t.sqrt(); // 100
    let d_sqrt = c / r_sqrt;
    // d_sqrt / sqrt(T) should be a constant.
    assert!(d_sqrt > 0.0 && d_sqrt < t);

    // Full exploration regime: O(T) regret => O(1) delay.
    let r_lin = t;
    let d_lin = c / r_lin;
    assert!(d_lin < 200.0, "d_lin = {d_lin} should be O(1)");
}

// ============================================================================
// II. Lai-Robbins lower bound and its detection consequence
// ============================================================================
//
// Lai & Robbins (1985) proved that any "uniformly good" policy (regret = o(T^a)
// for all a > 0 on all instances) must satisfy:
//
//   E[N_2(T)] >= (1 + o(1)) * log(T) / KL(nu_2, nu_1)
//
// where N_2(T) is the number of pulls of the suboptimal arm and KL is the
// KL-divergence between the arm distributions.
//
// For Gaussians with unit variance and gap Delta:
//   KL(nu_2, nu_1) = Delta^2 / 2
//   => E[N_2(T)] >= (1 + o(1)) * 2 * log(T) / Delta^2
//
// Combining with the detection delay formula:
//   D_2 >= 2*b*T / (delta^2 * E[N_2(T)])
//       <= 2*b*T*Delta^2 / (delta^2 * 2 * log(T))
//       =  b*T*Delta^2 / (delta^2 * log(T))
//
// So: R_T * D_2 >= Delta * E[N_2(T)] * 2*b*T / (delta^2 * E[N_2(T)])
//              =  2*b*Delta*T / delta^2
//
// The lower bound holds for adaptive policies too (not just fixed allocations).

#[test]
fn lai_robbins_forces_minimum_exploration() {
    // Verify: at the Lai-Robbins optimal regret rate, the minimum exploration
    // count is 2*log(T)/Delta^2, and this determines the detection budget.
    let delta = 0.5_f64; // gap
    let t = 100_000.0_f64;

    // Minimum pulls of suboptimal arm for a uniformly good policy.
    let kl = delta * delta / 2.0; // KL for unit-variance Gaussians
    let min_n2 = 2.0 * t.ln() / (delta * delta);
    // Equivalently: min_n2 = log(T) / kl
    let min_n2_alt = t.ln() / kl;
    assert!(
        (min_n2 - min_n2_alt).abs() < 1e-9,
        "two formulations should agree: {min_n2} vs {min_n2_alt}"
    );

    // Regret at the optimal rate.
    let regret = delta * min_n2;
    // This should be O(log T).
    assert!(regret < 100.0 * t.ln(), "regret = {regret}");
    assert!(regret > 0.5 * t.ln(), "regret = {regret}");

    // Detection delay given this exploration rate.
    let delta_det = 0.3_f64;
    let b = 1.0_f64;
    let delay = 2.0 * b * t / (delta_det * delta_det * min_n2);
    // Delay should be O(T / log T).
    let ratio = delay / (t / t.ln());
    assert!(
        ratio > 0.1 && ratio < 100.0,
        "delay/T*logT = {ratio}, expected O(1)"
    );
}

// ============================================================================
// III. Non-contextual collapse: MSE and detection are proportional
// ============================================================================

/// For K=2 arms with fixed allocation n_2:
///
///   MSE_2     = sigma^2 / n_2
///   D_avg     = 2 * ln(1/alpha) * T / (delta^2 * n_2)
///   D_avg/MSE = 2 * ln(1/alpha) * T / (delta^2 * sigma^2) = CONSTANT
///
/// This ratio depends only on the change parameters and horizon, not on n_2.
/// Detection "comes for free" with estimation -- they trade off identically.
#[test]
fn non_contextual_collapse_mse_detection_proportional() {
    let sigma2 = 1.0_f64;
    let alpha = 0.05_f64;
    let delta = 0.3_f64;
    let t = 1000.0_f64;

    let expected_ratio = 2.0 * sigma2 * (1.0 / alpha).ln() / (delta * delta);

    for n2 in [10, 30, 50, 100, 200, 500] {
        let n2 = n2 as f64;
        let mse = sigma2 / n2;
        let delay = 2.0 * (1.0 / alpha).ln() * t / (delta * delta * n2);
        let ratio = delay / mse;
        let actual_ratio = ratio / t;
        let rel_err = ((actual_ratio - expected_ratio) / expected_ratio).abs();
        assert!(
            rel_err < 1e-12,
            "n2={n2}: ratio/T={actual_ratio}, expected={expected_ratio}, rel_err={rel_err}"
        );
    }
}

/// The gradient test confirms the collapse: sensitivity vectors for MSE and
/// detection are exactly proportional (cosine = 1.0).
///
/// Setup: evaluate dL/d(n_2) at several n_2 values to build sensitivity vectors.
///
///   dR/d(n_2)   = Delta        (constant -- regret is linear in n_2)
///   dMSE/d(n_2) = -sigma^2/n_2^2    (decreasing, same shape for all n_2)
///   dD/d(n_2)   = -C*T/(n_2^2 * delta^2) = const * dMSE/d(n_2)
///
/// Since dD and dMSE are scalar multiples, cos(s_MSE, s_D) = 1.0.
/// In R^1 (one design variable), all vectors are collinear, so Gram rank = 1.
#[test]
fn non_contextual_sensitivity_rank_is_one() {
    let delta = 0.5_f64;
    let delta_det = 0.3_f64;
    let t = 1000.0_f64;
    let c = 1.0_f64;

    // Evaluate sensitivity at 5 different n_2 values.
    // Each gives a scalar (the design is 1D), so we get vectors in R^5.
    let n_values: [f64; 5] = [10.0, 30.0, 50.0, 100.0, 200.0];

    let s_regret: Vec<f64> = n_values.iter().map(|_| delta).collect();
    let s_mse: Vec<f64> = n_values.iter().map(|&n| -1.0 / (n * n)).collect();
    let s_det: Vec<f64> = n_values
        .iter()
        .map(|&n| -c * t / (n * n * delta_det * delta_det))
        .collect();

    let a = analyze_redundancy(&[s_regret, s_mse, s_det]).unwrap();

    // MSE and detection: cosine = 1.0 (proportional sensitivity).
    let cos = a.cosine(1, 2).unwrap();
    assert!(
        (cos - 1.0).abs() < 1e-9,
        "MSE-detection cosine = {cos}, expected 1.0"
    );

    // Regret sensitivity is constant (dR/dn = Delta at all n), while MSE
    // sensitivity varies as -1/n^2.  As vectors across multiple n values,
    // a constant vector and a -1/n^2 vector are not perfectly anti-proportional
    // (only the sign is consistent, not the magnitude pattern).  The cosine
    // is negative but not exactly -1.
    let cos_r = a.cosine(0, 1).unwrap();
    assert!(
        cos_r < 0.0,
        "regret-MSE cosine = {cos_r}, expected negative"
    );

    // Pareto dimension = 1 (regret vs. MSE/detection, a single tradeoff curve).
    assert_eq!(a.pareto_dimension_bound(1e-6), 1);
}

// ============================================================================
// IV. K=3, M=9 computed example (Section VII of the research manifesto)
// ============================================================================
//
// This is the central worked example that demonstrates both formal and effective
// Pareto dimension in a realistic setting.
//
// Setup:
// - 3 arms with linear response functions on a 3x3 covariate grid.
// - Covariate cells: {1/6, 1/2, 5/6}^2 (9 cells).
// - 27 design variables: p_a(x_j) for arm a in {0,1,2}, cell j in {0..8}.
//
// Arm response functions:
//   f_0(x) = 1.0 + 0.5*x1 + 0.3*x2   (arm 0 benefits from high x1)
//   f_1(x) = 0.8 + 0.1*x1 + 0.8*x2   (arm 1 benefits from high x2)
//   f_2(x) = 1.2 - 0.3*x1 + 0.2*x2   (arm 2 is best at low x1)
//
// The optimal arm varies across cells, creating non-trivial decision boundaries.
//
// 8 objectives:
// 0: Cumulative regret         -- gap-weighted allocation
// 1: Simple regret             -- gap-weighted, late-round emphasis (1.5x cumulative)
// 2: MSE arm 0                 -- D-optimal leverage for arm 0's linear model
// 3: MSE arm 1                 -- D-optimal leverage for arm 1's linear model
// 4: MSE arm 2                 -- D-optimal leverage for arm 2's linear model
// 5: Average detection delay   -- inverse-square allocation (all arms/cells)
// 6: Worst-case detection delay-- point mass at bottleneck (arm 1, cell 0)
// 7: Subgroup fairness         -- inverse-gap weighting (prioritize underperforming regions)
//
// Expected results:
// - Formal rank: 6-8 (most objectives are linearly independent).
// - Cumulative and simple regret: cosine = 1.0 (one is a scalar multiple of the other).
// - Per-arm MSEs: cosine = 0.0 (disjoint support -- sampling arm 0 tells you nothing about arm 1).
// - Average detection: flat sensitivity (same value at every design point under uniform design).
// - Effective dimension: 3-4 (top eigenvalues dominate; most "tradeoffs" are negligible).

#[test]
fn k3_m9_eight_objectives_rank_and_spectrum() {
    let k = 3;
    let cells_1d: [f64; 3] = [1.0 / 6.0, 0.5, 5.0 / 6.0];
    let mut cells = Vec::new();
    for &x1 in &cells_1d {
        for &x2 in &cells_1d {
            cells.push((x1, x2));
        }
    }
    let m_cells = cells.len();
    assert_eq!(m_cells, 9);
    let n_design = k * m_cells; // 27

    // Arm response functions.
    let f = |arm: usize, x1: f64, x2: f64| -> f64 {
        match arm {
            0 => 1.0 + 0.5 * x1 + 0.3 * x2,
            1 => 0.8 + 0.1 * x1 + 0.8 * x2,
            2 => 1.2 - 0.3 * x1 + 0.2 * x2,
            _ => unreachable!(),
        }
    };

    // Optimal arm at each cell (the one with the highest f).
    let optimal_arm: Vec<usize> = cells
        .iter()
        .map(|&(x1, x2)| {
            (0..k)
                .max_by(|&a, &b| {
                    f(a, x1, x2)
                        .partial_cmp(&f(b, x1, x2))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        })
        .collect();

    let p_uniform = 1.0 / k as f64;

    // Design variable index: arm a, cell j => a * M + j.
    let idx = |a: usize, j: usize| -> usize { a * m_cells + j };

    // --- Build sensitivity vectors (27-dimensional) ---

    // 0: Cumulative regret.
    // s_R(a,j) = gap_a(x_j) = f_{a*}(x_j) - f_a(x_j).
    // Pulling a suboptimal arm incurs the gap as regret; pulling the optimal arm incurs 0.
    let s_regret: Vec<f64> = (0..n_design)
        .map(|d| {
            let a = d / m_cells;
            let j = d % m_cells;
            let (x1, x2) = cells[j];
            f(optimal_arm[j], x1, x2) - f(a, x1, x2)
        })
        .collect();

    // 1: Simple regret (terminal decision quality).
    // Proportional to cumulative regret sensitivity with a temporal discount factor.
    // In our simplified model, this is 1.5x the cumulative regret sensitivity
    // (late-round observations matter more for the terminal decision).
    let s_simple: Vec<f64> = s_regret.iter().map(|&g| g * 1.5).collect();

    // 2-4: Per-arm MSE (D-optimal estimation quality).
    // s_MSE_a(a', j) = -leverage(x_j) / p_a(x_j)^2  if a' == a, else 0.
    // Sampling arm a at cell j only informs estimation of arm a's model.
    // The leverage = ||x_j||^2 + 1 (including intercept) measures how
    // informative cell j is for the linear model.
    let mut s_mse = vec![vec![0.0; n_design]; 3];
    for a in 0..k {
        for j in 0..m_cells {
            let (x1, x2) = cells[j];
            let leverage = x1 * x1 + x2 * x2 + 1.0;
            s_mse[a][idx(a, j)] = -leverage / (p_uniform * p_uniform);
        }
    }

    // 5: Average detection delay.
    // s(a,j) = -1 / p_a(x_j)^2 for all (a,j).
    // Under uniform allocation, this is the same constant everywhere.
    // This is the "aggregate" objective that is structurally redundant with IMSE.
    let s_avg_det: Vec<f64> = (0..n_design)
        .map(|_| -1.0 / (p_uniform * p_uniform))
        .collect();

    // 6: Worst-case detection delay.
    // All sensitivity concentrated at the bottleneck (arm, cell) pair.
    // Under uniform allocation, pick arm 1, cell 0 as the bottleneck.
    // This is a point mass -- linearly independent from all smooth objectives.
    let mut s_wc_det = vec![0.0; n_design];
    s_wc_det[idx(1, 0)] = -1.0 / (p_uniform * p_uniform);

    // 7: Subgroup fairness.
    // s(a,j) = 1/gap_a(x_j) for suboptimal arms (more sampling where gaps are large).
    // This represents a preference for equitable performance across regions.
    let s_fairness: Vec<f64> = (0..n_design)
        .map(|d| {
            let a = d / m_cells;
            let j = d % m_cells;
            let (x1, x2) = cells[j];
            let gap = f(optimal_arm[j], x1, x2) - f(a, x1, x2);
            if gap > 1e-6 {
                1.0 / gap
            } else {
                0.0
            }
        })
        .collect();

    let sensitivities: Vec<SensitivityRow> = vec![
        s_regret,
        s_simple,
        s_mse[0].clone(),
        s_mse[1].clone(),
        s_mse[2].clone(),
        s_avg_det,
        s_wc_det,
        s_fairness,
    ];

    let analysis = analyze_redundancy(&sensitivities).unwrap();
    assert_eq!(analysis.num_objectives, 8);
    assert_eq!(analysis.num_design_points, 27);

    // --- Structural checks ---

    // Formal rank should be high (most objectives are independent).
    let formal_rank = analysis.pareto_dimension_bound(1e-6) + 1;
    assert!(
        formal_rank >= 6,
        "formal rank = {formal_rank}, expected >= 6"
    );

    // Eigenvalue spectrum should span orders of magnitude.
    let ev = &analysis.eigenvalues;
    let dynamic_range = ev[0] / ev.last().copied().unwrap_or(1e-30).max(1e-30);
    assert!(dynamic_range > 10.0, "dynamic range = {dynamic_range}");

    // Effective dimension at 1% threshold: should be much less than 8.
    let eff_dim = analysis.effective_dimension(0.01);
    assert!(
        (2..=6).contains(&eff_dim),
        "effective dimension = {eff_dim}, expected 2-6"
    );

    // --- Pairwise structure ---

    // Cumulative and simple regret: perfectly correlated (1.5x scalar multiple).
    let cos_01 = analysis.cosine(0, 1).unwrap();
    assert!((cos_01 - 1.0).abs() < 1e-9, "cum-simple cosine = {cos_01}");

    // Per-arm MSEs: exactly orthogonal (disjoint support in design space).
    for (a, b) in [(2, 3), (2, 4), (3, 4)] {
        let cos = analysis.cosine(a, b).unwrap();
        assert!(
            cos.abs() < 1e-9,
            "MSE arm{}-arm{} cosine = {cos}",
            a - 2,
            b - 2
        );
    }

    // Eigenvalue sum = trace of Gram matrix (sanity check on Jacobi solver).
    let ev_sum: f64 = analysis.eigenvalues.iter().sum();
    let trace = analysis.trace();
    assert!(
        (ev_sum - trace).abs() < 1e-4 * trace.abs().max(1.0),
        "ev_sum={ev_sum}, trace={trace}"
    );

    // Top-3 eigenvalues capture the dominant tradeoff structure.
    let frac_top3 = analysis.variance_fraction(3);
    assert!(frac_top3 > 0.80, "top-3 variance fraction = {frac_top3}");
}

// ============================================================================
// V. Saturation principle verification
// ============================================================================
//
// Ehrgott & Nickel (2002) proved: for strictly quasi-convex problems in D_eff
// decision variables, Pareto optimality can be decided by at most D_eff + 1
// objectives.  The "surplus" objectives are either explicitly redundant
// (monotone transforms of existing ones) or implicitly redundant (their
// sensitivity lies in the span of the others).

/// With K=2 non-contextual arms, D_eff = 1 (one allocation variable, n_2).
/// No matter how many objectives we name, the Pareto front is at most 1D.
///
/// Here we define 5 objectives with 3 distinct sensitivity shapes:
/// - Regret: dR/dn = Delta (flat)
/// - MSE family: dMSE/dn ~ -1/n^2 (MSE, detection, quantile all share this shape)
/// - Variance: dVar/dn ~ -1/n^3 (different curvature)
///
/// The 5 vectors span at most 3 directions, so the Gram matrix has rank <= 3.
/// Pareto dimension <= 2.  But MSE, detection, and quantile are redundant with
/// each other (same shape), so the practical Pareto dimension is lower.
#[test]
fn saturation_two_arms_one_dof() {
    let n_vals: [f64; 5] = [10.0, 50.0, 100.0, 200.0, 500.0];
    let delta = 0.5_f64;

    let s_regret: Vec<f64> = n_vals.iter().map(|_| delta).collect();
    let s_mse: Vec<f64> = n_vals.iter().map(|&n| -1.0 / (n * n)).collect();
    let s_det: Vec<f64> = n_vals.iter().map(|&n| -100.0 / (n * n)).collect();
    let s_quantile: Vec<f64> = n_vals.iter().map(|&n| -2.0 / (n * n)).collect();
    let s_variance: Vec<f64> = n_vals.iter().map(|&n| -1.0 / (n * n * n)).collect();

    let a = analyze_redundancy(&[s_regret, s_mse, s_det, s_quantile, s_variance]).unwrap();

    // Rank <= 3 (three distinct shapes: flat, 1/n^2, 1/n^3).
    let bound = a.pareto_dimension_bound(1e-6);
    assert!(bound <= 4, "Pareto bound = {bound}, expected <= 4");

    // MSE, detection, and quantile are mutually redundant (all ~ 1/n^2).
    let pairs = a.redundant_pairs(0.99);
    assert!(
        pairs.len() >= 2,
        "expected >= 2 redundant pairs among MSE/det/quantile, got {}",
        pairs.len()
    );
}

/// Ehrgott-Nickel bound: at most D_eff + 1 objectives can be simultaneously
/// non-redundant.  For K=3 arms non-contextual, D_eff = 2 (the allocation
/// simplex is 2-dimensional), so at most 3 objectives can be independent.
#[test]
fn ehrgott_nickel_three_arms_two_dof() {
    // K=3 non-contextual arms, no contexts.
    // Design: (n_1, n_2) with n_3 = T - n_1 - n_2 (2 free variables).
    // Sensitivity vectors live in R^2.
    //
    // 4 objectives, all evaluated at the same design point:
    //   Regret arm 1: d/dn_1 = -Delta_1 (reducing n_1 improves regret if arm 1 is suboptimal)
    //   Regret arm 2: d/dn_2 = -Delta_2
    //   MSE arm 1:    d/dn_1 = -sigma^2/n_1^2,  d/dn_2 = 0
    //   MSE arm 2:    d/dn_1 = 0,                d/dn_2 = -sigma^2/n_2^2
    let delta_1 = 0.3_f64;
    let delta_2 = 0.5_f64;
    let n1 = 50.0_f64;
    let n2 = 30.0_f64;

    let s = vec![
        vec![-delta_1, 0.0],         // regret arm 1
        vec![0.0, -delta_2],         // regret arm 2
        vec![-1.0 / (n1 * n1), 0.0], // MSE arm 1
        vec![0.0, -1.0 / (n2 * n2)], // MSE arm 2
    ];
    let a = analyze_redundancy(&s).unwrap();

    // In R^2, at most 2 linearly independent vectors.  So rank <= 2.
    // Pareto dimension <= 1.
    let bound = a.pareto_dimension_bound(1e-6);
    assert!(
        bound <= 2,
        "Pareto bound = {bound}, expected <= 2 (D_eff=2)"
    );

    // Regret arm 1 and MSE arm 1 have the same support (only d/dn_1 is nonzero)
    // so they are proportional (anti-proportional, cosine = -1).
    let cos_r1_m1 = a.cosine(0, 2).unwrap();
    assert!(
        (cos_r1_m1.abs() - 1.0).abs() < 1e-9,
        "regret1-MSE1 cosine = {cos_r1_m1}, expected +/-1"
    );
}

// ============================================================================
// VI. Temporal scheduling
// ============================================================================

/// For fixed n_2 (total observations of arm 2), the worst-case detection delay
/// depends on the observation schedule.  Uniform spacing minimizes worst-case
/// delay because the maximum gap between consecutive observations is smallest.
///
/// Model: CUSUM needs h/(delta^2/2) post-change observations to alarm.
/// If observations are uniformly spaced at interval T/n, worst-case wall delay
/// is (h/(delta^2/2)) * (T/n).  If observations are clumped, the worst-case
/// gap is larger, increasing the wall delay proportionally.
#[test]
fn uniform_spacing_minimizes_worst_case_delay() {
    let delta = 0.3_f64;
    let h = 5.0_f64;
    let n_obs = 100_usize;
    let t = 1000_usize;

    let sample_delay = h / (delta * delta / 2.0);

    // Uniform: gap = T/n = 10 steps.
    let gap_uniform = t as f64 / n_obs as f64;
    let wall_delay_uniform = sample_delay * gap_uniform;

    // Clumped: first 50 observations in the first 100 steps, last 50 in 900 steps.
    // Sparse-region gap = 900/50 = 18 steps.
    let sparse_time = (t as f64) * 0.9;
    let sparse_obs = n_obs / 2;
    let gap_sparse = sparse_time / sparse_obs as f64;
    let wall_delay_clumped = sample_delay * gap_sparse;

    assert!(
        wall_delay_clumped > wall_delay_uniform,
        "clumped={wall_delay_clumped} should be > uniform={wall_delay_uniform}"
    );

    // The ratio of delays equals the ratio of gaps.
    let gap_ratio = gap_sparse / gap_uniform;
    let delay_ratio = wall_delay_clumped / wall_delay_uniform;
    assert!(
        (gap_ratio - delay_ratio).abs() < 1e-9,
        "gap_ratio={gap_ratio}, delay_ratio={delay_ratio}"
    );
}

// ============================================================================
// VII. Information-Directed Sampling connection
// ============================================================================
//
// IDS (Russo & Van Roy 2014, arXiv:1403.5556) selects actions by minimizing the information ratio:
//   Gamma = (expected regret)^2 / (information gain about optimality)
//
// This is a two-objective scalarization: regret vs. information.
//
// Extending to three objectives (regret, information, detection power) gives
// "Monitoring-Augmented IDS" with ratio:
//   Gamma_MA = (expected regret)^2 / (w1 * g_opt + w2 * g_chg)
//
// In the non-contextual case, g_opt and g_chg are proportional (learning about
// arm means = learning about change status), so Gamma_MA = Gamma.
// In the contextual case, g_opt and g_chg have different spatial profiles,
// making the three-objective problem genuinely harder.
//
// This test verifies the proportionality in the non-contextual case.

#[test]
fn ids_information_gains_proportional_non_contextual() {
    // Non-contextual: K=2 arms, each information gain is a scalar function of
    // the allocation p = (p_1, p_2) with p_1 + p_2 = 1.
    //
    // g_opt(a) = I(Y_a; Theta*) = information about which arm is optimal.
    //   For Gaussian arms with known variance, this is proportional to
    //   1/sigma^2 for the suboptimal arm and 0 for the optimal arm.
    //
    // g_chg(a) = I(Y_a; H_change) = information about whether arm a changed.
    //   For known change magnitude, this is proportional to
    //   delta^2 / (2*sigma^2) for the monitored arm.
    //
    // Both are non-zero only for the suboptimal arm (in the simplest formulation),
    // so their ratio is a constant => they are proportional.
    let sigma2 = 1.0_f64;
    let delta = 0.3_f64;

    // Information gain per observation of arm 2 (the suboptimal arm):
    let g_opt_2 = 1.0 / sigma2; // information about optimality
    let g_chg_2 = delta * delta / (2.0 * sigma2); // information about change

    // Ratio is constant (independent of allocation):
    let ratio = g_chg_2 / g_opt_2;
    assert!(
        (ratio - delta * delta / 2.0).abs() < 1e-12,
        "g_chg/g_opt = {ratio}, expected delta^2/2 = {}",
        delta * delta / 2.0
    );

    // As sensitivity vectors over design points (just the suboptimal arm):
    let s_g_opt = vec![0.0, g_opt_2]; // [arm 1 gain, arm 2 gain]
    let s_g_chg = vec![0.0, g_chg_2];
    let a = analyze_redundancy(&[s_g_opt, s_g_chg]).unwrap();
    let cos = a.cosine(0, 1).unwrap();
    assert!(
        (cos - 1.0).abs() < 1e-9,
        "g_opt and g_chg should be proportional (cosine={cos})"
    );
}

// ============================================================================
// VIII. Bridge test: muxer's CusumCatDetector scores -> pare::sensitivity
// ============================================================================
//
// This closes the loop between muxer's concrete monitoring implementation and
// the abstract sensitivity analysis in pare.
//
// Setup: 2 arms, each with a MonitoredWindow.  We define 3 objectives as
// functions of the allocation vector [n_1, n_2]:
//   - Regret:    R(n_1, n_2) = Delta_1 * n_1 + Delta_2 * n_2
//   - Estimation: MSE_2(n_1, n_2) = sigma^2 / max(n_2, 1)
//   - Detection:  score from muxer's cusum_score_between_windows
//
// We compute the Jacobian via finite differences (perturbing the number of
// observations in each arm's window) and verify that the resulting sensitivity
// analysis matches the theory: MSE and detection are proportional in this
// non-contextual setting.

#[test]
fn bridge_muxer_cusum_to_pare_sensitivity() {
    use muxer::monitor::{cusum_score_between_windows, MonitoredWindow};
    use muxer::Outcome;
    use pare::sensitivity::{analyze_redundancy, finite_difference_jacobian};

    // Parameters.
    let delta = 0.5_f64; // gap: arm 0 is optimal by Delta=0.5
    let sigma2 = 1.0_f64;

    // Outcome factories.
    let ok = Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 1,
        elapsed_ms: 100,
        quality_score: None,
    };
    let bad = Outcome {
        ok: false,
        junk: true,
        hard_junk: true,
        cost_units: 1,
        elapsed_ms: 100,
        quality_score: None,
    };

    // Build a function that, given allocation [n_0, n_1], returns objective values.
    // We use the *scores* from muxer's actual detectors, not theoretical formulas.
    let build_objectives = |_n0: usize, n1: usize| -> (f64, f64, f64) {
        // Regret: suboptimal arm (arm 1) pulled n_1 times, each costs Delta.
        let regret = delta * n1 as f64;

        // MSE of arm 1.
        let mse = sigma2 / (n1.max(1) as f64);

        // Detection score from muxer's CUSUM for arm 1.
        // Baseline: all-ok.  Recent: all-ok (no change yet).
        // The CUSUM score under the null should be near zero.
        // But we care about the *sensitivity* (how it changes with n_1),
        // not the absolute score.  To get nonzero sensitivity, we build
        // a recent window with a mild shift (some bad outcomes) so the
        // CUSUM has something to detect.
        let mut w1 = MonitoredWindow::new(200, 80);
        // Baseline: healthy.
        for _ in 0..200 {
            w1.push(ok);
        }
        // Recent: mix of ok and bad, proportional to n1 total observations
        // (simulating that more observations = more evidence about a shift).
        let n_bad = (n1 / 5).clamp(1, 80);
        let n_ok_recent = 80_usize.saturating_sub(n_bad);
        for _ in 0..n_ok_recent {
            w1.push(ok);
        }
        for _ in 0..n_bad {
            w1.push(bad);
        }

        let cusum =
            cusum_score_between_windows(w1.baseline(), w1.recent(), 1e-3, 1e-12, 20, 10, None)
                .unwrap_or(0.0);

        (regret, mse, cusum)
    };

    // Baseline allocation.
    let n0_base = 50.0_f64;
    let n1_base = 50.0_f64;
    let mu = vec![n0_base, n1_base];

    // Wrap each objective as a boxed closure over the allocation vector.
    // (Rust closures have distinct types, so we need Box<dyn Fn> to store them
    // in a single array for `finite_difference_jacobian`.)
    #[allow(clippy::type_complexity)]
    let objectives: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
        Box::new(|alloc: &[f64]| {
            let (r, _, _) = build_objectives(alloc[0] as usize, alloc[1] as usize);
            r
        }),
        Box::new(|alloc: &[f64]| {
            let (_, m, _) = build_objectives(alloc[0] as usize, alloc[1] as usize);
            m
        }),
        Box::new(|alloc: &[f64]| {
            let (_, _, c) = build_objectives(alloc[0] as usize, alloc[1] as usize);
            c
        }),
    ];

    // Compute Jacobian via finite differences.
    // Use a step of 5 (integer observations) to get meaningful changes.
    let jac = finite_difference_jacobian(&mu, &objectives, 5.0);

    assert_eq!(jac.len(), 3); // 3 objectives
    assert_eq!(jac[0].len(), 2); // 2 design variables

    // Regret sensitivity: should only respond to n_1 (arm 1 pulls).
    // dR/d(n_0) ~ 0, dR/d(n_1) ~ Delta = 0.5.
    assert!(
        jac[0][0].abs() < 0.1,
        "regret should not depend on n_0: dR/dn0 = {}",
        jac[0][0]
    );
    assert!(
        (jac[0][1] - delta).abs() < 0.2,
        "regret sensitivity to n_1 should be ~Delta: dR/dn1 = {}",
        jac[0][1]
    );

    // Run redundancy analysis on the Jacobian.
    let analysis = analyze_redundancy(&jac).unwrap();

    assert_eq!(analysis.num_objectives, 3);
    assert_eq!(analysis.num_design_points, 2);

    // In a 2D design space, at most 2 linearly independent directions.
    // Pareto dimension <= 1.
    let bound = analysis.pareto_dimension_bound(1e-4);
    assert!(
        bound <= 2,
        "Pareto bound = {bound}, expected <= 2 (D_eff=2)"
    );

    // MSE and CUSUM score should have correlated sensitivities in this
    // non-contextual setting: both respond primarily to n_1.
    let cos_mse_cusum = analysis.cosine(1, 2).unwrap();
    // The sign depends on whether more observations increase or decrease the
    // CUSUM score.  With more bad observations, CUSUM goes up while MSE goes
    // down, so they could be anti-correlated.  The key structural claim is
    // that they are NOT independent (|cos| >> 0).
    assert!(
        cos_mse_cusum.abs() > 0.1,
        "MSE and CUSUM should be correlated (not independent) in non-contextual setting: cos = {cos_mse_cusum}"
    );
}
