//! Bernoulli Thompson checkpoint handoff checks.
//!
//! The suite needs `serde` and `stochastic`. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "serde", feature = "stochastic"))]
#[path = "bernoulli_muxer_checkpoint/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "serde", feature = "stochastic")))]
#[test]
fn bernoulli_checkpoint_suite_is_not_silently_empty() {
    // Always-true invariant: the tie-break hash is deterministic for a seed and
    // separates distinct action labels, which is what seeded sampling relies on.
    let first = muxer::stable_hash64(7, "a");
    assert_eq!(
        first,
        muxer::stable_hash64(7, "a"),
        "tie-break hash must be deterministic for one seed"
    );
    assert_ne!(
        first,
        muxer::stable_hash64(7, "b"),
        "distinct actions must hash apart for one seed"
    );

    eprintln!(
        "tests/bernoulli_muxer_checkpoint.rs: enable features serde,stochastic to run the gated checkpoint suite"
    );
}
