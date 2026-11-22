//! Contextual (LinUCB/LinTS) checkpoint handoff checks.
//!
//! The suite needs `serde` and `contextual`. The gate lives on the module
//! declaration below instead of a file-level `#![cfg(...)]`, so a partial
//! feature set still compiles and runs a real check rather than reporting
//! success having compared nothing.

#[cfg(all(feature = "serde", feature = "contextual"))]
#[path = "contextual_muxer_checkpoint/tests_feature_gated.rs"]
mod tests_feature_gated;

#[cfg(not(all(feature = "serde", feature = "contextual")))]
#[test]
fn contextual_checkpoint_suite_is_not_silently_empty() {
    // Always-true invariant: the tie-break hash is deterministic for a seed and
    // separates distinct action labels, which is what contextual sampling relies
    // on for reproducible ordering.
    let first = muxer::stable_hash64(11, "arm-0");
    assert_eq!(
        first,
        muxer::stable_hash64(11, "arm-0"),
        "tie-break hash must be deterministic for one seed"
    );
    assert_ne!(
        first,
        muxer::stable_hash64(11, "arm-1"),
        "distinct arms must hash apart for one seed"
    );

    eprintln!(
        "tests/contextual_muxer_checkpoint.rs: enable features serde,contextual to run the gated checkpoint suite"
    );
}
