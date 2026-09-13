//! Replace a frozen contextual model only after pending feedback is resolved.

#[cfg(feature = "contextual")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use muxer::{BoundedReward, ContextualProfile, LinUcbConfig, Muxer};

    let actions = vec!["compact".to_owned(), "capable".to_owned()];
    let config = LinUcbConfig {
        dim: 3,
        ..LinUcbConfig::default()
    };
    let mut muxer = Muxer::new(actions, ContextualProfile::new(config))?;

    // These are application-owned, offline-produced embedding coordinates.
    let embedding = [0.12, 0.70, 0.31];
    let old = muxer.decide(&embedding)?;

    // A changed embedding basis gets a fresh learner and revision while the
    // old epoch remains available to consume this already-issued ticket.
    muxer.replace_policy_with_revisions(ContextualProfile::new(config), 2, 2)?;
    muxer.tell(old.id(), BoundedReward::new(0.86)?)?;
    let replacement_embedding = [0.08, 0.66, 0.40];
    let _next = muxer.decide(&replacement_embedding)?;
    Ok(())
}

#[cfg(not(feature = "contextual"))]
fn main() {}
