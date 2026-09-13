//! Contextual LinUCB keeps the decision-time features for delayed feedback.

use muxer::{BoundedReward, ContextualMode, ContextualProfile, LinUcbConfig, Muxer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let actions = vec!["fast".to_owned(), "careful".to_owned()];
    let profile = ContextualProfile::new(LinUcbConfig {
        dim: 2,
        ..LinUcbConfig::default()
    })
    .with_mode(ContextualMode::Softmax { temperature: 0.4 })?;
    let mut muxer = Muxer::new(actions, profile)?;

    let request_features = [0.2, 0.9];
    let decision = muxer.decide(&request_features)?;
    muxer.tell(decision.id(), BoundedReward::new(0.8)?)?;
    Ok(())
}
