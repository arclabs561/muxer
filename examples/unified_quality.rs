//! A complete quality-routing loop with delayed scoring.

use muxer::{Muxer, Outcome, QualityFeedback, RouterConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let actions = vec!["fast".to_owned(), "careful".to_owned()];
    let mut muxer = Muxer::quality(actions)
        .config(RouterConfig::default())
        .with_delayed_score()
        .build()?;

    // Readiness remains application-owned. Every selection stage uses this
    // authoritative subset, in catalogue order.
    let ready = vec!["fast".to_owned(), "careful".to_owned()];
    let decision = muxer.decide_from(&ready, &[])?;
    let action = decision.action().to_owned();

    // Execute the chosen action in the application, then report its final
    // categorical result. The decision remains open for its declared score.
    let execution = Outcome::success(2, 35);
    muxer.tell(decision.id(), QualityFeedback::execution(execution))?;

    // An external evaluator can finish later. Its score joins the same
    // execution row; it is not another observation or detector sample.
    muxer.tell(decision.id(), QualityFeedback::score(0.92)?)?;

    println!("selected {action}");
    Ok(())
}
