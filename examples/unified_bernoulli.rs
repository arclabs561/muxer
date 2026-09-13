//! A complete boolean-reward decision loop.

use muxer::Muxer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut muxer = Muxer::bernoulli(["small", "large"])?;
    let decision = muxer.decide(&())?;

    let accepted = decision.action() == "small";
    muxer.tell(decision.id(), accepted)?;
    Ok(())
}
