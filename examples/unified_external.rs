//! Plug an externally produced score into a feedbackless muxer decision.

use muxer::{ExternalScores, Muxer};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scorer = ExternalScores::new(|latency_budget: &u64, _: &[String]| {
        BTreeMap::from([
            (
                "cheap".to_owned(),
                if *latency_budget < 100 { 0.95 } else { 0.4 },
            ),
            ("accurate".to_owned(), 0.8),
        ])
    });
    let mut muxer = Muxer::new(vec!["cheap".to_owned(), "accurate".to_owned()], scorer)?;
    let decision = muxer.decide(&80)?;
    println!("{}", decision.action());
    Ok(())
}
