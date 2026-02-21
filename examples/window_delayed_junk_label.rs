use muxer::{Outcome, Window};

fn main() {
    // Production pattern: you push an initial outcome when the request returns,
    // then later (after downstream parsing/validation) you may learn it was "junk".
    let mut w = Window::new(5);

    // Request returned OK; we don't yet know junk-ness.
    w.push(Outcome {
        ok: true,
        junk: false,
        hard_junk: false,
        cost_units: 2,
        elapsed_ms: 420,
        quality_score: None,
    });

    // Later, downstream validation decides it was "soft junk".
    w.set_last_junk_level(true, false);

    let s = w.summary();
    eprintln!(
        "calls={} ok_rate={:.3} junk_rate={:.3} hard_junk_rate={:.3} soft_junk_rate={:.3}",
        s.calls,
        s.ok_rate(),
        s.junk_rate(),
        s.hard_junk_rate(),
        s.soft_junk_rate()
    );
}
