use muxer::{ObservationId, Outcome, Window};

fn main() {
    // Caller-owned IDs make delayed labels safe when calls overlap or complete
    // out of order.
    let mut w = Window::new(5);
    let first = ObservationId::new(1);
    let second = ObservationId::new(2);

    w.push_with_id(first, Outcome::success(2, 420));
    w.push_with_id(second, Outcome::success(2, 380));

    // The first request's validation arrives after the second request.
    w.set_junk_level_for_id(first, true, false);

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
