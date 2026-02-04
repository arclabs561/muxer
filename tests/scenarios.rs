use muxer::{select_mab_explain, MabConfig, Outcome, StickyConfig, StickyMab, Window};
use std::collections::BTreeMap;

fn push_n(w: &mut Window, n: usize, o: Outcome) {
    for _ in 0..n {
        w.push(o);
    }
}

#[test]
fn select_mab_prefers_non_hard_junk_arm_when_window_is_hard_junk_heavy() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // Preload realistic windows instead of hand-constructing Summary rows.
    let mut wa = Window::new(50);
    let mut wb = Window::new(50);

    // Arm "a" is currently producing lots of hard junk (operational failures).
    push_n(
        &mut wa,
        50,
        Outcome {
            ok: false,
            junk: true,
            hard_junk: true,
            cost_units: 1,
            elapsed_ms: 100,
        },
    );

    // Arm "b" is healthy.
    push_n(
        &mut wb,
        50,
        Outcome {
            ok: true,
            junk: false,
            hard_junk: false,
            cost_units: 1,
            elapsed_ms: 120,
        },
    );

    let summaries = BTreeMap::from([
        ("a".to_string(), wa.summary()),
        ("b".to_string(), wb.summary()),
    ]);

    let cfg = MabConfig {
        max_hard_junk_rate: Some(0.2),
        ..MabConfig::default()
    };

    let d = select_mab_explain(&arms, &summaries, cfg);
    assert_eq!(d.selection.chosen, "b");
}

#[test]
fn sticky_reduces_switching_under_alternating_small_advantages() {
    let arms = vec!["a".to_string(), "b".to_string()];

    // No dwell limit: only margin gate.
    let mut sticky = StickyMab::new(StickyConfig {
        min_dwell: 0,
        min_switch_margin: 0.2,
    });

    let cfg = MabConfig::default();

    // Alternate a tiny advantage every step (base policy will tend to flip).
    let mut base_switches = 0u64;
    let mut sticky_switches = 0u64;
    let mut prev_base: Option<String> = None;
    let mut prev_sticky: Option<String> = None;

    for t in 0..100u64 {
        let (ok_a, ok_b) = if t % 2 == 0 { (51, 50) } else { (50, 51) };

        let mut wa = Window::new(60);
        let mut wb = Window::new(60);
        push_n(
            &mut wa,
            60,
            Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 1,
                elapsed_ms: 100,
            },
        );
        push_n(
            &mut wb,
            60,
            Outcome {
                ok: true,
                junk: false,
                hard_junk: false,
                cost_units: 1,
                elapsed_ms: 100,
            },
        );

        // Nudge success rates via counts (simulating a “near-tie” race).
        // (This is intentionally synthetic but deterministic.)
        let sa = {
            let mut s = wa.summary();
            s.calls = 100;
            s.ok = ok_a;
            s
        };
        let sb = {
            let mut s = wb.summary();
            s.calls = 100;
            s.ok = ok_b;
            s
        };

        let summaries = BTreeMap::from([("a".to_string(), sa), ("b".to_string(), sb)]);
        let base = select_mab_explain(&arms, &summaries, cfg);
        let chosen_base = base.selection.chosen.clone();
        let chosen_sticky = sticky.apply_mab(base).selection.chosen;

        if let Some(prev) = &prev_base {
            if prev != &chosen_base {
                base_switches += 1;
            }
        }
        if let Some(prev) = &prev_sticky {
            if prev != &chosen_sticky {
                sticky_switches += 1;
            }
        }
        prev_base = Some(chosen_base);
        prev_sticky = Some(chosen_sticky);
    }

    assert!(
        sticky_switches < base_switches,
        "sticky_switches={} base_switches={}",
        sticky_switches,
        base_switches
    );
}
