use muxer::TrialRng;

#[test]
fn version_one_pins_raw_bits_and_restores_after_mixed_draws() {
    let mut rng = TrialRng::seeded(0);
    assert_eq!(rng.state().version(), 1);
    assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    rng.unit_f64();
    rng.below(17);
    let state = rng.state();
    assert_eq!(rng.state(), state);
    let mut restored = TrialRng::from_state(state).unwrap();
    for _ in 0..100 {
        assert_eq!(rng.next_u64(), restored.next_u64());
        assert_eq!(rng.unit_f64(), restored.unit_f64());
        assert_eq!(rng.below(7), restored.below(7));
    }
}

#[cfg(any(feature = "stochastic", feature = "contextual"))]
#[test]
fn rand_core_draw_consumption_and_byte_order_are_versioned() {
    use rand::RngCore;
    let mut rng = TrialRng::seeded(0);
    assert_eq!(rng.next_u32(), 0x7b1d_cdaf);
    let mut bytes = [0; 10];
    rng.fill_bytes(&mut bytes);
    assert_eq!(
        bytes,
        [0xf4, 0x65, 0xb9, 0xa1, 0x6a, 0x9e, 0x78, 0x6e, 0x4f, 0x45]
    );
    let mut direct = TrialRng::seeded(0);
    for _ in 0..3 {
        direct.next_u64();
    }
    assert_eq!(rng.state(), direct.state());
    rng.fill_bytes(&mut []);
    assert_eq!(rng.state(), direct.state());
    assert_eq!(RngCore::next_u64(&mut rng), direct.next_u64());
}

#[cfg(feature = "serde")]
#[test]
fn serialized_state_is_strict_and_continues_the_stream() {
    use muxer::TrialRngState;
    let mut rng = TrialRng::seeded(u64::MAX);
    rng.next_u64();
    let encoded = serde_json::to_string(&rng.state()).unwrap();
    let state: TrialRngState = serde_json::from_str(&encoded).unwrap();
    let mut restored = TrialRng::from_state(state).unwrap();
    assert_eq!(rng.next_u64(), restored.next_u64());

    let unknown: TrialRngState = serde_json::from_str(r#"{"version":2,"state":0}"#).unwrap();
    assert!(TrialRng::from_state(unknown).is_err());
    for malformed in [
        r#"{"state":0}"#,
        r#"{"version":1}"#,
        r#"{"version":1,"state":0,"extra":true}"#,
        r#"{"version":1,"version":1,"state":0}"#,
        r#"{"version":1,"state":-1}"#,
    ] {
        assert!(serde_json::from_str::<TrialRngState>(malformed).is_err());
    }
}
