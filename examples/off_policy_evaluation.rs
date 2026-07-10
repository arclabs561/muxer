use muxer::{ips_value, self_normalized_ips_value, LoggedReward};

fn main() {
    let mut rows = Vec::new();

    for _ in 0..8 {
        rows.push(LoggedReward {
            reward: 1.0,
            logging_propensity: 0.8,
            target_propensity: 0.5,
        });
    }
    for _ in 0..2 {
        rows.push(LoggedReward {
            reward: 0.0,
            logging_propensity: 0.2,
            target_propensity: 0.5,
        });
    }

    let naive = rows.iter().map(|row| row.reward).sum::<f64>() / rows.len() as f64;
    let ips = ips_value(rows.iter().copied()).unwrap();
    let snips = self_normalized_ips_value(rows).unwrap();

    assert!((naive - 0.8).abs() < 1e-12);
    assert!((ips - 0.5).abs() < 1e-12);
    assert!((snips - 0.5).abs() < 1e-12);

    println!("naive mean: {naive:.3}");
    println!("IPS estimate: {ips:.3}");
    println!("self-normalized IPS estimate: {snips:.3}");
}
