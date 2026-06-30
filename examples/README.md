# muxer examples

Each example answers one routing question and is runnable from the repo root.
Output excerpts below are real, captured from local runs. For derivations,
failure-mode discussion, and longer experiment notes, see
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Start Here

### `getting_started`: what does the basic routing loop do?

Creates three backends, lets the router explore each one, records delayed quality
labels, and shows how `quality_weight` breaks a tie.

```bash
cargo run --example getting_started
```
```text
=== First 6 rounds (explore-first) ===
  round  0: chose gpt-4o           quality=0.92  junk=false  prechosen=["gpt-4o"]
  round  1: chose claude-sonnet    quality=0.78  junk=false  prechosen=["claude-sonnet"]
  round  2: chose gemini-pro       quality=0.55  junk=true  prechosen=["gemini-pro"]

=== After 30 rounds ===
  gpt-4o           calls=10  ok=1.00  junk=0.00  quality=0.92
  claude-sonnet    calls=10  ok=1.00  junk=0.00  quality=0.78
  gemini-pro       calls=10  ok=1.00  junk=1.00  quality=0.55

Best arm now: "gpt-4o"  (mode: Normal)
With quality_weight=1.0: best arm is "gpt-4o"
```

### `guardrail_semantics`: are guardrails hard constraints?

Compares the soft pipeline, where novelty and coverage can pre-pick arms before
guardrails, with the guardrail-first pipeline.

```bash
cargo run --example guardrail_semantics
```
```text
== guardrail semantics demo ==
-- novelty + require_measured --
soft (novelty before guardrail): chosen=["unseen"]
hard (guardrail first, strict): chosen=["seen"], stopped_early=false

-- coverage + require_measured --
soft (coverage before guardrail): chosen=["c"]
hard (guardrail first, strict): chosen=["b"], stopped_early=false
```

## Routing Harnesses

### `matrix_harness`: does coverage touch every slice?

Runs an offline matrix over task, dataset, and backend cells. This is the
smallest applied harness.

```bash
cargo run --example matrix_harness
```
```text
== matrix_harness summary ==
ner.lang=en.dom=news         calls= 36
ner.lang=en.dom=social_media calls= 36
ner.lang=en.dom=biomedical   calls= 36
re.lang=en.dom=wikipedia     calls= 36
coref.lang=en.dom=news       calls= 36
```

### `llm_gateway_harness`: what happens after a model regresses?

Simulates model-gateway traffic with quality, latency, cost, and a prompt-change
drift point.

```bash
cargo run --release --example llm_gateway_harness
```
```text
== llm_gateway_harness ==
480 deterministic requests, drift at round 240: balanced degrades on extraction.

pre-drift aggregate
model           calls  ok     junk   quality  cost/call  ms/call
balanced          108  1.000  0.139    0.831       4.00    278.0
verifier           96  0.990  0.062    0.904      12.00    643.3

post-drift aggregate
model           calls  ok     junk   quality  cost/call  ms/call
balanced           36  0.972  1.000    0.650       4.00    338.9
local-small        57  0.965  0.140    0.796       1.00    137.5
verifier          137  1.000  0.066    0.889      12.00    631.5
```

### `synthetic_drift_harness`: which backend changes after drift?

Uses a controlled drift world where hard slices regress after a known epoch.

```bash
cargo run --example synthetic_drift_harness
```
```text
== synthetic_drift_harness ==
slice coverage:
  family=vision   hardness=easy region=us calls= 36
  family=ranking  hardness=hard region=us calls= 34
hard-slice backend picks pre/post drift:
  balanced_nn        pre= 25 post= 21
  hard_specialist    pre= 18 post= 16
  large_transformer  pre= 17 post=  7
  tiny_rule          pre= 12 post= 20
```

### `contextual_router`: how does LinUCB use context?

Requires the `contextual` feature. The example prints periodic decisions and
ends with the learned split on high-difficulty contexts.

```bash
cargo run --example contextual_router --features contextual
```
```text
t=   0 ctx=[0.17325464426155657, 0.15229643060221798] decision=Decision { policy: LinUcb, chosen: "small", probs: None, notes: [ExploreFirst] } reward=0
t= 200 ctx=[0.5287534261595772, 0.7204979174860174] decision=Decision { policy: LinUcb, chosen: "big", probs: None, notes: [DeterministicChoice] } reward=0
t=1800 ctx=[0.8879784495223955, 0.7420458346578667] decision=Decision { policy: LinUcb, chosen: "big", probs: None, notes: [DeterministicChoice] } reward=1
high-difficulty picks (t>=1000): big=204 small=0
```

## Drift Experiments

### `coverage_autotune`: how does a wall-delay target become coverage?

Requires the `stochastic` feature. It converts a target wall delay into a
coverage floor and checks the empirical detection table.

```bash
cargo run --example coverage_autotune --features stochastic
```
```text
== coverage_autotune ==
K=6 nu=20000 horizon=60000 trials=250
CUSUM: alpha=0.001 min_n=20 thr=12.0 | KL(p1||p0)≈0.045241 => pred_samples h/KL≈265.2
targetW | cov_frac  pred_wall | fa   ok  wrong | wall(mean/p90)  post(mean/p90)  post_rate  mean_frac
   1000 |  0.1667*    1591.5 | 0.004 0.996 0.000 |  1498.5/  2326     250.8/   389     0.167     0.167
   5000 |  0.0530     5000.0 | 0.000 1.000 0.000 |  1016.4/  2332     259.0/   399     0.446     0.170
  20000 |  0.0133    20000.0 | 0.000 1.000 0.000 |  2891.1/  6917     254.0/   396     0.349     0.193
```

### `bqcd_sampling`: what is the cost of focusing without coverage?

Requires the `stochastic` feature. The table compares round-robin, focus by
cumulative drift score, focus by CUSUM score, and CUSUM with coverage.

```bash
cargo run --example bqcd_sampling --features stochastic
```
```text
nu=20000 horizon=40000 alarm=CUSUM(min_n=30,thr=12) alt=[0.05, 0.05, 0.45, 0.45]
policy                              | fa_rate det_rate  wall(mean/p90)  post(mean/p90)  mean_frac_on_chg
round_robin                         |  0.025  0.595   3533.0/ 13038     589.4/  2174            0.167
eps_focus(eps=0.020, catkl, cov(off)) |  0.040  0.565   6366.0/ 17193    2430.1/  7741            0.236
eps_focus(eps=0.020, cusum, cov(off)) |  0.040  0.315   2088.8/  8667      24.1/    57            0.161
eps_focus(eps=0.020, cusum, cov(frac=0.020,floor=10)) |  0.025  0.385   2098.7/  8265      52.5/   176            0.148
```

## More

Other runnable harnesses cover PCAP triage, ad auctions, fraud scoring, search
ranking, medical triage, Thompson sampling, EXP3-IX, monitored routing, sticky
MAB behavior, and delayed junk labels. See `cargo run --example <name>` and
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the longer experiment set.
