# Design: `muxer-tower` Integration

status: proposal
date: 2026-07-09
grounded-in:
- `.claude/reports/useful-2026-03-15.md`
- `docs/design/post-round-3-roadmap.md`

## Problem

`muxer` is useful as a routing primitive, but service teams usually need a
`tower::Layer` or similar integration point to route HTTP or gRPC requests
between backends. Adding tower directly to `muxer` would pull async/service
concerns into a crate that should remain a small primitive.

## Chosen Direction

Prototype tower integration outside the core crate as `muxer-tower` or as a
separate example workspace first. The integration should wrap a `Router`, select
an arm before dispatch, and provide a reporting hook for the eventual `Outcome`.

The core `muxer` crate should not depend on `tower`, `http`, `hyper`, `tonic`,
or a runtime.

## Non-Goals

- No HTTP or gRPC dependencies in `muxer`.
- No dashboard, persistent store, retry policy, or load-balancer framework.
- No attempt to hide delayed feedback. Callers must still report outcomes when
  they know them.
- No production-ready service mesh abstraction in the first pass.

## Minimal Prototype

- A `BanditBalanceLayer` that chooses a backend arm.
- A backend registry keyed by arm name.
- A reporting handle that accepts `Outcome`.
- One runnable example with two mock services and deterministic assertions.

## Gates

- Do not start until `RouterDecision` and `Router::observe` ergonomics are
  stable enough that the layer does not need workaround APIs.
- Keep the first prototype outside `muxer`.
- If the prototype needs storage, metrics sinks, retries, or health checks,
  split those into separate design decisions.

## Review Trigger

Move any reusable core abstraction back into `muxer` only if it is independent
of tower and useful to non-HTTP callers.
