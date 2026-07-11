# Design: `muxer-tower` Integration

status: design
date: 2026-07-09
session: 019f488e
grounded-in:
- `.claude/reports/useful-2026-03-15.md`
- `docs/design/post-round-3-roadmap.md`
- Tower `Service` / `Layer` docs

## Problem

`muxer` is useful as a routing primitive, but Rust service code usually wants a
`tower::Layer` or `tower::Service` integration point. Adding Tower directly to
`muxer` would pull async service concerns into a crate that should remain a
small routing library.

The adapter also has a Tower-specific constraint: readiness is part of the
`Service` contract. A design that selects an arm and then calls a backend that
was not made ready by `poll_ready` is wrong, even if the routing choice itself
is correct.

## Context

Tower's core shape is an asynchronous request/response service with a separate
`poll_ready` step before `call`. Middleware composes through `Layer`.

`muxer`'s current core surface is enough for a first adapter:

- `Router::select(k, seed)` selects without mutating observation state.
- `Router::select_from(eligible, k, seed)` makes a caller-computed readiness
  set authoritative for every routing stage.
- `RouterDecision::primary()` returns the ordinary one-backend arm.
- `Router::observe` and `Router::observe_with_context` record outcomes after a
  response or error is known.
- `Router::mode`, `arms`, `summary`, and `summaries` expose inspection without
  private-field access.

The missing concerns belong in the adapter: shared state, seed generation,
request context extraction, backend readiness, elapsed-time measurement, and
mapping service responses/errors into `Outcome`.

## Chosen Approach

Build `muxer-tower` as a separate crate first. The first public surface should
be a route-decision layer, not a full load balancer.

The route-decision layer wraps one downstream dispatcher service. It chooses an
arm with `Router::select`, wraps the request with routing metadata, and passes
that routed request to the downstream service:

```rust
pub struct RoutedRequest<Req> {
    pub arm: String,
    pub decision: muxer::RouterDecision,
    pub context: Vec<f64>,
    pub request: Req,
}

pub struct MuxerRouteLayer<C, O> {
    router: RouterHandle,
    seed: SeedCounter,
    context: C,
    outcome: O,
}
```

The wrapped service implements `Service<RoutedRequest<Req>>`. That service owns
the actual backend registry and therefore owns backend readiness. This keeps the
first adapter honest: Tower readiness remains the dispatcher's responsibility,
while `muxer-tower` proves selection, metadata propagation, and feedback.

`RouterHandle` is a small cloneable wrapper around `Arc<Mutex<Router>>`.
The mutex is held only while selecting or observing. It is never held across an
`.await`. A poisoned lock maps to an adapter error.

Seed generation is deterministic by default:

```rust
pub struct SeedCounter(Arc<AtomicU64>);
```

Each call gets the next counter value. Users that need request-stable seeds can
replace this later with a `SeedPolicy<Req>` trait, but the first pass should not
add that abstraction until a real caller needs it.

Outcome recording is caller-defined:

```rust
pub trait OutcomeClassifier<Response, Error> {
    fn response(&self, arm: &str, elapsed_ms: u64, response: &Response) -> Option<muxer::Outcome>;
    fn error(&self, arm: &str, elapsed_ms: u64, error: &Error) -> Option<muxer::Outcome>;
}

pub trait ContextExtractor<Request> {
    fn context(&self, request: &Request) -> Vec<f64>;
}
```

Returning `None` means "do not observe automatically"; callers can report later
through `RouterHandle::observe` or its `ObservationId`-aware equivalent. The
example classifier can map successful
responses to `Outcome::success(0, elapsed_ms)` and inner errors to
`Outcome::failure(0, elapsed_ms)`, but that default should be documented as an
example policy, not a general truth about response quality.

## Options Considered

### Add Tower Support To `muxer`

Rejected. It would add async service dependencies and runtime-adjacent concepts
to the core crate. The core crate should stay transport-neutral.

### Public `BanditBalanceLayer` First

Deferred. A service that owns many inner services must preserve Tower readiness.
The conservative implementation can poll every backend ready before selecting,
but that makes one unavailable backend block all traffic. A ready-aware
implementation can poll its backends, build the ready arm set, and pass it to
`Router::select_from`.

The prototype may include an experimental `BanditBalanceService` for two mock
services, using the conservative all-ready policy. It should not become the
main public API until real service readiness pressure proves the adapter
semantics.

### Use `tower::steer::Steer`

Deferred. `Steer` is useful prior art for routing among services, but the first
adapter needs to control how decisions are observed back into `Router`. Build
the route-decision layer first; compare against `Steer` when implementing the
convenience balancer.

## Non-Goals

- No `tower`, `http`, `hyper`, `tonic`, or runtime dependency in `muxer`.
- No retry, timeout, health-check, circuit-breaker, or storage policy in the
  first adapter.
- No attempt to infer quality from transport success alone. Outcome semantics
  remain caller-defined.
- No hidden delayed-feedback mechanism. Late quality scoring must go through an
  explicit handle.
- No additional core API changes before the adapter proves another actual gap.

## Implementation Plan

1. Create a sibling `muxer-tower` crate with a path dependency on `../muxer`,
   plus `tower-service` and `tower-layer`. Use `tower` itself only in examples
   or tests if the smaller trait crates are enough.
2. Implement `RouterHandle`, `SeedCounter`, `RoutedRequest`, and
   `MuxerRouteLayer`.
3. Implement `Layer<S>` for `MuxerRouteLayer` and `Service<Req>` for the
   produced route service by:
   - polling the downstream dispatcher readiness normally,
   - selecting the primary arm under the router lock,
   - extracting context before moving the request,
   - calling the downstream dispatcher with `RoutedRequest`,
   - observing a classifier-produced `Outcome` after response/error.
4. Add one deterministic example with two mock backends behind one dispatcher.
   The example should assert that a lower-junk backend receives more traffic
   after observations, and that errors are recorded as failures by the example
   classifier.
5. Only after the route-decision layer works, prototype
   `BanditBalanceService<S>` with an explicit all-ready readiness policy and
   document that it is conservative.

## Decision Gates

- The first published adapter API must not require a Tower dependency in
  `muxer`.
- No lock may be held across an awaited backend future.
- The first example must have deterministic assertions, not just printed output.
- A ready-aware balancer must pass only successfully polled backends to
  `Router::select_from`; it must not route to a backend that was not ready.
- If HTTP request extensions are useful, add them behind an optional `http`
  feature in `muxer-tower`, not in `muxer`.

## Open Questions

- Should `RoutedRequest` carry `String` arm names directly, or should
  `muxer-tower` introduce an `ArmName` newtype and convert at the boundary?
- Should a convenience balancer use `Router::select_from`, or leave all
  readiness-aware dispatch in the downstream service?
- Should automatic observation see request-derived metadata beyond the context
  vector? Defer until the first real caller has that need.

## Review Trigger

Move reusable code back into `muxer` only if it is independent of Tower and
useful to non-service callers. A Tower-specific convenience belongs in
`muxer-tower`.
