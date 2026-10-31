# Built-in checkpoint core

Status: implemented; process-boundary acceptance and local CI matrix passed

## Problem

Quality now has a complete same-build runtime checkpoint, but scalar and
contextual profiles still have only move-only runtime handoff. Copying the
quality envelope would duplicate its receipt, epoch, retention, finality and
event-history invariants. Those are properties of `Muxer`, not of quality
routing.

## Chosen approach

Extract a private generic wire record and restoration core from the existing
quality implementation. Keep public checkpoint types concrete and opaque.
`QualityMuxerCheckpoint` becomes a transparent wrapper without changing its
version-1 serialized shape; `BernoulliMuxerCheckpoint` supplies the second real
consumer. A private adapter supplies profile state and ticket encoding, channel
contracts, selection metadata rules and grouped pending-ticket validation.
The core continues to own runtime topology, revision rules, retention limits,
global event identity/order and process-local namespace reservation.

Bernoulli continuation requires its Thompson configuration, posterior state
and profile-owned trial RNG. Its legacy posterior snapshot omits configuration
and silently skips invalid rows during restore, so it is not itself a complete
checkpoint. The adapter must validate before invoking that warm-start restore
operation. The hidden legacy kernel RNG does not drive high-level profile
issuance; the explicit profile RNG does. Preserve the original kernel seed too:
`inner().clone()` exposes the kernel's independent seeded continuation. The
high-level profile never advances that kernel-owned stream.
The public policy trait also permits callers to inject a low-level prepared
update or transfer issuance between profiles. Capture must compare the actual
kernel RNG against the stored seed's initial state and reject a mismatch;
otherwise such injection could silently restore a different stream. Preserve
the existing prepared-update API rather than adding a new public wrapper just
to constrain this optional serialization path. Move-only handoff remains valid.

## Options considered

- Separate copied envelopes would avoid private generic bounds but duplicate
  the most safety-sensitive validation and make fixes diverge between profiles.
- A public checkpoint-policy trait would allow arbitrary adapters, but commits
  to an extension contract before external model resolution has been proved.
- The private shared core adds internal generic machinery, but gives two
  concrete consumers one lifecycle implementation without expanding the public
  extension surface.

## Non-goals

- No public generic serialization trait in this pass.
- No external model loader, storage service or distributed writer fence.
- No changes to legacy warm-start normalization or seeded kernel behavior.
- No claim that unsupported profiles become portable merely through extraction.

## Decision gates

- Existing quality-v1 checkpoint bytes must still decode and resume; retain a
  small fixture produced by the pre-extraction implementation.
- All quality lifecycle/corruption tests must continue to exercise the shared
  validator, not a retained parallel implementation.
- Bernoulli must resume pending feedback in a fresh process and match subsequent
  choices, posterior updates, retained epochs, receipts and RNG continuation.
- Malformed posterior data must fail, never silently drop an arm. Profile-kind
  and build mismatches must fail before namespace reservation.
- Preserve current feature boundaries and pass the canonical CI matrix.

Decided: 2026-09-13
