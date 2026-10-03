//! Opt-in requested-heap sizing for retained open runtime tickets.
//!
//! This reports allocator-requested bytes, not allocator-rounded heap use or
//! resident memory. It keeps issue-time allocation traffic separate from the
//! net live allocation delta that remains after returned receipts are dropped.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use muxer::{Muxer, Outcome, QualityFeedback, QualityProfile, RuntimeConfig};

#[cfg(feature = "stochastic")]
use muxer::{BernoulliThompson, ThompsonConfig};
#[cfg(feature = "contextual")]
use muxer::{BoundedReward, ContextualProfile, LinUcbConfig};

struct CountingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static FREED_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: This wrapper delegates every allocation operation to `System` and
// records only the layouts supplied by `GlobalAlloc` callers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` comes from the `GlobalAlloc::alloc` caller and is
        // therefore valid for the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` comes from the `GlobalAlloc::alloc_zeroed` caller
        // and is therefore valid for the system allocator.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_deallocation(layout.size());
        // SAFETY: the caller upholds the `GlobalAlloc::dealloc` contract for
        // `pointer` and `layout`; forwarding preserves that contract.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller upholds the `GlobalAlloc::realloc` contract for
        // `pointer`, `layout`, and `new_size`; forwarding preserves it.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            REALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            record_deallocation(layout.size());
            record_allocation(new_size);
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_allocation(bytes: usize) {
    ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
}

fn record_deallocation(bytes: usize) {
    FREED_BYTES.fetch_add(bytes, Ordering::Relaxed);
    DEALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    allocated_bytes: usize,
    freed_bytes: usize,
    allocation_calls: usize,
    deallocation_calls: usize,
    reallocation_calls: usize,
}

impl AllocationSnapshot {
    fn now() -> Self {
        Self {
            allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
            freed_bytes: FREED_BYTES.load(Ordering::Relaxed),
            allocation_calls: ALLOCATION_CALLS.load(Ordering::Relaxed),
            deallocation_calls: DEALLOCATION_CALLS.load(Ordering::Relaxed),
            reallocation_calls: REALLOCATION_CALLS.load(Ordering::Relaxed),
        }
    }

    fn live_bytes(self) -> i128 {
        self.allocated_bytes as i128 - self.freed_bytes as i128
    }
}

fn actions() -> Vec<String> {
    (0..5).map(|index| format!("arm{index}")).collect()
}

fn config() -> RuntimeConfig {
    RuntimeConfig::default()
}

fn print_measurement(workload: &str, pending: usize, start: AllocationSnapshot) {
    let end = AllocationSnapshot::now();
    let live_requested_bytes = end.live_bytes() - start.live_bytes();
    let allocated_bytes = end.allocated_bytes - start.allocated_bytes;
    let freed_bytes = end.freed_bytes - start.freed_bytes;
    println!(
        "workload={workload} pending={pending} live_requested_bytes={live_requested_bytes} \
         amortized_live_requested_bytes={:.2} issue_allocated_bytes={allocated_bytes} \
         issue_freed_bytes={freed_bytes} allocation_calls={} deallocation_calls={} \
         reallocation_calls={}",
        live_requested_bytes as f64 / pending as f64,
        end.allocation_calls - start.allocation_calls,
        end.deallocation_calls - start.deallocation_calls,
        end.reallocation_calls - start.reallocation_calls,
    );
}

#[cfg(feature = "stochastic")]
fn run_scalar(pending: usize) {
    let actions = actions();
    let mut runtime = Muxer::with_config(
        actions.clone(),
        BernoulliThompson::with_seed(ThompsonConfig::default(), 17),
        config(),
    )
    .expect("valid scalar runtime");

    for action in &actions {
        let receipt = runtime
            .decide_from(std::slice::from_ref(action), &())
            .unwrap();
        runtime.tell(receipt.id(), true).unwrap();
        drop(receipt);
    }

    let start = AllocationSnapshot::now();
    for _ in 0..pending {
        drop(runtime.decide(&()).unwrap());
    }
    assert_eq!(runtime.pending_len(), pending);
    print_measurement("scalar-bernoulli/actions=5/context=unit", pending, start);
}

#[cfg(feature = "contextual")]
fn run_contextual(pending: usize) {
    let actions = actions();
    let context = [0.125_f64; 8];
    let mut runtime = Muxer::with_config(
        actions.clone(),
        ContextualProfile::new(LinUcbConfig::default()),
        config(),
    )
    .expect("valid contextual runtime");
    let reward = BoundedReward::new(0.5).unwrap();

    for action in &actions {
        let receipt = runtime
            .decide_from(std::slice::from_ref(action), &context)
            .unwrap();
        runtime.tell(receipt.id(), reward).unwrap();
        drop(receipt);
    }

    let start = AllocationSnapshot::now();
    for _ in 0..pending {
        drop(runtime.decide(&context).unwrap());
    }
    assert_eq!(runtime.pending_len(), pending);
    print_measurement("contextual-linucb/actions=5/dim=8", pending, start);
}

fn run_quality(pending: usize) {
    let actions = actions();
    let context = [0.125_f64; 8];
    let profile = QualityProfile::new(actions.clone(), Default::default()).unwrap();
    let mut runtime = Muxer::with_config(actions.clone(), profile, config()).unwrap();

    for action in &actions {
        let receipt = runtime
            .decide_from(std::slice::from_ref(action), &context)
            .unwrap();
        runtime
            .tell(
                receipt.id(),
                QualityFeedback::execution(Outcome::success(3, 80)),
            )
            .unwrap();
        drop(receipt);
    }

    let start = AllocationSnapshot::now();
    for _ in 0..pending {
        drop(runtime.decide(&context).unwrap());
    }
    assert_eq!(runtime.pending_len(), pending);
    print_measurement("quality-immediate/actions=5/context=8", pending, start);
}

fn main() {
    let defaults = config();
    println!(
        "defaults pending_capacity={} terminal_capacity={} event_capacity={} retired_epoch_capacity={}",
        defaults.pending_capacity,
        defaults.terminal_capacity,
        defaults.event_capacity,
        defaults.retired_epoch_capacity,
    );

    for pending in [1, 100, 1_024] {
        #[cfg(feature = "stochastic")]
        run_scalar(pending);
        #[cfg(feature = "contextual")]
        run_contextual(pending);
        run_quality(pending);
    }
}
