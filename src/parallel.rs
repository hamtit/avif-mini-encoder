//! A minimal scoped thread helper (no external dependencies).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// `(0..n).map(f).collect()`, spread over up to `threads` threads. Indices are
/// handed out in order, so the result is identical to the sequential map.
pub fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, n.max(1));
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let v = f(i);
                    results.lock().unwrap()[i] = Some(v);
                }
            });
        }
    });
    results.into_inner().unwrap().into_iter().map(|v| v.expect("missing result")).collect()
}
