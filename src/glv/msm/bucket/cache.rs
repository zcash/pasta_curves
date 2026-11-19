//! Reuse of window scratch across MSM calls.
//!
//! Scratch is checked out for the duration of a task and returned when its
//! [`Checkout`] drops. With `multicore`, each thread keeps one scratch per
//! curve between calls, so repeated MSMs on a Rayon worker reuse its buffers.
//! Without `multicore`, a checkout is fresh scratch that is dropped after the
//! call, and this module needs no `std`.

use core::ops::{Deref, DerefMut};

use super::WindowReducer;
use crate::arithmetic::{CurveAffine, VartimeField};

// The crate otherwise stays no_std; only the per-thread slots need it.
#[cfg(feature = "multicore")]
extern crate std;
#[cfg(feature = "multicore")]
use core::cell::RefCell;

/// The largest scratch kept on a thread between calls. A prover's occasional
/// huge MSM should not pin its buffers on every worker for the process's
/// lifetime, so larger scratch is dropped when it is returned.
#[cfg(feature = "multicore")]
pub(super) const MAX_RETAINED_BYTES: usize = 8 << 20;

/// A curve whose window scratch can be cached, given by its affine type. Each
/// curve has its own typed thread-local slot; the affine types of Pallas and
/// Vesta are the only implementors.
pub(crate) trait CachedAffine: CurveAffine<Base: VartimeField> {
    /// This thread's slot: the scratch kept between calls, if any.
    #[cfg(feature = "multicore")]
    fn slot() -> &'static std::thread::LocalKey<RefCell<Option<WindowReducer<Self>>>>;
}

macro_rules! cached_affine {
    ($affine:ty) => {
        impl CachedAffine for $affine {
            #[cfg(feature = "multicore")]
            fn slot() -> &'static std::thread::LocalKey<RefCell<Option<WindowReducer<Self>>>> {
                std::thread_local! {
                    static SLOT: RefCell<Option<WindowReducer<$affine>>> =
                        const { RefCell::new(None) };
                }
                &SLOT
            }
        }
    };
}

cached_affine!(crate::pallas::Affine);
cached_affine!(crate::vesta::Affine);

/// Scratch checked out of this thread's slot.
///
/// While it is checked out, the slot is empty: nested work that runs on this
/// thread meanwhile, such as a Rayon job stolen while this one waits, takes
/// fresh scratch instead of these buffers. Ownership makes that independence
/// structural rather than a protocol callers must follow.
pub(super) struct Checkout<A>
where
    A: CachedAffine,
{
    // `None` only during `drop`, which moves the scratch back to the slot.
    scratch: Option<WindowReducer<A>>,
}

impl<A> Checkout<A>
where
    A: CachedAffine,
{
    /// Takes this thread's cached scratch, or fresh scratch if the slot is
    /// empty or `multicore` is disabled.
    pub(super) fn take() -> Self {
        #[cfg(feature = "multicore")]
        let scratch = A::slot()
            .try_with(|slot| slot.borrow_mut().take())
            .ok()
            .flatten()
            .unwrap_or_default();
        #[cfg(not(feature = "multicore"))]
        let scratch = WindowReducer::default();
        Self {
            scratch: Some(scratch),
        }
    }
}

impl<A> Deref for Checkout<A>
where
    A: CachedAffine,
{
    type Target = WindowReducer<A>;

    fn deref(&self) -> &Self::Target {
        self.scratch
            .as_ref()
            .expect("a checkout holds its scratch until dropped")
    }
}

impl<A> DerefMut for Checkout<A>
where
    A: CachedAffine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch
            .as_mut()
            .expect("a checkout holds its scratch until dropped")
    }
}

impl<A> Drop for Checkout<A>
where
    A: CachedAffine,
{
    /// Returns the scratch to this thread's slot unless it grew past
    /// `MAX_RETAINED_BYTES`. Nested work may already have returned other
    /// scratch for this field; keeping one per field, not one per nesting
    /// level, bounds the cache. During thread teardown the slot may be gone,
    /// and the scratch is simply dropped.
    fn drop(&mut self) {
        #[cfg(feature = "multicore")]
        if let Some(scratch) = self.scratch.take() {
            if scratch.capacity_bytes() <= MAX_RETAINED_BYTES {
                let _ = A::slot().try_with(|slot| *slot.borrow_mut() = Some(scratch));
            }
        }
    }
}
