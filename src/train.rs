//! Running a model: the loop, the schedule and what it leaves on disk.

pub mod checkpoint;
pub mod schedule;

mod run;

pub use run::{Error, Run, run};
pub use schedule::Wsd;
