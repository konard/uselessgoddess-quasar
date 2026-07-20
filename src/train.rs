//! Running a model: the loop, the schedule and what it leaves on disk.

pub mod checkpoint;
pub mod schedule;

mod optim;
mod run;
mod ui;

pub use optim::Optim;
pub use run::{Error, Run, run};
pub use schedule::Wsd;
