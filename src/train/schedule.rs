//! The learning-rate schedule: warmup, stable, decay.
//!
//! WSD rather than cosine, and for a reason specific to a run that shares one
//! GPU with its owner: cosine has to know the total step count in advance, so
//! stopping early lands on a rate that was never annealed and a checkpoint that
//! underperforms its loss curve. WSD's stable phase is flat, so any checkpoint
//! taken during it is a valid starting point for a decay of any length — the
//! run can be extended, cut short or forked without invalidating what came
//! before.
//!
//! The decay is `1 - sqrt(progress)`, which MiniCPM measured to beat linear and
//! cosine over the same number of decay steps.

/// Warmup-stable-decay, a pure function of the step index.
#[derive(Debug, Clone, Copy)]
pub struct Wsd {
    peak: f64,
    /// Final rate as a fraction of `peak`; never zero, because the last steps
    /// still have to move.
    floor: f64,
    warmup: usize,
    decay: usize,
    steps: usize,
}

impl Wsd {
    pub fn new(peak: f64, floor: f64, warmup: usize, decay: usize, steps: usize) -> Self {
        assert!(warmup + decay <= steps, "warmup and decay overlap in {steps} steps");
        Self { peak, floor, warmup, decay, steps }
    }

    /// The rate for `step`, counted from zero.
    pub fn lr(&self, step: usize) -> f64 {
        let stable = self.steps - self.decay;
        if step < self.warmup {
            // From one whole step, not from zero: a first step at lr 0 is a
            // wasted forward pass.
            return self.peak * (step + 1) as f64 / self.warmup as f64;
        }
        if step < stable {
            return self.peak;
        }
        let progress = (step - stable + 1) as f64 / self.decay as f64;
        let floor = self.peak * self.floor;
        floor + (self.peak - floor) * (1.0 - progress.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> Wsd {
        Wsd::new(1e-3, 0.1, 10, 20, 100)
    }

    #[test]
    fn warmup_ends_on_the_peak() {
        assert!((schedule().lr(9) - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn the_stable_phase_does_not_move() {
        assert_eq!(schedule().lr(10), schedule().lr(79));
    }

    #[test]
    fn the_last_step_reaches_the_floor() {
        assert!((schedule().lr(99) - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn the_rate_never_rises_after_warmup() {
        let wsd = schedule();

        let rates: Vec<f64> = (9..100).map(|step| wsd.lr(step)).collect();

        assert!(rates.windows(2).all(|w| w[0] >= w[1]), "{rates:?}");
    }
}
