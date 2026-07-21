//! The official Burn training dashboard, adapted to the custom training loop.

use std::io::{self, IsTerminal};
use std::sync::Arc;

use burn::train::Interrupter;
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAggregation,
    NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{MetricState, MetricsRenderer};

use crate::eval;

/// Burn's renderer plus the metric identities shared by training and validation.
pub(super) struct Dashboard {
    renderer: Option<Box<dyn MetricsRenderer>>,
    interrupter: Interrupter,
    metrics: Metrics,
}

impl Dashboard {
    pub(super) fn new(total_steps: usize, completed_steps: usize) -> Self {
        let interrupter = Interrupter::new();
        let renderer = io::stdout().is_terminal().then(|| {
            Box::new(TuiMetricsRendererWrapper::new(interrupter.clone(), None))
                as Box<dyn MetricsRenderer>
        });
        Self::with_optional_renderer(renderer, interrupter, total_steps, completed_steps)
    }

    fn with_optional_renderer(
        renderer: Option<Box<dyn MetricsRenderer>>,
        interrupter: Interrupter,
        total_steps: usize,
        completed_steps: usize,
    ) -> Self {
        let metrics = Metrics::new();
        let mut dashboard = Self { renderer, interrupter, metrics };
        if let Some(renderer) = dashboard.renderer.as_mut() {
            for definition in dashboard.metrics.definitions() {
                renderer.register_metric(definition);
            }
            renderer.start(1, Some(total_steps));
            renderer.start_split("optimizer steps", total_steps);
            renderer.update_split(completed_steps);
        }
        dashboard
    }

    #[cfg(test)]
    fn with_renderer(
        renderer: Box<dyn MetricsRenderer>,
        total_steps: usize,
        completed_steps: usize,
    ) -> Self {
        Self::with_optional_renderer(
            Some(renderer),
            Interrupter::new(),
            total_steps,
            completed_steps,
        )
    }

    pub(super) fn active(&self) -> bool {
        self.renderer.is_some()
    }

    pub(super) fn should_stop(&self) -> bool {
        self.interrupter.should_stop()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn train(
        &mut self,
        step: usize,
        loss: f64,
        lr: f64,
        throughput: f64,
        tokens: u64,
        eta_hours: f64,
        tflops: f64,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.update_split(step);
        renderer.update_train(self.metrics.loss.state(loss));
        renderer.update_train(self.metrics.lr.state(lr));
        renderer.update_train(self.metrics.throughput.state(throughput));
        renderer.update_train(self.metrics.tokens.state(tokens as f64));
        renderer.update_train(self.metrics.eta.state(eta_hours));
        renderer.update_train(self.metrics.compute.state(tflops));
    }

    pub(super) fn valid(&mut self, report: eval::Report) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        renderer.update_valid(self.metrics.loss.state(report.nll));
        renderer.update_valid(self.metrics.perplexity.state(report.perplexity));
        renderer.update_valid(self.metrics.bits_per_byte.state(report.bits_per_byte));
    }

    pub(super) fn finish(mut self) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.end_split();
            renderer.end();
            let _ = renderer.on_train_end(None);
        }
    }
}

struct Metrics {
    loss: Scalar,
    perplexity: Scalar,
    bits_per_byte: Scalar,
    lr: Scalar,
    throughput: Scalar,
    tokens: Scalar,
    eta: Scalar,
    compute: Scalar,
}

impl Metrics {
    fn new() -> Self {
        Self {
            loss: Scalar::new("Loss", None, false, 4),
            perplexity: Scalar::new("Perplexity", None, false, 2),
            bits_per_byte: Scalar::new("Bits per byte", Some("bpb"), false, 4),
            lr: Scalar::new("Learning rate", None, false, 3),
            throughput: Scalar::new("Throughput", Some("tok/s"), true, 0),
            tokens: Scalar::new("Tokens", None, true, 0),
            eta: Scalar::new("ETA", Some("h"), false, 1),
            compute: Scalar::new("Effective compute", Some("TFLOP/s"), true, 2),
        }
    }

    fn definitions(&self) -> impl Iterator<Item = MetricDefinition> + '_ {
        [
            &self.loss,
            &self.perplexity,
            &self.bits_per_byte,
            &self.lr,
            &self.throughput,
            &self.tokens,
            &self.eta,
            &self.compute,
        ]
        .into_iter()
        .map(Scalar::definition)
    }
}

struct Scalar {
    id: MetricId,
    name: &'static str,
    unit: Option<&'static str>,
    higher_is_better: bool,
    precision: usize,
}

impl Scalar {
    fn new(
        name: &'static str,
        unit: Option<&'static str>,
        higher_is_better: bool,
        precision: usize,
    ) -> Self {
        Self {
            id: MetricId::new(Arc::new(name.to_owned())),
            name,
            unit,
            higher_is_better,
            precision,
        }
    }

    fn definition(&self) -> MetricDefinition {
        MetricDefinition {
            metric_id: self.id.clone(),
            name: self.name.to_owned(),
            description: None,
            attributes: MetricAttributes::Numeric(NumericAttributes {
                unit: self.unit.map(str::to_owned),
                higher_is_better: self.higher_is_better,
                aggregation: NumericAggregation::Last,
            }),
        }
    }

    fn state(&self, value: f64) -> MetricState {
        let formatted = match self.unit {
            Some(unit) => format!("{}: {:.*} {unit}", self.name, self.precision, value),
            None => format!("{}: {:.*}", self.name, self.precision, value),
        };
        let entry =
            MetricEntry::new(self.id.clone(), SerializedEntry::new(formatted, value.to_string()));
        MetricState::Numeric(entry, NumericEntry::Value(value))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use burn::train::LearnerSummary;
    use burn::train::logger::{EvaluationProgressLogger, TrainingProgressLogger};
    use burn::train::metric::MetricDefinition;
    use burn::train::renderer::{
        EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
        MetricsRendererTraining,
    };

    use super::*;

    #[derive(Debug, PartialEq)]
    enum Seen {
        Metric(String),
        Progress(usize),
        Train(f64),
        Valid(f64),
        End,
    }

    struct Recorder(Arc<Mutex<Vec<Seen>>>);

    impl MetricsRendererTraining for Recorder {
        fn update_train(&mut self, state: MetricState) {
            if let MetricState::Numeric(_, value) = state {
                self.0.lock().unwrap().push(Seen::Train(value.current()));
            }
        }

        fn update_valid(&mut self, state: MetricState) {
            if let MetricState::Numeric(_, value) = state {
                self.0.lock().unwrap().push(Seen::Valid(value.current()));
            }
        }

        fn on_train_end(
            &mut self,
            _summary: Option<LearnerSummary>,
        ) -> Result<(), Box<dyn std::error::Error>> {
            self.0.lock().unwrap().push(Seen::End);
            Ok(())
        }
    }

    impl TrainingProgressLogger for Recorder {
        fn start(&mut self, _total_epochs: usize, _total_items: Option<usize>) {}
        fn update_epoch(&mut self, _epoch: usize) {}
        fn start_split(&mut self, _split: &str, _total_items: usize) {}
        fn update_split(&mut self, items: usize) {
            self.0.lock().unwrap().push(Seen::Progress(items));
        }
        fn end_split(&mut self) {}
        fn end(&mut self) {}
        fn log_event_training(&mut self, _event: String) {}
    }

    impl MetricsRenderer for Recorder {
        fn manual_close(&mut self) {}
        fn register_metric(&mut self, definition: MetricDefinition) {
            self.0.lock().unwrap().push(Seen::Metric(definition.name));
        }
    }

    impl MetricsRendererEvaluation for Recorder {
        fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
    }

    impl EvaluationProgressLogger for Recorder {
        fn start_global_progress(&mut self, _total_tests: usize) {}
        fn start_test(&mut self, _name: &str, _total_items: usize) {}
        fn update_test_progress(&mut self, _items_processed: usize) {}
        fn end_test(&mut self) {}
        fn end_global_progress(&mut self) {}
        fn log_event_evaluation(&mut self, _event: String) {}
    }

    #[test]
    fn loss_and_progress_reach_the_burn_renderer() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let renderer = Box::new(Recorder(seen.clone()));
        let mut dashboard = Dashboard::with_renderer(renderer, 100, 40);

        dashboard.train(41, 2.5, 3e-3, 12_345.0, 65_536, 12.0, 13.5);
        dashboard.valid(eval::Report {
            nll: 2.0,
            perplexity: 2.0_f64.exp(),
            bits_per_byte: 0.75,
            tokens: 1024,
        });
        dashboard.finish();

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.iter()
                .filter_map(|event| match event {
                    Seen::Metric(name) => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [
                "Loss",
                "Perplexity",
                "Bits per byte",
                "Learning rate",
                "Throughput",
                "Tokens",
                "ETA",
                "Effective compute",
            ]
        );
        assert!(seen.contains(&Seen::Progress(40)), "resume position is rendered");
        assert!(seen.contains(&Seen::Progress(41)), "new step is rendered");
        assert!(seen.contains(&Seen::Train(2.5)), "training loss is rendered");
        assert!(seen.contains(&Seen::Valid(2.0)), "validation loss is rendered");
        assert!(seen.contains(&Seen::End));
    }
}
