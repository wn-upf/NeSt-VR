use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub struct SlidingWindowAverage<T> {
    history_buffer: VecDeque<(T, Instant)>,
    max_history_size: Option<usize>,    // For sample-based
    history_interval: Option<Duration>, // For time-based
    ewma_value: Option<f32>,            // Current value for EWMA
    ewma_weight: Option<f32>,           // For EWMA
}

impl<T: Clone> SlidingWindowAverage<T> {
    pub fn new(
        initial_value: T,
        max_history_size: Option<usize>,
        history_interval: Option<Duration>,
        ewma_weight: Option<f32>,
    ) -> Self {
        Self {
            history_buffer: VecDeque::from([(initial_value, Instant::now())]),
            max_history_size,
            history_interval,
            ewma_value: None,
            ewma_weight,
        }
    }

    pub fn update_history_interval(&mut self, history_interval: Option<Duration>) {
        if self.history_interval != history_interval {
            self.history_interval = history_interval;
        }
    }

    pub fn update_max_history_size(&mut self, max_history_size: Option<usize>) {
        if self.max_history_size != max_history_size {
            self.max_history_size = max_history_size;
        }
    }

    pub fn retain(&mut self, count: usize) {
        self.history_buffer
            .drain(0..self.history_buffer.len().saturating_sub(count));
    }

    pub fn history_buffer_len(&self) -> usize {
        self.history_buffer.len()
    }
}

impl SlidingWindowAverage<f32> {
    pub fn update_ewma_weight(&mut self, ewma_weight: Option<f32>) {
        if self.ewma_weight != ewma_weight {
            self.ewma_weight = ewma_weight;
            // Reset EWMA value using last sample
            self.ewma_value = self
                .history_buffer
                .back()
                .map(|(value, _)| (*value).clone());
        }
    }

    pub fn submit_sample(&mut self, sample: f32) {
        let now = Instant::now();

        // Handle sample-based
        if let Some(history_size) = self.max_history_size {
            if self.history_buffer.len() >= history_size {
                self.history_buffer.pop_front();
            }
        }

        // Handle time-based: remove old samples that are outside the time window
        if let Some(history_interval) = self.history_interval {
            while let Some(&(_, timestamp)) = self.history_buffer.front() {
                if now.duration_since(timestamp) > history_interval {
                    self.history_buffer.pop_front();
                } else {
                    break; // Keep the samples that are within the time window
                }
            }
        }

        self.history_buffer.push_back((sample, now));

        // Update EWMA value if ewma weight is set
        if let Some(ewma_weight) = self.ewma_weight {
            if let Some(ewma_prev) = self.ewma_value.clone() {
                // Calculate EWMA: EWMA = ewma_weight * current_value + (1-ewma_weight) * ewma_prev
                self.ewma_value = Some(ewma_weight * sample + (1. - ewma_weight) * ewma_prev);
            } else {
                // Initialize EWMA if it's the first sample
                self.ewma_value = Some(sample);
            }
        }
    }

    pub fn get_average(&self) -> f32 {
        if self.ewma_weight.is_some() {
            return self.ewma_value.unwrap_or(0.0);
        }

        self.get_simple_average()
    }

    pub fn get_simple_average(&self) -> f32 {
        if self.history_buffer.is_empty() {
            return 0.0;
        }

        self.history_buffer
            .iter()
            .map(|&(value, _)| value)
            .sum::<f32>()
            / self.history_buffer.len() as f32
    }

    pub fn get_std(&self) -> f32 {
        if self.history_buffer.len() < 2 {
            return 0.0;
        }
        let average = self.get_simple_average();
        let variance = self
            .history_buffer
            .iter()
            .map(|&(value, _)| (value - average).powf(2.))
            .sum::<f32>()
            / (self.history_buffer.len() - 1) as f32; // sample variance
        variance.sqrt()
    }
}

impl SlidingWindowAverage<Duration> {
    pub fn update_ewma_weight(&mut self, ewma_weight: Option<f32>) {
        if self.ewma_weight != ewma_weight {
            self.ewma_weight = ewma_weight;
            // Reset EWMA value using last sample
            self.ewma_value = self
                .history_buffer
                .back()
                .map(|(value, _)| (*value).clone().as_secs_f32());
        }
    }

    pub fn submit_sample(&mut self, sample: Duration) {
        let now = Instant::now();

        // Handle sample-based
        if let Some(history_size) = self.max_history_size {
            if self.history_buffer.len() >= history_size {
                self.history_buffer.pop_front();
            }
        }

        // Handle time-based: remove old samples that are outside the time window
        if let Some(history_interval) = self.history_interval {
            while let Some(&(_, timestamp)) = self.history_buffer.front() {
                if now.duration_since(timestamp) > history_interval {
                    self.history_buffer.pop_front();
                } else {
                    break; // Keep the samples that are within the time window
                }
            }
        }

        self.history_buffer.push_back((sample, now));

        // Update EWMA value if ewma weight is set
        if let Some(ewma_weight) = self.ewma_weight {
            if let Some(ewma_prev) = self.ewma_value.clone() {
                // Calculate EWMA: EWMA = ewma_weight * current_value + (1-ewma_weight) * ewma_prev
                self.ewma_value =
                    Some(ewma_weight * sample.as_secs_f32() + (1. - ewma_weight) * ewma_prev);
            } else {
                // Initialize EWMA if it's the first sample
                self.ewma_value = Some(sample.as_secs_f32());
            }
        }
    }

    pub fn get_average(&self) -> Duration {
        if self.ewma_weight.is_some() {
            return Duration::from_secs_f32(self.ewma_value.unwrap_or(0.0));
        }

        self.get_simple_average()
    }

    pub fn get_simple_average(&self) -> Duration {
        if self.history_buffer.is_empty() {
            return Duration::ZERO;
        }

        self.history_buffer
            .iter()
            .map(|&(value, _)| value)
            .sum::<Duration>()
            / self.history_buffer.len() as u32
    }
}
