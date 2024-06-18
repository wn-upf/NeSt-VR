use std::collections::VecDeque;

pub struct SlidingWindowWeighted<T> {
    history_buffer: VecDeque<T>,
    interval_buffer: VecDeque<f32>,
}

impl<T> SlidingWindowWeighted<T> {
    pub fn new(initial_value: T, initial_interval: f32) -> Self {
        Self {
            history_buffer: [initial_value].into_iter().collect(),
            interval_buffer: [initial_interval].into_iter().collect(),
        }
    }

    pub fn submit_sample(&mut self, sample: T, interval: f32) {
        self.history_buffer.push_back(sample);
        self.interval_buffer.push_back(interval);
    }

    fn cleanup_old_samples(&mut self) {
        self.history_buffer.clear();
        self.interval_buffer.clear();
    }

    pub fn get_interval_buffer_sum(&self) -> f32 {
        self.interval_buffer.iter().sum::<f32>()
    }
}

impl SlidingWindowWeighted<f32> {
    pub fn weighted_sum(&self) -> f32 {
        self.history_buffer
            .iter()
            .zip(self.interval_buffer.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    }

    pub fn get_average(&mut self) -> f32 {
        let average = self.weighted_sum() / self.get_interval_buffer_sum();
        self.cleanup_old_samples();
        return average;
    }
}
