use std::collections::VecDeque;

pub struct SlidingWindowTimely<T> {
    history_buffer: VecDeque<T>,
    interval_buffer: VecDeque<f32>,
    max_window_duration: f32,
}

impl<T> SlidingWindowTimely<T> {
    pub fn new(initial_value: T, initial_interval: f32, max_window_duration: f32) -> Self {
        Self {
            history_buffer: [initial_value].into_iter().collect(),
            interval_buffer: [initial_interval].into_iter().collect(),
            max_window_duration,
        }
    }

    pub fn submit_sample(&mut self, sample: T, interval: f32) {
        self.history_buffer.push_back(sample);
        self.interval_buffer.push_back(interval);
        self.cleanup_old_samples();
    }

    fn cleanup_old_samples(&mut self) {
        let mut total_interval = 0.0;
        let mut index = self.interval_buffer.len();

        for &interval in self.interval_buffer.iter().rev() {
            total_interval += interval;
            if total_interval > self.max_window_duration {
                break;
            }
            index -= 1;
        }

        while index > 0 {
            if self.interval_buffer.len() > 1 {
                // keep at least one
                self.history_buffer.pop_front();
                self.interval_buffer.pop_front();
            }
            index -= 1;
        }
    }

    pub fn get_interval_buffer_sum(&self) -> f32 {
        self.interval_buffer.iter().sum::<f32>()
    }

    pub fn get_interval_buffer_mean(&self) -> f32 {
        self.get_interval_buffer_sum() / self.interval_buffer.len() as f32
    }
}

impl SlidingWindowTimely<f32> {
    pub fn get_average(&self) -> f32 {
        self.history_buffer.iter().sum::<f32>() / self.history_buffer.len() as f32
    }

    pub fn get_sum(&self) -> f32 {
        self.history_buffer.iter().sum::<f32>()
    }

    pub fn get_std(&self) -> f32 {
        if self.history_buffer.len() < 2 {
            return 0.;
        }
        let average = self.get_average();
        let variance = self
            .history_buffer
            .iter()
            .map(|&x| (x - average).powf(2.))
            .sum::<f32>()
            / (self.history_buffer.len() - 1) as f32; // sample variance
        variance.sqrt()
    }
}
