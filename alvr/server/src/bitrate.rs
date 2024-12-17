use crate::{bitrate, FfiDynamicEncoderParams};
use alvr_common::SlidingWindowAverage;
use alvr_events::{EventType, HeuristicStats, NominalBitrateStats};
use alvr_session::{
    settings_schema::Switch, BitrateAdaptiveFramerateConfig, BitrateConfig, BitrateMode,
};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
    cmp::Ordering, 
};
use alvr_common::warn; 
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

const UPDATE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Default, PartialEq)]
pub struct LastNestSettings {
    min: f32, 
    max: f32,
    n_steps: usize, 
}


pub struct BitrateManager {
    nominal_frame_interval: Duration,
    frame_interval_average: SlidingWindowAverage<Duration>,
    // note: why packet_sizes_bits_history is a queue and not a sliding average? Because some
    // network samples will be dropped but not any packet size sample
    packet_sizes_bits_history: VecDeque<(Duration, usize)>,
    encoder_latency_average: SlidingWindowAverage<Duration>,
    network_latency_average: SlidingWindowAverage<Duration>,
    bitrate_average: SlidingWindowAverage<f32>,
    decoder_latency_overstep_count: usize,
    last_frame_instant: Instant,
    last_update_instant: Instant,
    dynamic_max_bitrate: f32,
    previous_config: Option<BitrateConfig>,
    update_needed: bool,

    last_target_bitrate_bps: f32,
    update_interval_s: Duration,

    rtt_average: SlidingWindowAverage<Duration>,
    peak_throughput_average: SlidingWindowAverage<f32>,
    frame_interarrival_average: SlidingWindowAverage<f32>,


    bitrate_ladder_nestvr: Option<Vec<f32>>,
    delta_bitrate_step: f32,
    last_nest_settings: LastNestSettings, 
}




impl BitrateManager {
    pub fn new(max_history_size: usize, initial_framerate: f32, initial_bitrate: f32) -> Self {
        Self {
            nominal_frame_interval: Duration::from_secs_f32(1. / initial_framerate),
            frame_interval_average: SlidingWindowAverage::new(
                Duration::from_millis(16),
                max_history_size,
            ),
            packet_sizes_bits_history: VecDeque::new(),
            encoder_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
            ),
            network_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
            ),
            bitrate_average: SlidingWindowAverage::new(initial_bitrate * 1e6, max_history_size),
            decoder_latency_overstep_count: 0,
            last_frame_instant: Instant::now(),
            last_update_instant: Instant::now(),
            dynamic_max_bitrate: f32::MAX,
            previous_config: None,
            update_needed: true,

            last_target_bitrate_bps: initial_bitrate * 1e6,
            update_interval_s: UPDATE_INTERVAL,

            rtt_average: SlidingWindowAverage::new(Duration::from_millis(5), max_history_size),
            peak_throughput_average: SlidingWindowAverage::new(300E6, max_history_size),
            frame_interarrival_average: SlidingWindowAverage::new(
                1. / initial_framerate,
                max_history_size,
            ),
            bitrate_ladder_nestvr: None, 
            delta_bitrate_step: 0.0, 
            last_nest_settings: LastNestSettings::default(), 
        }
    }

    // Note: This is used to calculate the framerate/frame interval. The frame present is the most
    // accurate event for this use.
    pub fn report_frame_present(&mut self, config: &Switch<BitrateAdaptiveFramerateConfig>) {
        let now = Instant::now();

        let interval = now - self.last_frame_instant;
        self.last_frame_instant = now;

        self.frame_interval_average.submit_sample(interval);

        if let Some(config) = config.as_option() {
            let interval_ratio =
                interval.as_secs_f32() / self.frame_interval_average.get_average().as_secs_f32();

            if interval_ratio > config.framerate_reset_threshold_multiplier
                || interval_ratio < 1.0 / config.framerate_reset_threshold_multiplier
            {
                // Clear most of the samples, keep some for stability
                self.frame_interval_average.retain(5);
                self.update_needed = true;
            }
        }
    }

    pub fn report_frame_encoded(
        &mut self,
        timestamp: Duration,
        encoder_latency: Duration,
        size_bytes: usize,
    ) {
        self.encoder_latency_average.submit_sample(encoder_latency);

        self.packet_sizes_bits_history
            .push_back((timestamp, size_bytes * 8));
    }

    pub fn report_network_statistics(
        &mut self,
        network_rtt: Duration,
        peak_throughput_bps: f32,
        frame_interarrival_s: f32,
    ) {
        self.rtt_average.submit_sample(network_rtt);

        self.peak_throughput_average
            .submit_sample(peak_throughput_bps);

        self.frame_interarrival_average
            .submit_sample(frame_interarrival_s);
    }

    pub fn report_frame_latencies(
        &mut self,
        config: &BitrateMode,
        timestamp: Duration,
        network_latency: Duration,
        decoder_latency: Duration,
    ) {
        if network_latency.is_zero() {
            return;
        }
        self.network_latency_average.submit_sample(network_latency);

        while let Some(&(timestamp_, size_bits)) = self.packet_sizes_bits_history.front() {
            if timestamp_ == timestamp {
                self.bitrate_average
                    .submit_sample(size_bits as f32 / network_latency.as_secs_f32());

                self.packet_sizes_bits_history.pop_front();

                break;
            } else {
                self.packet_sizes_bits_history.pop_front();
            }
        }

        if let BitrateMode::Adaptive {
            decoder_latency_limiter: Switch::Enabled(config),
            ..
        } = &config
        {
            if decoder_latency > Duration::from_millis(config.max_decoder_latency_ms) {
                self.decoder_latency_overstep_count += 1;

                if self.decoder_latency_overstep_count == config.latency_overstep_frames {
                    self.dynamic_max_bitrate =
                        f32::min(self.bitrate_average.get_average(), self.dynamic_max_bitrate)
                            * config.latency_overstep_multiplier;

                    self.update_needed = true;

                    self.decoder_latency_overstep_count = 0;
                }
            } else {
                self.decoder_latency_overstep_count = 0;
            }
        }
    }

    pub fn get_encoder_params(
        &mut self,
        config: &BitrateConfig,
    ) -> (FfiDynamicEncoderParams, Option<NominalBitrateStats>) {
        let now = Instant::now();

        if let BitrateMode::NestVr {
            update_interval_nestvr_s,
            ..
        } = &config.mode
        {
            self.update_interval_s = Duration::from_secs_f32(*update_interval_nestvr_s);
        } else {
            self.update_interval_s = UPDATE_INTERVAL;
        }

        if self
            .previous_config
            .as_ref()
            .map(|prev| config != prev)
            .unwrap_or(true)
        {
            self.previous_config = Some(config.clone());
        } else if !self.update_needed
            && (now < (self.last_update_instant + self.update_interval_s)
                || matches!(config.mode, BitrateMode::ConstantMbps(_)))
        {
            return (
                FfiDynamicEncoderParams {
                    updated: 0,
                    bitrate_bps: 0,
                    framerate: 0.0,
                },
                None,
            );
        }

        self.last_update_instant = now;
        self.update_needed = false;

        let mut stats = NominalBitrateStats::default();

        let bitrate_bps = match &config.mode {
            BitrateMode::ConstantMbps(bitrate_mbps) => *bitrate_mbps as f32 * 1e6,
            BitrateMode::NestVr {
                max_bitrate_mbps,
                min_bitrate_mbps,
                initial_bitrate_mbps,
                // step_size_mbps,
                capacity_scaling_factor,
                rtt_explor_prob,
                nfr_thresh,
                // rtt_thresh_scaling_factor,
                num_steps_bitrate_ladder,
                mulitplier_bitrate_increase,
                multiplier_bitrate_decrease,
                ..
            } => {

                fn upper_bound_bitrate(bitrate_bps: f32, bitrate_ladder: &Vec<f32>) -> f32 {
                    // Perform binary search to find the largest value less than or equal to `bitrate_bps`
                    match bitrate_ladder.binary_search_by(|x| x.partial_cmp(&bitrate_bps).unwrap_or(Ordering::Less)) {
                        Ok(index) => bitrate_ladder[index], // Exact match found
                        Err(index) => {
                            // If not found, `index` is where the value would be inserted to maintain sorted order
                            if index == 0 {
                                // If `bitrate_bps` is smaller than the first element, return the first element
                                bitrate_ladder.first().copied().unwrap_or(bitrate_bps)
                            } else {
                                // Otherwise, return the element just before the insertion point (i.e., the largest <= bitrate_bps)
                                bitrate_ladder[index - 1]
                            }
                        }
                    }
                }

                fn minmax_bitrate(
                    bitrate_bps: f32,
                    max_bitrate_mbps: &Switch<f32>,
                    min_bitrate_mbps: &Switch<f32>,
                ) -> f32 {
                    let mut bitrate = bitrate_bps;
                    if let Switch::Enabled(max) = max_bitrate_mbps {
                        let max = *max as f32 * 1e6;
                        bitrate = f32::min(bitrate, max);

                    }
                    if let Switch::Enabled(min) = min_bitrate_mbps {
                        let min = *min as f32 * 1e6;
                        bitrate = f32::max(bitrate, min);
                    }
                    bitrate
                }

                let mut recompute_bitrate_ladder = false; 

                if let Switch::Enabled(max) = max_bitrate_mbps {
                    if let Switch::Enabled(min) = min_bitrate_mbps {

                        let current_settings = LastNestSettings {
                            max: *max, 
                            min: *min,
                            n_steps: *num_steps_bitrate_ladder,
                        };

                        if current_settings != self.last_nest_settings {
                            recompute_bitrate_ladder = true; 
                        }
                    }
                }

                if self.bitrate_ladder_nestvr.is_none() || recompute_bitrate_ladder == true { // if uninitialized, initialize it for first time
                    
                    let num_steps = *num_steps_bitrate_ladder; 
                    let mut max_b: f32 = 0.0; 
                    let mut min_b: f32 = 0.0; 
                    
                    if let Switch::Enabled(max) = max_bitrate_mbps {
                        max_b = *max as f32 * 1e6;
                    }
                    if let Switch::Enabled(min) = min_bitrate_mbps {
                        min_b = *min as f32 * 1e6;
                    }
                    
                    if max_b != 0.0 && min_b != 0.0 { // ensure values are initialized
                        let mut vec_bitrates =Vec::new(); 

                        let delta_b_step = (max_b - min_b)/num_steps as f32;  

                        let mut last_value = min_b; 
                        vec_bitrates.push(min_b); // first bitrate is min
                        for i in 0..num_steps{
                            last_value += delta_b_step; 
                            vec_bitrates.push(last_value); 

                            // warn!("Adding bitrate to ladder: {}", last_value ); 
                        }
                        // warn!("[DBG NEW LADDER] bitrate_ladder all values: {:?}", vec_bitrates); 
                        self.bitrate_ladder_nestvr = Some(vec_bitrates); // at this point vec_bitrates is initialized, we can unwrap safely to retrieve
                        self.delta_bitrate_step = delta_b_step; 

                        self.last_target_bitrate_bps = upper_bound_bitrate(self.last_target_bitrate_bps, &self.bitrate_ladder_nestvr.clone().unwrap()); // if the bitrate of last pass is not in ladder, put it to the closer lower bound value
                    }          
                }

                // Sample from uniform distribution
                let mut rng = thread_rng();
                let uniform_dist = Uniform::new(0.0, 1.0);
                let random_prob = rng.sample(uniform_dist);

                let mut bitrate_bps: f32 = self.last_target_bitrate_bps;

                let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();
                let rtt_avg_heur_s = self.rtt_average.get_average().as_secs_f32();

                let server_fps = if frame_interval_s != 0.0 {
                    1.0 / frame_interval_s
                } else {
                    0.0
                };
                let heur_fps = if self.frame_interarrival_average.get_average() != 0.0 {
                    1.0 / self.frame_interarrival_average.get_average()
                } else {
                    0.0
                };

                let estimated_capacity_bps = self.peak_throughput_average.get_average();

                let threshold_fps = *nfr_thresh * server_fps;
                let threshold_rtt = frame_interval_s;
                let threshold_u = *rtt_explor_prob;
          
                // mulitplier_bitrate_increase,
                // multiplier_bitrate_decrease,

                if heur_fps >= threshold_fps {
                    if rtt_avg_heur_s > threshold_rtt {
                        if random_prob >= threshold_u {
                            bitrate_bps -= (self.delta_bitrate_step * (*multiplier_bitrate_decrease as f32)); // decrease bitrate by 1 step
                        }
                    } else {
                        if random_prob <= threshold_u {
                            bitrate_bps += (self.delta_bitrate_step * (*mulitplier_bitrate_increase as f32)); // decrease bitrate by 1 step
                        }
                    }
                } else {
                    bitrate_bps -= (self.delta_bitrate_step * (*multiplier_bitrate_decrease as f32)); // decrease bitrate by 1 step
                }

                // Ensure bitrate is within allowed range
                bitrate_bps = minmax_bitrate(bitrate_bps, max_bitrate_mbps, min_bitrate_mbps);

                // Ensure bitrate is below the estimated network capacity
                let capacity_upper_limit = *capacity_scaling_factor * estimated_capacity_bps;


                bitrate_bps = upper_bound_bitrate(f32::min(bitrate_bps, capacity_upper_limit), &self.bitrate_ladder_nestvr.clone().unwrap()); 



                let heur_stats = HeuristicStats {
                    frame_interval_s: frame_interval_s,
                    server_fps: server_fps, // fps_tx
                    steps_bps: self.delta_bitrate_step,

                    network_heur_fps: heur_fps, // fps_rx
                    rtt_avg_heur_s: rtt_avg_heur_s,
                    random_prob: random_prob,

                    threshold_fps: threshold_fps,
                    threshold_rtt_s: threshold_rtt,
                    threshold_u: threshold_u,

                    requested_bitrate_bps: bitrate_bps,
                };
                alvr_events::send_event(EventType::HeuristicStats(heur_stats));

                if let Switch::Enabled(max) = max_bitrate_mbps {
                    let maxi = *max as f32 * 1e6;
                    stats.manual_max_bps = Some(maxi);
                }
                if let Switch::Enabled(min) = min_bitrate_mbps {
                    let mini = *min as f32 * 1e6;
                    stats.manual_min_bps = Some(mini);
                }
                bitrate_bps
            }
              
            BitrateMode::Adaptive {
                saturation_multiplier,
                max_bitrate_mbps,
                min_bitrate_mbps,
                max_network_latency_ms,
                encoder_latency_limiter,
                ..
            } => {
                let initial_bitrate_average_bps = self.bitrate_average.get_average();

                let mut bitrate_bps = initial_bitrate_average_bps * saturation_multiplier;
                stats.scaled_calculated_bps = Some(bitrate_bps);

                bitrate_bps = f32::min(bitrate_bps, self.dynamic_max_bitrate);
                stats.decoder_latency_limiter_bps = Some(self.dynamic_max_bitrate);

                if let Switch::Enabled(max_ms) = max_network_latency_ms {
                    let max = initial_bitrate_average_bps * (*max_ms as f32 / 1000.0)
                        / self.network_latency_average.get_average().as_secs_f32();
                    bitrate_bps = f32::min(bitrate_bps, max);

                    stats.network_latency_limiter_bps = Some(max);
                }

                if let Switch::Enabled(config) = encoder_latency_limiter {
                    let saturation = self.encoder_latency_average.get_average().as_secs_f32()
                        / self.nominal_frame_interval.as_secs_f32();
                    let max =
                        initial_bitrate_average_bps * config.max_saturation_multiplier / saturation;
                    stats.encoder_latency_limiter_bps = Some(max);

                    if saturation > config.max_saturation_multiplier {
                        // Note: this assumes linear relationship between bitrate and encoder
                        // latency but this may not be the case
                        bitrate_bps = f32::min(bitrate_bps, max);
                    }
                }

                if let Switch::Enabled(max) = max_bitrate_mbps {
                    let max = *max as f32 * 1e6;
                    bitrate_bps = f32::min(bitrate_bps, max);

                    stats.manual_max_bps = Some(max);
                }
                if let Switch::Enabled(min) = min_bitrate_mbps {
                    let min = *min as f32 * 1e6;
                    bitrate_bps = f32::max(bitrate_bps, min);

                    stats.manual_min_bps = Some(min);
                }

                bitrate_bps
            }
        };

        stats.requested_bps = bitrate_bps;
        self.last_target_bitrate_bps = bitrate_bps;

        let frame_interval = if config.adapt_to_framerate.enabled() {
            self.frame_interval_average.get_average()
        } else {
            self.nominal_frame_interval
        };

        (
            FfiDynamicEncoderParams {
                updated: 1,
                bitrate_bps: bitrate_bps as u64,
                framerate: 1.0 / frame_interval.as_secs_f32().min(1.0),
            },
            Some(stats),
        )
    }
}
