use crate::FfiDynamicEncoderParams;
use alvr_common::SlidingWindowAverage;
use alvr_events::{EventType, HeuristicStats, NominalBitrateStats};
use alvr_session::GccBandwidthEstimator;
use alvr_session::{
    get_profile_config, settings_schema::Switch, AveragingStrategy, BitrateAdaptiveFramerateConfig,
    BitrateConfig, BitrateMode, WindowType,
};
use alvr_packets::NetworkStatisticsPacket; 
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};
use std::{
    cmp::Ordering,
    collections::VecDeque,
    time::{Duration, Instant},
};

const UPDATE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Default, PartialEq)]
pub struct LastNestSettings {
    max_bps: f32,
    min_bps: f32,
    bitrate_step_count: usize,
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
    update_interval: Duration,

    rtt_average: SlidingWindowAverage<Duration>,
    peak_throughput_average: SlidingWindowAverage<f32>,
    frame_interarrival_average: SlidingWindowAverage<f32>,

    bitrate_ladder_bps: Option<Vec<f32>>,
    bitrate_step_size_bps: f32,
    last_nest_settings: LastNestSettings,
    last_nada_target_bitrate_mbps: Option<f64>, 
    
    pub gcc_estimator: Option<GccBandwidthEstimator>,
    pub nada_sender: Option<Arc<Mutex<NadaSender>>>,

    everest_last_capacity: f32,
    everest_last_throughput: f32,
    everest_capacity_ewma: f32,
    everest_throughput_ewma: f32,
    everest_last_dshort: f32,
    everest_last_dlong: f32,
    everest_time_last_capacity_update: Instant,
    everest_time_last_throughput_update: Instant,
    everest_last_order: EverestCommand,
}

impl BitrateManager {
    pub fn new(
        max_history_size: Option<usize>,
        initial_framerate: f32,
        initial_bitrate: f32,
        history_interval: Option<Duration>,
        ewma_weight_val: Option<f32>,
    ) -> Self {
        Self {
            nominal_frame_interval: Duration::from_secs_f32(1. / initial_framerate),
            frame_interval_average: SlidingWindowAverage::new(
                Duration::from_millis(16),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            packet_sizes_bits_history: VecDeque::new(),
            encoder_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            network_latency_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            bitrate_average: SlidingWindowAverage::new(
                initial_bitrate * 1E6,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            decoder_latency_overstep_count: 0,
            last_frame_instant: Instant::now(),
            last_update_instant: Instant::now(),
            dynamic_max_bitrate: f32::MAX,
            previous_config: None,
            update_needed: true,

            last_target_bitrate_bps: initial_bitrate * 1E6,
            update_interval: UPDATE_INTERVAL,

            rtt_average: SlidingWindowAverage::new(
                Duration::from_millis(5),
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            peak_throughput_average: SlidingWindowAverage::new(
                300E6,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),
            frame_interarrival_average: SlidingWindowAverage::new(
                1. / initial_framerate,
                max_history_size,
                history_interval,
                ewma_weight_val,
            ),

            bitrate_ladder_bps: None,
            bitrate_step_size_bps: 0.0,
            last_nest_settings: LastNestSettings::default(),
            last_nada_target_bitrate_mbps: None, 
            gcc_estimator: None, 
            nada_sender: None, 
            everest_last_capacity: 0.0,
            everest_last_throughput: 0.0,
            everest_last_dlong: 0.0,
            everest_last_dshort: 0.0,
            everest_capacity_ewma: 0.0,
            everest_throughput_ewma: 0.0,
            everest_time_last_capacity_update: Instant::now(),
            everest_time_last_throughput_update: Instant::now(),
            everest_last_order: EverestCommand::Continue,
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
        network_stats: NetworkStatisticsPacket, // pass whole message

    ) {
        self.rtt_average.submit_sample(network_rtt);

        self.peak_throughput_average
            .submit_sample(peak_throughput_bps);

        self.frame_interarrival_average
            .submit_sample(frame_interarrival_s);

         /// EVEREST METRICS COMPUTE ///////
        const T_USER_WIN: f32 = 5.0; // from original Everest paper

        let everest_capacity_sample = network_stats.everest_capacity_update;
        let everest_throughput_sample = network_stats.everest_throughput_update;

        if everest_capacity_sample > 0.0 {
            // we only get one or the other, which is computed is based on frame size. They are initialized to -1.0, only positive samples count
            self.everest_last_capacity = everest_capacity_sample;
            let last_c_f32 = now
                .duration_since(self.everest_time_last_capacity_update)
                .as_secs_f32();

            self.everest_capacity_ewma = last_c_f32 / T_USER_WIN * everest_capacity_sample
                + (1.0 - last_c_f32 / T_USER_WIN) * self.everest_capacity_ewma;
            self.everest_time_last_capacity_update = now;
        }

        if everest_throughput_sample > 0.0 {
            self.everest_last_throughput = everest_throughput_sample;
            let last_t_f32 = now
                .duration_since(self.everest_time_last_throughput_update)
                .as_secs_f32();

            self.everest_throughput_ewma = last_t_f32 / T_USER_WIN * everest_throughput_sample
                + (1.0 - last_t_f32 / T_USER_WIN) * self.everest_throughput_ewma;
            self.everest_time_last_throughput_update = now;
        }
        self.everest_last_dshort = network_stats.everest_dshort;
        self.everest_last_dlong = network_stats.everest_dlong;
        self.everest_last_order = network_stats.everest_command;
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

        if self
            .previous_config
            .as_ref()
            .map(|prev| config != prev)
            .unwrap_or(true)
        {
            self.previous_config = Some(config.clone());

            let mut max_history_size = Some(256);
            let mut history_interval = None;
            let mut ewma_weight_val = None;

            match &config.mode {
                BitrateMode::NestVr {
                    max_bitrate_mbps,
                    min_bitrate_mbps,
                    initial_bitrate_mbps,
                    averaging_strategy,
                    nest_vr_profile,
                    ..
                } => {
                    let profile_config = get_profile_config(
                        *max_bitrate_mbps,
                        *min_bitrate_mbps,
                        *initial_bitrate_mbps,
                        nest_vr_profile,
                    );

                    self.update_interval =
                        Duration::from_secs_f32(profile_config.update_interval_nestvr_s);

                    match averaging_strategy {
                        AveragingStrategy::SimpleWindowAverage { window_type, .. } => {
                            match window_type {
                                WindowType::BySeconds {
                                    sliding_window_secs,
                                    ..
                                } => {
                                    history_interval = Some(Duration::from_secs_f32(
                                        sliding_window_secs
                                            .unwrap_or(profile_config.update_interval_nestvr_s),
                                    ));

                                    max_history_size = None;
                                }
                                WindowType::BySamples {
                                    sliding_window_samp,
                                    ..
                                } => {
                                    max_history_size = Some(*sliding_window_samp);
                                }
                            }
                        }
                        AveragingStrategy::ExponentialMovingAverage { ewma_weight, .. } => {
                            ewma_weight_val = Some(*ewma_weight);
                        }
                    }
                }
                BitrateMode::Adaptive { history_size, .. } => {
                    self.update_interval = UPDATE_INTERVAL;

                    max_history_size = Some(*history_size);
                }

                // Everest and GCC run per-frame 

                BitrateMode::EverestPort { ..}  => {
                    self.update_interval = self.nominal_frame_interval; 
                }
                BitrateMode::GCCPort { ..} => {
                    self.update_interval = self.nominal_frame_interval
                }                
                _ => {
                    self.update_interval = UPDATE_INTERVAL;
                }
            }
            let averages_dur = [
                &mut self.frame_interval_average,
                &mut self.encoder_latency_average,
                &mut self.network_latency_average,
                &mut self.rtt_average,
            ];
            let averages_f32 = [
                &mut self.bitrate_average,
                &mut self.peak_throughput_average,
                &mut self.frame_interarrival_average,
            ];

            for average in averages_dur {
                average.update_max_history_size(max_history_size);
                average.update_history_interval(history_interval);
                average.update_ewma_weight(ewma_weight_val);
            }
            for average in averages_f32 {
                average.update_max_history_size(max_history_size);
                average.update_history_interval(history_interval);
                average.update_ewma_weight(ewma_weight_val);
            }
        } else if !self.update_needed
            && (now < (self.last_update_instant + self.update_interval)
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
                nest_vr_profile,
                ..
            } => {
                pub fn upper_bound_bitrate(bitrate_bps: f32, bitrate_ladder: &Vec<f32>) -> f32 {
                    // Perform binary search to find the largest value less than or equal to `bitrate_bps`
                    match bitrate_ladder
                        .binary_search_by(|x| x.partial_cmp(&bitrate_bps).unwrap_or(Ordering::Less))
                    {
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
                    max_bitrate_bps: f32,
                    min_bitrate_bps: f32,
                ) -> f32 {
                    let mut bitrate = bitrate_bps;

                    bitrate = f32::min(bitrate, max_bitrate_bps);
                    bitrate = f32::max(bitrate, min_bitrate_bps);

                    bitrate
                }

                let profile_config = get_profile_config(
                    *max_bitrate_mbps,
                    *min_bitrate_mbps,
                    *initial_bitrate_mbps,
                    nest_vr_profile,
                );

                let mut recompute_bitrate_ladder = false;

                let current_settings = LastNestSettings {
                    max_bps: max_bitrate_mbps * 1E6,
                    min_bps: min_bitrate_mbps * 1E6,
                    bitrate_step_count: profile_config.bitrate_step_count,
                };

                if current_settings != self.last_nest_settings {
                    recompute_bitrate_ladder = true;
                }

                // ensure max is bigger than min
                let (min_bps, max_bps) = if min_bitrate_mbps > max_bitrate_mbps {
                    (max_bitrate_mbps * 1E6, min_bitrate_mbps * 1E6)
                } else {
                    (min_bitrate_mbps * 1E6, max_bitrate_mbps * 1E6)
                };

                if self.bitrate_ladder_bps.is_none() || recompute_bitrate_ladder == true {
                    let bitrate_step_count = profile_config.bitrate_step_count;

                    if max_bps != 0.0 && min_bps != 0.0 {
                        let mut vec_bitrates = Vec::new();

                        let bitrate_step_size_bps = (max_bps - min_bps) / bitrate_step_count as f32;

                        let mut last_value = min_bps;

                        vec_bitrates.push(min_bps); // first bitrate is min
                        for _ in 0..bitrate_step_count {
                            last_value += bitrate_step_size_bps;
                            vec_bitrates.push(last_value);
                        }

                        self.bitrate_ladder_bps = Some(vec_bitrates);
                        self.bitrate_step_size_bps = bitrate_step_size_bps;

                        self.last_target_bitrate_bps = upper_bound_bitrate(
                            self.last_target_bitrate_bps,
                            &self.bitrate_ladder_bps.clone().unwrap(),
                        );
                    }
                }

                // Sample from uniform distribution
                let mut rng = thread_rng();
                let uniform_dist = Uniform::new(0.0, 1.0);

                let r_rtt = rng.sample(uniform_dist);
                let r_inc = rng.sample(uniform_dist);

                let frame_interval_s = self.frame_interval_average.get_average().as_secs_f32();

                let fps_tx_avg = if frame_interval_s != 0.0 {
                    1.0 / frame_interval_s
                } else {
                    0.0
                };

                let fps_rx_avg = if self.frame_interarrival_average.get_average() != 0.0 {
                    1.0 / self.frame_interarrival_average.get_average()
                } else {
                    0.0
                };

                let nfr_avg = fps_rx_avg / fps_tx_avg;
                let rtt_avg_ms = self.rtt_average.get_average().as_secs_f32() * 1000.0;

                let estimated_capacity_bps = self.peak_throughput_average.get_average();

                let mut bitrate_bps: f32 = self.last_target_bitrate_bps;

                if nfr_avg < profile_config.nfr_thresh {
                    // decrease
                    bitrate_bps -=
                        profile_config.bitrate_dec_steps as f32 * self.bitrate_step_size_bps;
                } else {
                    if rtt_avg_ms > profile_config.rtt_thresh_ms {
                        if r_rtt <= profile_config.rtt_adj_prob {
                            // decrease
                            bitrate_bps -= profile_config.bitrate_dec_steps as f32
                                * self.bitrate_step_size_bps;
                        }
                    } else {
                        if r_inc <= profile_config.bitrate_inc_prob {
                            // increase
                            bitrate_bps += profile_config.bitrate_inc_steps as f32
                                * self.bitrate_step_size_bps;
                        }
                    }
                }

                // Ensure bitrate is below the estimated network capacity
                let capacity_upper_limit =
                    profile_config.capacity_scaling_factor * estimated_capacity_bps;

                bitrate_bps = f32::min(bitrate_bps, capacity_upper_limit);

                // Ensure bitrate is always within the configured range
                bitrate_bps = minmax_bitrate(bitrate_bps, max_bps, min_bps);

                bitrate_bps =
                    upper_bound_bitrate(bitrate_bps, &self.bitrate_ladder_bps.clone().unwrap());

                let heur_stats = HeuristicStats {
                    bitrate_step_count: profile_config.bitrate_step_count,

                    bitrate_dec_steps: profile_config.bitrate_dec_steps,
                    bitrate_inc_steps: profile_config.bitrate_inc_steps,

                    bitrate_step_size_bps: self.bitrate_step_size_bps,

                    r_rtt: r_rtt,
                    r_inc: r_inc,

                    rtt_adj_prob: profile_config.rtt_adj_prob,
                    bitrate_inc_prob: profile_config.bitrate_inc_prob,

                    fps_tx_avg: fps_tx_avg,
                    fps_rx_avg: fps_rx_avg,

                    nfr_avg: nfr_avg,
                    rtt_avg_ms: rtt_avg_ms,

                    nfr_thresh: profile_config.nfr_thresh,
                    rtt_thresh_ms: profile_config.rtt_thresh_ms,

                    requested_bitrate_bps: bitrate_bps,
                };
                alvr_events::send_event(EventType::HeuristicStats(heur_stats));

                stats.manual_max_bps = Some(max_bitrate_mbps * 1E6);
                stats.manual_min_bps = Some(min_bitrate_mbps * 1E6);

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
        
            BitrateMode::EverestPort {
            } => {
                    // let mut bitrate_bps = self.last_target_bitrate_bps;
                    // print_red!("bitrate first: {} Mbps", bitrate_bps / 1e6);

                    let current_mbps = (self.last_target_bitrate_bps as f32) / 1e6;
                    let new_mbps = match self.everest_last_order {
                        EverestCommand::Continue => {
                            // stay on the same rung (or the closest one)
                            bitrate_ladder_mbps
                                .iter()
                                .find(|&&x| (x - current_mbps).abs() < std::f32::EPSILON)
                                .copied()
                                .unwrap_or(current_mbps)
                        }
                        EverestCommand::SpeedUp => {
                            // first entry strictly greater than current
                            bitrate_ladder_mbps
                                .iter()
                                .find(|&&x| x > current_mbps)
                                .copied()
                                .unwrap_or(*bitrate_ladder_mbps.last().unwrap())
                        }
                        EverestCommand::SlowDown => {
                            // last entry strictly less than current
                            bitrate_ladder_mbps
                                .iter()
                                .rfind(|&&x| x < current_mbps)
                                .copied()
                                .unwrap_or(bitrate_ladder_mbps[0])
                        }
                    };
                    let mut bitrate_bps = new_mbps * 1e6;

                    let n_users =
                        (self.everest_capacity_ewma / self.everest_throughput_ewma).ceil() as usize;
                    let capacity_margin_bps = self.everest_capacity_ewma / (n_users as f32 + 1.0);

                    bitrate_bps = f32::min(capacity_margin_bps, bitrate_bps);
                    if let Some(ladder) = &self.bitrate_ladder_bps {
                        bitrate_bps = upper_bound_bitrate(bitrate_bps, ladder);
                    } else {
                        // print_red!("t: {:.6} -> no bitrate ladder? ", format_elapsed!(now));
                    }

                    self.last_target_bitrate_bps = bitrate_bps;

                    bitrate_bps
                },
                
                BitrateMode::GCCPort {
                } => {
                    let bitrate_bps = self.gcc_estimator.get_target_bitrate_bps();
                    self.last_target_bitrate_bps = bitrate_bps as f32;
                    self.last_target_bitrate_bps
                }

                BitrateMode::NadaPort {} => {
                    if let Some(last_order_bitrate_mbps) = self.last_nada_target_bitrate_mbps {
                        let bitrate_bps = last_order_bitrate_mbps * 1e6;
                        self.last_target_bitrate_bps = bitrate_bps as f32;

                        self.last_target_bitrate_bps
                    } else {
                        // panic!("WHATS HAPPPPPPPPPPENING CATCH!");

                        // print_dblue!("NADA not configured yet, bitrate : {} Mbps ", self.last_target_bitrate_bps/1e6);
                        self.last_target_bitrate_bps = NADA_INITIAL_RATE as f32;
                        self.last_target_bitrate_bps
                    }
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
