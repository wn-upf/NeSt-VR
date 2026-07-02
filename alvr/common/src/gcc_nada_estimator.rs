#![allow(warnings)]
// use csv::Writer;
// use show_image::glam::f64;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    f64::{INFINITY, NAN},
    time::{self, Duration, Instant},
};
// use chrono::{Utc, TimeZone};
// use crate::{lib::DebugColor, print_brown};
// use crate::lib::models_XR::{MAX_MBPS_LADDER, MIN_MBPS_LADDER};

pub const MAX_MBPS_LADDER: f32 = 100.0;
pub const MIN_MBPS_LADDER: f32 = 10.0; 

use crate::SlidingWindowAverageLegacy;

const DEBUG_GCC: bool = true;

macro_rules! gcc_debug {
    ($fmt:expr, $($arg:tt)*) => {
        if DEBUG_GCC
        {
            let msg = format!($fmt, $($arg)*);
            crate::warn!("{}", msg);
        }
    };
}

pub const GCC_WINDOW_SIZE: usize = 20;
pub const GCC_MIN_CONFIGURED_BITRATE: f64 = 5.0 * 1000. * 1000.; //5Mbps
pub const GCC_MAX_CONFIGURED_BITRATE: f64 = MAX_MBPS_LADDER as f64 * 1000.0 * 1000.0; //100Mbps
pub const GCC_INIT_CONFIGURED_BITRATE: f64 = 15.0 * 1000. * 1000.; //15Mbps
pub const GCC_INCREASE_COEF_ALPHA: f64 = 1.08;
pub const GCC_DECREASE_COEF_BETA: f64 = 0.85;
pub const GCC_DEFAULT_RTT: i64 = 200; //200ms
                                      // pub const GCC_FRAME_INTERVAL: f64 = 1./72. // not const anymore, so it is adaptive to different framerates.
pub const GCC_BITRATE_ESTIMATOR_INIT_WINDOW_MS: i64 = 350; //350ms
pub const GCC_BITRATE_ESTIMATOR_NONINIT_WINDOW_MS: i64 = 250; //250ms

pub struct TimestampGroup {
    pub size: usize,
    pub first_timestamp: i64,
    pub timestamp: i64,
    pub first_arrival_ms: i64,
    pub complete_time_ms: i64,
    pub last_system_time_ms: i64,
}
impl Default for TimestampGroup {
    fn default() -> Self {
        Self {
            size: 0,
            first_timestamp: 0,
            timestamp: 0,
            first_arrival_ms: -1,
            complete_time_ms: -1,
            last_system_time_ms: -1,
        }
    }
}

pub struct InterArrival {
    pub kReorderedResetThreshold: i32,
    pub kArrivalTimeOffsetThresholdMs: i64,
    pub kTimestampGroupLengthTicks: i32,
    pub current_timestamp_group_: TimestampGroup,
    pub prev_timestamp_group_: TimestampGroup,
    pub timestamp_to_ms_coeff_: f64,
    pub num_consecutive_reordered_packets_: i64,
}
impl InterArrival {
    pub fn new(timestamp_group_length_ticks: i32, timestamp_to_ms_coeff: f64) -> Self {
        Self {
            kReorderedResetThreshold: 3,
            kArrivalTimeOffsetThresholdMs: 3000,
            kTimestampGroupLengthTicks: timestamp_group_length_ticks,
            current_timestamp_group_: TimestampGroup {
                ..Default::default()
            },
            prev_timestamp_group_: TimestampGroup {
                ..Default::default()
            },
            timestamp_to_ms_coeff_: timestamp_to_ms_coeff,
            num_consecutive_reordered_packets_: 0,
        }
    }

    pub fn ComputeDeltas(
        &mut self,
        timestamp: i64,
        arrival_time_ms: i64,
        system_time_ms: i64,
        packet_size: usize,
        timestamp_delta: &mut i64,
        arrival_time_delta_ms: &mut i64,
        packet_size_delta: &mut i64,
    ) -> bool {
        let mut calculated_deltas = false;
        if self.current_timestamp_group_.complete_time_ms == -1 {
            self.current_timestamp_group_.timestamp = timestamp;
            self.current_timestamp_group_.first_timestamp = timestamp;
            self.current_timestamp_group_.first_arrival_ms = arrival_time_ms;
        } else if !self.PacketInOrder(timestamp) {
            return false;
        } else if self.NewTimestampGroup(arrival_time_ms, timestamp) {
            // First packet of a later frame, the previous frame sample is ready.
            if self.prev_timestamp_group_.complete_time_ms >= 0 {
                *timestamp_delta =
                    self.current_timestamp_group_.timestamp - self.prev_timestamp_group_.timestamp;
                *arrival_time_delta_ms = self.current_timestamp_group_.complete_time_ms
                    - self.prev_timestamp_group_.complete_time_ms;
                // Check system time differences to see if we have an unproportional jump
                // in arrival time. In that case reset the inter-arrival computations.
                let mut system_time_delta_ms = self.current_timestamp_group_.last_system_time_ms
                    - self.prev_timestamp_group_.last_system_time_ms;
                if *arrival_time_delta_ms - system_time_delta_ms
                    >= self.kArrivalTimeOffsetThresholdMs
                {
                    self.Reset();
                    return false;
                }
                if *arrival_time_delta_ms < 0 {
                    // The group of packets has been reordered since receiving its local
                    // arrival timestamp.
                    self.num_consecutive_reordered_packets_ += 1;
                    if self.num_consecutive_reordered_packets_
                        >= self.kReorderedResetThreshold as i64
                    {
                        self.Reset();
                    }
                    return false;
                } else {
                    self.num_consecutive_reordered_packets_ = 0;
                }
                //RTC_DCHECK_GE(*arrival_time_delta_ms, 0);
                *packet_size_delta = self.current_timestamp_group_.size as i64
                    - self.prev_timestamp_group_.size as i64;
                calculated_deltas = true;
            }
            self.prev_timestamp_group_.complete_time_ms =
                self.current_timestamp_group_.complete_time_ms;
            self.prev_timestamp_group_.first_arrival_ms =
                self.current_timestamp_group_.first_arrival_ms;
            self.prev_timestamp_group_.first_timestamp =
                self.current_timestamp_group_.first_timestamp;
            self.prev_timestamp_group_.last_system_time_ms =
                self.current_timestamp_group_.last_system_time_ms;
            self.prev_timestamp_group_.size = self.current_timestamp_group_.size;
            self.prev_timestamp_group_.timestamp = self.current_timestamp_group_.timestamp;
            // The new timestamp is now the current frame.
            self.current_timestamp_group_.first_timestamp = timestamp;
            self.current_timestamp_group_.timestamp = timestamp;
            self.current_timestamp_group_.first_arrival_ms = arrival_time_ms;
            self.current_timestamp_group_.size = 0;
        } else {
            if self.current_timestamp_group_.timestamp < timestamp {
                self.current_timestamp_group_.timestamp = timestamp
            }
        }
        // Accumulate the frame size.
        self.current_timestamp_group_.size += packet_size;
        self.current_timestamp_group_.complete_time_ms = arrival_time_ms;
        self.current_timestamp_group_.last_system_time_ms = system_time_ms;

        return calculated_deltas;
    }

    pub fn PacketInOrder(&mut self, timestamp: i64) -> bool {
        if self.current_timestamp_group_.complete_time_ms == -1 {
            return true;
        } else {
            let timestamp_diff = timestamp - self.current_timestamp_group_.first_timestamp;
            return timestamp_diff < 0x80000000;
        }
    }

    pub fn NewTimestampGroup(&mut self, arrival_time_ms: i64, timestamp: i64) -> bool {
        if self.current_timestamp_group_.complete_time_ms == -1 {
            return false;
        } else if self.BelongsToBurst(arrival_time_ms, timestamp) {
            return false;
        } else {
            let timestamp_diff = timestamp - self.current_timestamp_group_.first_timestamp;
            return timestamp_diff > self.kTimestampGroupLengthTicks as i64;
        }
    }

    pub fn Reset(&mut self) {
        self.num_consecutive_reordered_packets_ = 0;
        self.current_timestamp_group_ = TimestampGroup {
            ..Default::default()
        };
        self.prev_timestamp_group_ = TimestampGroup {
            ..Default::default()
        };
    }

    pub fn BelongsToBurst(&mut self, arrival_time_ms: i64, timestamp: i64) -> bool {
        //RTC_DCHECK_GE(current_timestamp_group_.complete_time_ms, 0);
        // let mut arrival_time_delta_ms =
        // arrival_time_ms - self.current_timestamp_group_.complete_time_ms;
        // let mut timestamp_diff = timestamp - self.current_timestamp_group_.timestamp;
        // let mut ts_delta_ms = (self.timestamp_to_ms_coeff_ * timestamp_diff as f64 + 0.5) as i64;
        // if ts_delta_ms == 0{
        //     return true;
        // }

        // let mut propagation_delta_ms = arrival_time_delta_ms - ts_delta_ms;
        // if propagation_delta_ms < 0 &&
        // arrival_time_delta_ms <= 5 &&
        // arrival_time_ms - self.current_timestamp_group_.first_arrival_ms <
        // 100{
        //     return true;
        // }

        return false;
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum BandwidthUsage {
    kBwNormal = 0,
    kBwUnderusing = 1,
    kBwOverusing = 2,
    kLast,
}

pub struct PacketTiming {
    pub arrival_time_ms: f64,
    pub smoothed_delay_ms: f64,
    pub raw_delay_ms: f64,
}
impl PacketTiming {
    fn new(arrival_time_ms: f64, smoothed_delay_ms: f64, raw_delay_ms: f64) -> Self {
        Self {
            arrival_time_ms,
            smoothed_delay_ms,
            raw_delay_ms,
        }
    }
}

pub fn LinearFitSlope(packets: &VecDeque<PacketTiming>) -> Option<f64> {
    if packets.len() > 2 {
        // Compute the "center of mass".
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for packet in packets {
            sum_x += packet.arrival_time_ms;
            sum_y += packet.smoothed_delay_ms;
        }
        let x_avg = sum_x / packets.len() as f64;
        let y_avg = sum_y / packets.len() as f64;
        // Compute the slope k = \sum (x_i-x_avg)(y_i-y_avg) / \sum (x_i-x_avg)^2
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for packet in packets {
            let x = packet.arrival_time_ms;
            let y = packet.smoothed_delay_ms;
            numerator += (x - x_avg) * (y - y_avg);
            denominator += (x - x_avg) * (x - x_avg);
        }
        if denominator == 0.0 {
            return Option::None;
        }

        return Some(numerator / denominator);
    } else {
        return Option::None;
    }
}

pub struct TrendlineEstimator {
    //TrendlineEstimatorSettings settings_;
    pub smoothing_coef_: f64,
    pub threshold_gain_: f64,
    // Used by the existing threshold.
    pub num_of_deltas_: i64,
    // Keep the arrival times small by using the change from the first packet.
    pub first_arrival_time_ms_: i64,
    // Exponential backoff filtering.
    pub accumulated_delay_: f64,
    pub smoothed_delay_: f64,
    // Linear least squares regression.
    pub k_up_: f64,
    pub k_down_: f64,
    pub overusing_time_threshold_: f64,
    pub threshold_: f64,
    pub prev_modified_trend_: f64,
    pub last_update_ms_: i64,
    pub prev_trend_: f64,
    pub time_over_using_: f64,
    pub overuse_counter_: i64,
    pub hypothesis_: BandwidthUsage,
    pub hypothesis_predicted_: BandwidthUsage,
    //NetworkStatePredictor* network_state_predictor_;
    pub delay_hist_: VecDeque<PacketTiming>,
    pub current_trend_for_testing: f64,
    pub current_threshold_for_testing: f64,
}
impl TrendlineEstimator {
    pub fn new() -> Self {
        Self {
            smoothing_coef_: 0.9,
            threshold_gain_: 4.0,
            // Used by the existing threshold.
            num_of_deltas_: 0,
            // Keep the arrival times small by using the change from the first packet.
            first_arrival_time_ms_: -1,
            // Exponential backoff filtering.
            accumulated_delay_: 0.0,
            smoothed_delay_: 0.0,
            // Linear least squares regression.
            k_up_: 0.0087,
            k_down_: 0.039,
            overusing_time_threshold_: 10.0,
            threshold_: 12.5, //12.5
            prev_modified_trend_: NAN,
            last_update_ms_: -1,
            prev_trend_: 0.0,
            time_over_using_: -1.0,
            overuse_counter_: 0,
            hypothesis_: BandwidthUsage::kBwNormal,
            hypothesis_predicted_: BandwidthUsage::kBwNormal,
            //NetworkStatePredictor* network_state_predictor_;
            delay_hist_: VecDeque::new(),
            current_threshold_for_testing: 0.0,
            current_trend_for_testing: 0.0,
        }
    }

    pub fn UpdateThreshold(&mut self, modified_trend: f64, now_ms: i64) {
        if self.last_update_ms_ == -1 {
            self.last_update_ms_ = now_ms;
        }

        if modified_trend.abs() > self.threshold_ + 15.0 {
            // Avoid adapting the threshold to big latency spikes, caused e.g.,
            // by a sudden capacity drop.
            gcc_debug!(
                "[Thresh] spike: |mod_trend|={:.2} >> thr+15 ({}). skip adapt",
                modified_trend.abs(),
                self.threshold_
            );
            self.last_update_ms_ = now_ms;
            return;
        }
        let k = if modified_trend.abs() < self.threshold_ {
            self.k_down_
        } else {
            self.k_up_
        };

        let kMaxTimeDeltaMs = 100;
        let time_delta_ms = std::cmp::min(now_ms - self.last_update_ms_, kMaxTimeDeltaMs);
        let prev = self.threshold_.clone();

        self.threshold_ += k * (modified_trend.abs() - self.threshold_) * time_delta_ms as f64;
        if self.threshold_ > 600.0 as f64 {
            self.threshold_ = 600.0;
        } else if self.threshold_ < 6.0 as f64 {
            self.threshold_ = 6.0;
        }
        self.last_update_ms_ = now_ms;

        gcc_debug!(
            "[Thresh] dt={} ms, k={:.4}, old={:.2} → new={:.2}",
            time_delta_ms as i64,
            k,
            prev,
            self.threshold_
        );
    }

    pub fn Detect(&mut self, trend: f64, ts_delta: f64, now_ms: i64) {
        if self.num_of_deltas_ < 2 {
            gcc_debug!(
                "[Detect] warmup: deltas={} → hyp=Normal",
                self.num_of_deltas_
            );
            self.hypothesis_ = BandwidthUsage::kBwNormal;
            return;
        }
        let modified_trend =
            std::cmp::min(self.num_of_deltas_, 60) as f64 * trend * self.threshold_gain_;
        let prev = self.hypothesis_.clone();
        self.prev_modified_trend_ = modified_trend;

        if modified_trend > self.threshold_ {
            if self.time_over_using_ == -1.0 {
                // Initialize the timer. Assume that we've been
                // over-using half of the time since the previous
                // sample.
                self.time_over_using_ = ts_delta / 2.0;
            } else {
                // Increment timer
                self.time_over_using_ += ts_delta;
            }
            self.overuse_counter_ += 1;
            if (self.time_over_using_ > self.overusing_time_threshold_ && self.overuse_counter_ > 1)
            {
                if trend >= self.prev_trend_ {
                    self.time_over_using_ = 0.0;
                    self.overuse_counter_ = 0;
                    self.hypothesis_ = BandwidthUsage::kBwOverusing;
                }
            }
        } else if modified_trend < -self.threshold_ {
            self.time_over_using_ = -1.0;
            self.overuse_counter_ = 0;
            self.hypothesis_ = BandwidthUsage::kBwUnderusing;
        } else {
            self.time_over_using_ = -1.0;
            self.overuse_counter_ = 0;
            self.hypothesis_ = BandwidthUsage::kBwNormal;
        }

        gcc_debug!(
            "[Detect] mod_trend={:.3} thr={:.2} overuse_t={:.3}s cnt={} hyp={:?} (prev={:?})",
            modified_trend,
            self.threshold_,
            self.time_over_using_,
            self.overuse_counter_,
            self.hypothesis_,
            prev
        );

        self.current_threshold_for_testing = self.threshold_;
        self.current_trend_for_testing = modified_trend;
        self.prev_trend_ = trend;
        self.UpdateThreshold(modified_trend, now_ms);
    }

    pub fn UpdateTrendline(
        &mut self,
        recv_delta_ms: f64,
        send_delta_ms: f64,
        send_time_ms: i64,
        arrival_time_ms: i64,
        packet_size: i64,
    ) {
        let delta_ms = recv_delta_ms - send_delta_ms;
        self.num_of_deltas_ += 1;
        self.num_of_deltas_ = std::cmp::min(self.num_of_deltas_, 1000);
        if self.first_arrival_time_ms_ == -1 {
            self.first_arrival_time_ms_ = arrival_time_ms;
        }

        // Exponential backoff filter.
        self.accumulated_delay_ += delta_ms;
        self.smoothed_delay_ = self.smoothing_coef_ * self.smoothed_delay_
            + (1.0 - self.smoothing_coef_) * self.accumulated_delay_;

        // Maintain packet window
        self.delay_hist_.push_back(PacketTiming::new(
            (arrival_time_ms - self.first_arrival_time_ms_) as f64,
            self.smoothed_delay_,
            self.accumulated_delay_,
        ));
        if self.delay_hist_.len() > GCC_WINDOW_SIZE {
            self.delay_hist_.pop_front();
        }

        // Simple linear regression.
        let mut trend = self.prev_trend_;
        if self.delay_hist_.len() == GCC_WINDOW_SIZE {
            // Update trend_ if it is possible to fit a line to the data. The delay
            // trend can be seen as an estimate of (send_rate - capacity)/capacity.
            // 0 < trend < 1   ->  the delay increases, queues are filling up
            //   trend == 0    ->  the delay does not change
            //   trend < 0     ->  the delay decreases, queues are being emptied
            trend = LinearFitSlope(&self.delay_hist_).unwrap_or(trend);
        }

        gcc_debug!(
                    "[Trend] Δms: recv={:.3}, send={:.3}, delta={:.3} | accum={:.3}, smooth={:.3}, trend={:.5}",
                    recv_delta_ms, send_delta_ms, delta_ms,
                    self.accumulated_delay_, self.smoothed_delay_, trend
                );

        self.Detect(trend, send_delta_ms, arrival_time_ms);
    }
}

pub struct LinkCapacityEstimator {
    pub estimate_kbps_: Option<f64>,
    pub deviation_kbps_: f64,
}
impl LinkCapacityEstimator {
    pub fn new() -> Self {
        Self {
            estimate_kbps_: Option::None,
            deviation_kbps_: 0.4,
        }
    }
    pub fn deviation_estimate_kbps(&mut self) -> f64 {
        return (self.deviation_kbps_ * self.estimate_kbps_.unwrap()).sqrt();
    }
    pub fn UpperBound(&mut self) -> f64 {
        if !self.estimate_kbps_.is_none() {
            return (self.estimate_kbps_.unwrap() + 3.0 * self.deviation_estimate_kbps()) * 1000.0;
            //check the unit of deviation estimate kbps
        }

        return f64::INFINITY;
    }

    pub fn LowerBound(&mut self) -> f64 {
        if !self.estimate_kbps_.is_none() {
            return f64::max(
                0.0,
                self.estimate_kbps_.unwrap() - 3.0 * self.deviation_estimate_kbps(),
            ) * 1000.0; //check the unit of deviation estimate kbps
        }

        return 0.0;
    }

    pub fn Reset(&mut self) {
        self.estimate_kbps_ = Option::None;
    }

    pub fn OnOveruseDetected(&mut self, acknowledged_rate: f64) {
        self.Update(acknowledged_rate, 0.05);
    }

    pub fn OnProbeRate(&mut self, probe_rate: f64) {
        self.Update(probe_rate, 0.5);
    }

    pub fn Update(&mut self, capacity_sample: f64, alpha: f64) {
        let mut sample_kbps = capacity_sample * 0.001;
        if self.estimate_kbps_.is_none() {
            self.estimate_kbps_ = Some(sample_kbps);
        } else {
            self.estimate_kbps_ =
                Some((1.0 - alpha) * self.estimate_kbps_.unwrap() + alpha * sample_kbps);
        }
        // Estimate the variance of the link capacity estimate and normalize the
        // variance with the link capacity estimate.
        let norm = f64::max(self.estimate_kbps_.unwrap(), 1.0);
        let mut error_kbps = self.estimate_kbps_.unwrap() - sample_kbps;
        self.deviation_kbps_ =
            (1.0 - alpha) * self.deviation_kbps_ + alpha * error_kbps * error_kbps / norm;
        // 0.4 ~= 14 kbit/s at 500 kbit/s
        // 2.5f ~= 35 kbit/s at 500 kbit/s
        if self.deviation_kbps_ > 2.5 as f64 {
            self.deviation_kbps_ = 2.5;
        } else if self.deviation_kbps_ < 0.4 as f64 {
            self.deviation_kbps_ = 0.4;
        }
    }

    pub fn has_estimate(&mut self) -> bool {
        return !self.estimate_kbps_.is_none();
    }

    pub fn estimate(&mut self) -> f64 {
        return self.estimate_kbps_.unwrap() * 1000.0;
    }
}
pub struct NetworkStateEstimate {
    pub confidence: f64,
    pub update_time: i64,
    pub last_receive_time: i64,
    pub last_send_time: i64,
    pub link_capacity: f64,
    pub link_capacity_lower: f64,
    pub link_capacity_upper: f64,
    pub pre_link_buffer_delay: i64,
    pub post_link_buffer_delay: i64,
    pub propagation_delay: i64,
}
impl Default for NetworkStateEstimate {
    fn default() -> Self {
        Self {
            confidence: f64::NAN,
            update_time: i64::MIN,
            last_receive_time: i64::MIN,
            last_send_time: i64::MIN,
            link_capacity: -std::f64::INFINITY,
            link_capacity_lower: -std::f64::INFINITY,
            link_capacity_upper: -std::f64::INFINITY,
            pre_link_buffer_delay: i64::MIN,
            post_link_buffer_delay: i64::MIN,
            propagation_delay: i64::MIN,
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum RateControlState {
    kRcHold = 0,
    kRcIncrease = 1,
    kRcDecrease = 2,
}
pub struct RateControlInput {
    pub bw_state: BandwidthUsage,
    pub estimated_throughput: Option<f64>,
}
impl RateControlInput {
    pub fn new(bw_state: BandwidthUsage, estimated_throughput: Option<f64>) -> Self {
        Self {
            bw_state: bw_state,
            estimated_throughput: estimated_throughput,
        }
    }
}

pub struct AimdRateControl {
    pub min_configured_bitrate_: f64,
    pub max_configured_bitrate_: f64,
    pub current_bitrate_: f64,
    pub latest_estimated_throughput_: f64,
    pub link_capacity_: LinkCapacityEstimator,
    pub network_estimate_: Option<NetworkStateEstimate>,
    pub rate_control_state_: RateControlState,
    pub time_last_bitrate_change_: i64,
    pub time_last_bitrate_decrease_: i64,
    pub time_first_throughput_estimate_: i64,
    pub bitrate_is_initialized_: bool,
    pub beta_: f64,
    pub in_alr_: bool,
    pub rtt_: i64,
    pub send_side_: bool,
    pub no_bitrate_increase_in_alr_: bool,
    pub last_decrease_: Option<f64>,
    pub esitmate_thr_testing: f64,
    pub gcc_frame_interval: f64,
}
impl AimdRateControl {
    pub fn new(send_side: bool, framerate: f64) -> Self {
        Self {
            min_configured_bitrate_: GCC_MIN_CONFIGURED_BITRATE,
            max_configured_bitrate_: GCC_MAX_CONFIGURED_BITRATE,
            current_bitrate_: GCC_INIT_CONFIGURED_BITRATE,
            latest_estimated_throughput_: GCC_INIT_CONFIGURED_BITRATE,
            link_capacity_: LinkCapacityEstimator::new(),
            rate_control_state_: RateControlState::kRcHold,
            time_last_bitrate_change_: i64::MIN,
            time_last_bitrate_decrease_: i64::MIN,
            time_first_throughput_estimate_: i64::MIN,
            bitrate_is_initialized_: false,
            beta_: GCC_DECREASE_COEF_BETA,
            in_alr_: false,
            rtt_: GCC_DEFAULT_RTT,
            send_side_: send_side,
            no_bitrate_increase_in_alr_: true,
            last_decrease_: Some(0.0),
            network_estimate_: Some(NetworkStateEstimate {
                ..Default::default()
            }),
            esitmate_thr_testing: 0.0,
            gcc_frame_interval: 1.0 / framerate,
        }
    }
    pub fn ChangeState(&mut self, input: &RateControlInput, at_time: i64) {
        match input.bw_state {
            BandwidthUsage::kBwNormal => {
                if self.rate_control_state_ == RateControlState::kRcHold {
                    self.time_last_bitrate_change_ = at_time;
                    self.rate_control_state_ = RateControlState::kRcIncrease;
                } else if self.rate_control_state_ == RateControlState::kRcDecrease {
                    self.rate_control_state_ = RateControlState::kRcHold;
                }
            }
            BandwidthUsage::kBwOverusing => {
                if self.rate_control_state_ != RateControlState::kRcDecrease {
                    self.rate_control_state_ = RateControlState::kRcDecrease;
                }
            }
            BandwidthUsage::kBwUnderusing => {
                self.rate_control_state_ = RateControlState::kRcHold;
            }
            _ => {}
        }
    }
    pub fn GetNearMaxIncreaseRateBpsPerSecond(&mut self) -> f64 {
        //RTC_DCHECK(!current_bitrate_.IsZero());
        let kFrameInterval = self.gcc_frame_interval;
        let frame_size = self.current_bitrate_ * kFrameInterval;
        let kPacketSize = 1500 * 8;
        let packets_per_frame = frame_size as f64 / kPacketSize as f64;
        let avg_packet_size = frame_size / packets_per_frame;

        // Approximate the over-use estimator delay to 100 ms.
        let mut response_time = (self.rtt_ + 100) as f64 * 0.001;

        response_time = response_time * 2.;
        let increase_rate_bps_per_second = avg_packet_size / response_time as f64;
        let kMinIncreaseRateBpsPerSecond = 4000.0;
        return f64::max(kMinIncreaseRateBpsPerSecond, increase_rate_bps_per_second);
    }

    pub fn MultiplicativeRateIncrease(
        &mut self,
        at_time: i64,
        last_time: i64,
        current_bitrate: f64,
    ) -> f64 {
        let mut alpha = GCC_INCREASE_COEF_ALPHA;
        if last_time == i64::MIN {
            let time_since_last_update = at_time - last_time;
            alpha = alpha.powf(((time_since_last_update as f64 / 1000.0).min(1.0)) as f64);
        }
        let multiplicative_increase = f64::max(current_bitrate * (alpha - 1.0), 1000.0);
        return multiplicative_increase;
    }

    pub fn AdditiveRateIncrease(&mut self, at_time: i64, last_time: i64) -> f64 {
        let time_period_seconds = ((at_time - last_time) as f64) / 1000.0;
        let data_rate_increase_bps =
            self.GetNearMaxIncreaseRateBpsPerSecond() * time_period_seconds;
        return data_rate_increase_bps;
    }

    pub fn Update(&mut self, input: &RateControlInput, at_time: i64) -> f64 {
        // Set the initial bit rate value to what we're receiving the first half
        // second.
        // TODO(bugs.webrtc.org/9379): The comment above doesn't match to the code.
        if !self.bitrate_is_initialized_ {
            let kInitializationTime = 5000 as i64;
            //RTC_DCHECK_LE(kBitrateWindowMs, kInitializationTime.ms());
            if self.time_first_throughput_estimate_ == i64::MIN {
                if input.estimated_throughput.is_some() {
                    self.time_first_throughput_estimate_ = at_time;
                }
            } else if at_time - self.time_first_throughput_estimate_ > kInitializationTime
                && input.estimated_throughput.is_some()
            {
                self.current_bitrate_ = input.estimated_throughput.unwrap();
                self.bitrate_is_initialized_ = true;
            }
        }

        self.ChangeBitrate(input, at_time);
        return self.current_bitrate_;
    }

    pub fn ClampBitrate(&mut self, new_bitrate: f64) -> f64 {
        let mut new_bitrate_r = new_bitrate;
        if self.network_estimate_.is_some()
            && self.network_estimate_.as_mut().unwrap().link_capacity_upper != -std::f64::INFINITY
        {
            let upper_bound = self.network_estimate_.as_mut().unwrap().link_capacity_upper;
            new_bitrate_r = f64::min(upper_bound, new_bitrate_r);
        }
        if self.network_estimate_.is_some()
            && self.network_estimate_.as_mut().unwrap().link_capacity_lower != -std::f64::INFINITY
            && new_bitrate_r < self.current_bitrate_
        {
            new_bitrate_r = f64::min(
                self.current_bitrate_,
                f64::max(
                    new_bitrate_r,
                    self.network_estimate_.as_mut().unwrap().link_capacity_lower * self.beta_,
                ),
            );
        }
        new_bitrate_r = f64::max(new_bitrate_r, self.min_configured_bitrate_);
        return new_bitrate_r;
    }

    pub fn ChangeBitrate(&mut self, input: &RateControlInput, at_time: i64) {
        let mut new_bitrate = Option::None;
        let mut estimated_throughput = input
            .estimated_throughput
            .unwrap_or(self.latest_estimated_throughput_);
        if input.estimated_throughput.is_some() {
            self.latest_estimated_throughput_ = input.estimated_throughput.unwrap();
        }

        // An over-use should always trigger us to reduce the bitrate, even though
        // we have not yet established our first estimate. By acting on the over-use,
        // we will end up with a valid estimate.
        if !self.bitrate_is_initialized_ && input.bw_state != BandwidthUsage::kBwOverusing {
            return;
        }

        self.ChangeState(input, at_time);

        match self.rate_control_state_ {
            RateControlState::kRcHold => {
                // print_brown!("GCC STATE -> HOLD", );
            }
            RateControlState::kRcIncrease => {
                // print_brown!("GCC STATE -> INCREASE", );

                if estimated_throughput > self.link_capacity_.UpperBound() {
                    self.link_capacity_.Reset();
                }
                let mut increase_limit = 1.5 * estimated_throughput + 10000.0;

                if self.current_bitrate_ < increase_limit {
                    let mut increased_bitrate = -std::f64::INFINITY;
                    if self.link_capacity_.has_estimate() {
                        let additive_increase =
                            self.AdditiveRateIncrease(at_time, self.time_last_bitrate_change_);
                        increased_bitrate = self.current_bitrate_ + additive_increase;
                    } else {
                        let multiplicative_increase = self.MultiplicativeRateIncrease(
                            at_time,
                            self.time_last_bitrate_change_,
                            self.current_bitrate_,
                        );
                        increased_bitrate = self.current_bitrate_ + multiplicative_increase;
                    }
                    new_bitrate = Some(increased_bitrate.min(increase_limit));
                }
                self.time_last_bitrate_change_ = at_time;
            }
            RateControlState::kRcDecrease => {
                // print_brown!("GCC STATE -> DECREASE",);

                let mut decreased_bitrate = std::f64::INFINITY;

                decreased_bitrate = estimated_throughput * self.beta_;
                if decreased_bitrate > self.current_bitrate_ {
                    if self.link_capacity_.has_estimate() {
                        decreased_bitrate = self.beta_ * self.link_capacity_.estimate();
                    }
                }

                if decreased_bitrate < self.current_bitrate_ {
                    new_bitrate = Some(decreased_bitrate);
                }

                if self.bitrate_is_initialized_ && estimated_throughput < self.current_bitrate_ {
                    if let Some(new_br) = new_bitrate {
                        self.last_decrease_ = Some(self.current_bitrate_ - new_br);
                    } else {
                        self.last_decrease_ = Option::None;
                    }
                }

                if estimated_throughput < self.link_capacity_.LowerBound() {
                    self.link_capacity_.Reset();
                }

                self.bitrate_is_initialized_ = true;
                self.esitmate_thr_testing = estimated_throughput;
                self.link_capacity_.OnOveruseDetected(estimated_throughput);
                self.rate_control_state_ = RateControlState::kRcHold;
                self.time_last_bitrate_change_ = at_time;
                self.time_last_bitrate_decrease_ = at_time;
            }
            _ => {}
        }
        if self.network_estimate_.is_none() {
            self.network_estimate_ = Some(NetworkStateEstimate {
                ..Default::default()
            });
        }
        if self.link_capacity_.LowerBound() != 0.0 {
            if self.link_capacity_.LowerBound() < self.min_configured_bitrate_ {
                self.network_estimate_.as_mut().unwrap().link_capacity_lower =
                    self.min_configured_bitrate_;
            } else {
                self.network_estimate_.as_mut().unwrap().link_capacity_lower =
                    self.link_capacity_.LowerBound();
            }
        } else {
            self.network_estimate_.as_mut().unwrap().link_capacity_lower =
                self.min_configured_bitrate_;
        }
        if self.link_capacity_.UpperBound() != f64::INFINITY {
            if self.link_capacity_.UpperBound() > self.max_configured_bitrate_ {
                self.network_estimate_.as_mut().unwrap().link_capacity_upper =
                    self.max_configured_bitrate_;
            } else {
                self.network_estimate_.as_mut().unwrap().link_capacity_upper =
                    self.link_capacity_.UpperBound();
            }
        } else {
            self.network_estimate_.as_mut().unwrap().link_capacity_upper =
                self.max_configured_bitrate_;
        }

        self.current_bitrate_ = self.ClampBitrate(new_bitrate.unwrap_or(self.current_bitrate_));
    }
}

pub struct BitrateEstimator {
    pub sum_: i64,
    pub initial_window_ms_: i64,
    pub noninitial_window_ms_: i64,
    pub uncertainty_scale_: f64,
    pub uncertainty_scale_in_alr_: f64,
    pub small_sample_uncertainty_scale_: f64,
    pub small_sample_threshold_: usize,
    pub uncertainty_symmetry_cap_: usize,
    pub estimate_floor_: usize,
    pub current_window_ms_: i64,
    pub prev_time_ms_: i64,
    pub bitrate_estimate_kbps_: f64,
    pub bitrate_estimate_var_: f64,
}
impl BitrateEstimator {
    pub fn new() -> Self {
        Self {
            sum_: 0,
            initial_window_ms_: GCC_BITRATE_ESTIMATOR_INIT_WINDOW_MS,
            noninitial_window_ms_: GCC_BITRATE_ESTIMATOR_NONINIT_WINDOW_MS,
            uncertainty_scale_: 10.0,
            uncertainty_scale_in_alr_: 10.0,
            small_sample_uncertainty_scale_: 10.0,
            small_sample_threshold_: 0,
            uncertainty_symmetry_cap_: 0,
            estimate_floor_: 0,
            current_window_ms_: 0,
            prev_time_ms_: -1,
            bitrate_estimate_kbps_: -1.0,
            bitrate_estimate_var_: 50.0,
        }
    }
    pub fn UpdateWindow(
        &mut self,
        now_ms: i64,
        bytes: usize,
        rate_window_ms: i64,
        is_small_sample: &mut bool,
    ) -> f64 {
        if now_ms < self.prev_time_ms_ {
            self.prev_time_ms_ = -1;
            self.sum_ = 0;
            self.current_window_ms_ = 0;
        }
        if self.prev_time_ms_ >= 0 {
            self.current_window_ms_ += now_ms - self.prev_time_ms_;
            // Reset if nothing has been received for more than a full window.
            if now_ms - self.prev_time_ms_ > rate_window_ms {
                self.sum_ = 0;
                self.current_window_ms_ %= rate_window_ms;
            }
        }
        self.prev_time_ms_ = now_ms;
        let mut bitrate_sample = -1.0;
        if self.current_window_ms_ >= rate_window_ms {
            *is_small_sample = self.sum_ < self.small_sample_threshold_ as i64;
            bitrate_sample = 8.0 * self.sum_ as f64 / rate_window_ms as f64;
            self.current_window_ms_ -= rate_window_ms;
            self.sum_ = 0;
        }
        self.sum_ += bytes as i64;
        return bitrate_sample;
    }

    pub fn Update(&mut self, at_time: i64, amount: usize, in_alr: bool) {
        let mut rate_window_ms = self.noninitial_window_ms_;
        // We use a larger window at the beginning to get a more stable sample that
        // we can use to initialize the estimate.
        if self.bitrate_estimate_kbps_ < 0.0 {
            rate_window_ms = self.initial_window_ms_;
        }

        let mut is_small_sample = false;
        let mut bitrate_sample_kbps =
            self.UpdateWindow(at_time, amount, rate_window_ms, &mut is_small_sample);
        if bitrate_sample_kbps < 0.0 {
            return;
        }

        if self.bitrate_estimate_kbps_ < 0.0 {
            // This is the very first sample we get. Use it to initialize the estimate.
            self.bitrate_estimate_kbps_ = bitrate_sample_kbps;
            return;
        }
        // Optionally use higher uncertainty for very small samples to avoid dropping
        // estimate and for samples obtained in ALR.
        let mut scale = self.uncertainty_scale_;
        if is_small_sample && bitrate_sample_kbps < self.bitrate_estimate_kbps_ {
            scale = self.small_sample_uncertainty_scale_;
        } else if in_alr && bitrate_sample_kbps < self.bitrate_estimate_kbps_ {
            // Optionally use higher uncertainty for samples obtained during ALR.
            scale = self.uncertainty_scale_in_alr_;
        }
        // Define the sample uncertainty as a function of how far away it is from the
        // current estimate. With low values of uncertainty_symmetry_cap_ we add more
        // uncertainty to increases than to decreases. For higher values we approach
        // symmetry.
        let mut sample_uncertainty = scale
            * (self.bitrate_estimate_kbps_ - bitrate_sample_kbps).abs()
            / (self.bitrate_estimate_kbps_
                + f64::min(bitrate_sample_kbps, self.uncertainty_symmetry_cap_ as f64));

        let mut sample_var = sample_uncertainty * sample_uncertainty;
        // Update a bayesian estimate of the rate, weighting it lower if the sample
        // uncertainty is large.
        // The bitrate estimate uncertainty is increased with each update to model
        // that the bitrate changes over time.
        let mut pred_bitrate_estimate_var = self.bitrate_estimate_var_ + 5.0;
        self.bitrate_estimate_kbps_ = (sample_var * self.bitrate_estimate_kbps_
            + pred_bitrate_estimate_var * bitrate_sample_kbps)
            / (sample_var + pred_bitrate_estimate_var);
        self.bitrate_estimate_kbps_ =
            f64::max(self.bitrate_estimate_kbps_, self.estimate_floor_ as f64);
        self.bitrate_estimate_var_ =
            sample_var * pred_bitrate_estimate_var / (sample_var + pred_bitrate_estimate_var);
        //BWE_TEST_LOGGING_PLOT(1, "acknowledged_bitrate", at_time.ms(),
        //bitrate_estimate_kbps_ * 1000);
    }
    pub fn bitrate(&mut self) -> Option<f64> {
        if self.bitrate_estimate_kbps_ < 0.0 {
            return Option::None;
        } else {
            return Some(self.bitrate_estimate_kbps_ * 1000.0);
        }
    }
    pub fn PeekRate(&mut self) -> Option<f64> {
        if self.current_window_ms_ > 0 {
            return Some(self.sum_ as f64 / (self.current_window_ms_ as f64 * 0.001));
        } else {
            return Option::None;
        }
    }
    pub fn ExpectFastRateChange(&mut self) {
        self.bitrate_estimate_var_ += 200.0;
    }
}

pub struct GccBandwidthEstimator {
    pub trendline_manager: TrendlineEstimator,
    pub aimd_manager: AimdRateControl,
    pub rate_control_input_manager: RateControlInput,
    pub bitrate_estimator_manager: BitrateEstimator,
    pub last_frame_send_timestamp: f64,
    pub last_frame_arrival_timestamp: f64,

    pub t0: Instant, 
}
impl GccBandwidthEstimator {
    pub fn new(framerate: f64, now: Instant,) -> Self {
        Self {
            trendline_manager: TrendlineEstimator::new(),
            aimd_manager: AimdRateControl::new(true, framerate),
            rate_control_input_manager: RateControlInput::new(
                BandwidthUsage::kBwNormal,
                Some(GCC_INIT_CONFIGURED_BITRATE),
            ),
            bitrate_estimator_manager: BitrateEstimator::new(),
            last_frame_send_timestamp: 0.,
            last_frame_arrival_timestamp: 0.,
            t0: now, 
        }   
    }

    pub fn Update(
        &mut self,
        current_frame_send_timestamp_us: f64,
        current_frame_arrival_timestamp_us: f64,
        current_frame_size: i64,
        now: Instant,
    ) -> f64 {
        gcc_debug!(
            "[GCC] Update: send_ts_us={:.0} arr_ts_us={:.0} size={}B",
            current_frame_send_timestamp_us,
            current_frame_arrival_timestamp_us,
            current_frame_size
        );

        let mut send_delta_ms = 0.0;
        let mut recv_delta_ms = 0.0;

        if self.last_frame_send_timestamp != 0. {
            send_delta_ms =
                (current_frame_send_timestamp_us - self.last_frame_send_timestamp) * 0.001; // * 0.001 converting to ms, meaning that input is expected to be microsecs.
            recv_delta_ms =
                (current_frame_arrival_timestamp_us - self.last_frame_arrival_timestamp) * 0.001;
        }
        gcc_debug!(
            "[GCC] Δsend={:.3} ms, Δrecv={:.3} ms (since last frame)",
            send_delta_ms,
            recv_delta_ms
        );
        self.last_frame_send_timestamp = current_frame_send_timestamp_us;
        self.last_frame_arrival_timestamp = current_frame_arrival_timestamp_us;
        let send_time_ms = (current_frame_send_timestamp_us * 0.001) as i64;
        let arrival_time_ms = (current_frame_arrival_timestamp_us * 0.001) as i64;
        let packet_size = current_frame_size;

        self.trendline_manager.UpdateTrendline(
            recv_delta_ms,
            send_delta_ms,
            send_time_ms,
            arrival_time_ms,
            packet_size,
        );

        gcc_debug!(
            "[Trend] smoothed={:.3} ms, accum={:.3} ms, win={}, trend={:.5}, mod_trend={:.3}, thr={:.2}, hyp={:?}",
            self.trendline_manager.smoothed_delay_,
            self.trendline_manager.accumulated_delay_,
            self.trendline_manager.delay_hist_.len(),
            self.trendline_manager.prev_trend_,
            self.trendline_manager.current_trend_for_testing,
            self.trendline_manager.current_threshold_for_testing,
            self.trendline_manager.hypothesis_
        );

        if self.trendline_manager.hypothesis_ == BandwidthUsage::kBwNormal {
            self.rate_control_input_manager.bw_state = BandwidthUsage::kBwNormal;
        } else if self.trendline_manager.hypothesis_ == BandwidthUsage::kBwOverusing {
            self.rate_control_input_manager.bw_state = BandwidthUsage::kBwOverusing;
        } else {
            self.rate_control_input_manager.bw_state = BandwidthUsage::kBwUnderusing;
        }

        self.bitrate_estimator_manager
            .Update(arrival_time_ms, packet_size as usize, false);

        if let Some(br) = self.bitrate_estimator_manager.bitrate() {
            self.rate_control_input_manager.estimated_throughput = Some(br);
            gcc_debug!(
                "[RateEst] sample={} kbps, peek={:?} kbps",
                br / 1000.0,
                self.bitrate_estimator_manager
                    .PeekRate()
                    .map(|v| v / 1000.0)
            );
        } else {
            gcc_debug!("[RateEst] sample=∅ (insufficient window)",);
        }

        // let at_time = (Utc::now().timestamp_micros() as f64 * 0.001) as i64;

        let at_time = now.duration_since(self.t0).as_millis() as i64; //retrieve current timestamp as millis

        gcc_debug!(
            "[AIMD] input: bw_state={:?}, est_thr={:?} kbps, cur={} kbps",
            self.rate_control_input_manager.bw_state,
            self.rate_control_input_manager
                .estimated_throughput
                .map(|v| v / 1000.0),
            self.aimd_manager.current_bitrate_ / 1000.0
        );

        let target_bitrate_bps = self
            .aimd_manager
            .Update(&self.rate_control_input_manager, at_time);
        gcc_debug!(
            "[AIMD] state={:?}, new={} kbps, bounds=[{:.0},{:.0}] kbps, last_dec={:?} kbps",
            self.aimd_manager.rate_control_state_,
            target_bitrate_bps / 1000.0,
            self.aimd_manager
                .network_estimate_
                .as_ref()
                .map(|n| n.link_capacity_lower / 1000.0)
                .unwrap_or(0.0),
            self.aimd_manager
                .network_estimate_
                .as_ref()
                .map(|n| n.link_capacity_upper / 1000.0)
                .unwrap_or(0.0),
            self.aimd_manager.last_decrease_.map(|v| v / 1000.0)
        );

        return target_bitrate_bps;
    }

    pub fn get_target_bitrate_bps(&self) -> f64 {
        return self.aimd_manager.current_bitrate_;
    }

    pub fn get_estimated_througput(&mut self) -> f64 {
        if self.bitrate_estimator_manager.bitrate().is_some() {
            return self.bitrate_estimator_manager.bitrate().unwrap();
        } else {
            return 0.0;
        }
    }
}

/////////////////////////// NADA IMPLEMENTATION /////////////////////////////////
use chrono::{Local, Utc};

pub const NADA_PARAM_PRIO: f64 = 1.0; //Weight of priority of the flow | 1.0
/**
 * Min and Max rate of application supported by media encoder | 150 Kbps & 1.5 Mbps
 **/
pub const RMCAT_CC_DEFAULT_RMIN: i64 = 2_000_000; // 5Mbps
pub const RMCAT_CC_DEFAULT_RMAX: i64 = GCC_MAX_CONFIGURED_BITRATE as i64; //100Mbps
pub const NADA_INITIAL_RATE: i64 = 15_000_000; //15Mbps
pub const NADA_PARAM_XREF: i64 = 20; //Reference congestion level | 20ms
pub const NADA_PARAM_KAPPA: f64 = 0.5; //Scaling parameter for gradual rate update | 0.5
pub const NADA_PARAM_ETA: f64 = 2.0; //Scaling parameter for gradual rate update | 2.0
pub const NADA_PARAM_TAU: i64 = 500; //Upper bound of RTT in gradual rate update |  500ms
pub const NADA_PARAM_DELTA: i64 = 100; //Target feedback interval | 100ms
pub const NADA_PARAM_DELTA_US: i64 = 100_000; //in Nano second
pub const NADA_PARAM_DFILT: i64 = 120; //Bound on filtering delay | 120ms
pub const NADA_PARAM_DFILT_US: i64 = 120_000; //in Nano second
pub const NADA_PARAM_LOGWIN: i64 = 500; //Observation time window in for calculating packet summary statistics at receiver  | 500ms       |
pub const NADA_PARAM_QEPS: i64 = 10; //Threshold for determining queuing delay build up at receiver| 10ms
pub const NADA_PARAM_QEPS_US: i64 = 10_000; //in Nano second

pub const NADA_PARAM_QTH: i64 = 50; //Delay threshold for non-linear warping | 50ms
pub const NADA_PARAM_QMAX: i64 = 400; //Delay upper bound for non-linear warping| 400ms
pub const NADA_PARAM_DLOSS: i64 = 10; //1_000; //Delay penalty for loss  | 1.0s
pub const NADA_PARAM_DMARK: i64 = 200; //Delay penalty for ECN marking | 200ms

pub const NADA_PARAM_GAMMA_MAX: f64 = 0.5; //Upper bound on rate increase ratio for accelerated ramp-up | 50%
pub const NADA_PARAM_QBOUND: i64 = 50; //Upper bound on self-inflicted queuing delay during ramp up | 50ms

pub const NADA_PARAM_FPS: f64 = 30.0; //Frame rate of incoming video | 30
pub const NADA_PARAM_BETA_S: f64 = 0.1; //Scaling parameter for modulating outgoing sending rate |  0.1
pub const NADA_PARAM_BETA_V: f64 = 0.1; //Scaling parameter for modulating video encoder target rate | 0.1
pub const NADA_PARAM_ALPHA: f64 = 0.1; // Smoothing factor of loss and marking ratios | 0.1

//Added by Ze
pub const NADA_PARAM_LAMBDA: f64 = 0.5;
pub const NADA_PARAM_MULTILOSS: f64 = 7.0;
pub const NADA_PARAM_PLRREF: f64 = 0.01;
pub const NADA_PARAM_XMAX: f64 = 500.0;

//History Size
pub const NADA_RTT_HISTORY_SIZE: usize = 15;

//RTCP NADA Feedback Report, from NADA Receiver
#[derive(Debug)]
pub struct NADAFeedbackReport {
    pub rmode: i8,
    pub x_curr: f64,
    pub r_recv: i64,

    //Only For Debuging NADA Receiver
    pub d_queue: i64,
    pub d_tilde: f64,
    pub p_loss: f64,
}

impl NADAFeedbackReport {
    pub fn new(
        rmode: i8,
        x_curr: f64,
        r_recv: i64,
        d_queue: i64,
        d_tilde: f64,
        p_loss: f64,
    ) -> Self {
        Self {
            rmode: rmode,
            x_curr: x_curr,
            r_recv: r_recv,

            //Only For Debuging NADA Receiver
            d_queue: d_queue,
            d_tilde: d_tilde,
            p_loss: p_loss,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum RateUpdateMode {
    AcceleratedRampUp = 0, // Corresponds to rmode = 0
    GradualUpdate = 1,     // Corresponds to rmode = 1
}

pub struct NadaSender {
    pub r_ref: i64, //Reference rate based on network congestion
    pub rtt_history: SlidingWindowAverageLegacy<i64>, //Estimated round-trip-time
    pub r_recv: i64, //Receiving rate
    pub rmode: RateUpdateMode, //Rate update mode: (0 = accelerated ramp-up | 1 = gradual update)
    pub x_curr: f64, //Aggregate congestion signal
    pub x_prev: f64, //Prev value of aggregate congestion signal
    pub r_vin: i64, //target rate for the live video encoder
    pub r_send: i64, //actual sending rate for regulating traffic
    pub prev_r_vin: i64,
    pub prev_r_send: i64,
    pub t_last: i64, //Last time receiving a feedback
    pub t_curr: i64,

    //Only for Debugging the NADA Receiver
    pub d_queue: i64, //Estimated queueing delay
    pub d_tilde: f64, //Equivalent delay after non-linear warping
    pub p_loss: f64,  //Estimated packet loss ratio

    pub t0: Instant, 
}
impl NadaSender {
    pub fn new(now: Instant) -> Self {
        Self {
            r_ref: NADA_INITIAL_RATE,
            rtt_history: SlidingWindowAverageLegacy::new(0, NADA_RTT_HISTORY_SIZE),
            r_recv: NADA_INITIAL_RATE, //30Mbps
            rmode: RateUpdateMode::GradualUpdate,
            x_curr: 0.0,
            x_prev: 0.0,
            r_vin: NADA_INITIAL_RATE,       //30Mbps
            r_send: NADA_INITIAL_RATE,      //30Mbps
            prev_r_vin: NADA_INITIAL_RATE,  //30Mbps
            prev_r_send: NADA_INITIAL_RATE, //30Mbps
            t_last: now.duration_since(Instant::now()).as_micros() as i64,
            t_curr: now.duration_since(Instant::now()).as_micros() as i64,
            //Only for Debugging the NADA Receiver
            d_queue: 0,
            d_tilde: 0.0,
            p_loss: 0.0,
            t0: now, 
        }
    }

    // Function to save data to CSV   || UNUSED
    // pub fn write_sender_values_to_csv(&self, filename: &str) -> Result<(), Box<dyn Error>> {
    //     let eval_rmode = match self.rmode {
    //         RateUpdateMode::AcceleratedRampUp => 0,
    //         RateUpdateMode::GradualUpdate => 1,
    //         _ => 2,
    //     };
    //     let linux_timestamp = Local::now().format("%Y%m%d%H%M%S").to_string();
    //     let rtt_average = self.rtt_history.get_average() as f64 / 1000.0; // /1000 to convert us to ms

    //     let nada_values = [
    //         self.r_ref.to_string(),
    //         rtt_average.to_string(),
    //         self.r_recv.to_string(),
    //         eval_rmode.to_string(),
    //         self.x_curr.to_string(),
    //         self.x_prev.to_string(),
    //         self.r_vin.to_string(),
    //         self.r_send.to_string(),
    //         self.prev_r_vin.to_string(),
    //         self.prev_r_send.to_string(),
    //         self.t_curr.to_string(),
    //         self.t_last.to_string(),
    //         //Only to debug Receiver values
    //         (self.d_queue as f64 / 1000.0).to_string(),
    //         self.d_tilde.to_string(),
    //         self.p_loss.to_string(),
    //         linux_timestamp,
    //     ];

    //     // let _= write_nada_variable_values_to_csv(filename, nada_values);
    //     let file = OpenOptions::new().write(true).append(true).open(filename)?;
    //     let mut writer = Writer::from_writer(file);
    //     writer.write_record(&nada_values)?;

    //     Ok(())
    // }

    fn update_accelerated_rampup(&mut self) -> i64 {
        let rtt_average_ms = self.rtt_history.get_average() / 1000.0; // /1000 to convert us to ms
        let res = (NADA_PARAM_QBOUND as f64
            / (rtt_average_ms as f64 + NADA_PARAM_DELTA as f64 + NADA_PARAM_DFILT as f64)); // changes for casting to f64 ( to allow decimals )
                                                                                            //gamma: Rate increase multiplier in accelerated ramp-up mode
        let gamma = if res < NADA_PARAM_GAMMA_MAX {
            res
        } else {
            NADA_PARAM_GAMMA_MAX
        };
        let updated_r_ref = if self.r_ref as f64 > (1.0 + gamma) * self.r_recv as f64 {
            self.r_ref
        } else {
            let result = (1.0 + gamma) * (self.r_recv as f64);
            result.round() as i64 // Round the result and convert to i64
        };

        updated_r_ref
    }

    fn update_gradual(&mut self, delta_us: i64) -> i64 {
        let delta = delta_us as f64 / 1000.0; // nano sec (us) to ms

        let right_side = NADA_PARAM_PRIO * NADA_PARAM_XREF as f64 * RMCAT_CC_DEFAULT_RMAX as f64
            / self.r_ref as f64;
        let x_offset = self.x_curr - right_side;
        let x_diff = self.x_curr - self.x_prev;
        let updated_r_ref = self.r_ref as f64
            - NADA_PARAM_KAPPA
                * (delta / NADA_PARAM_TAU as f64)
                * (x_offset / NADA_PARAM_TAU as f64)
                * self.r_ref as f64
            - NADA_PARAM_KAPPA
                * NADA_PARAM_ETA
                * (x_diff / NADA_PARAM_TAU as f64)
                * self.r_ref as f64;

        updated_r_ref.round() as i64
    }

    /** on receiving feedback report:
      1. obtain current timestamp from system clock: t_curr
      2. obtain values of rmode, x_curr, and r_recv from feedback report
      3. update estimation of rtt
      4. measure feedback interval: delta = t_curr - t_last
      if rmode == 0: update r_ref via accelerated ramp-up rules
      else:          update r_ref via gradual update rules
      6. clip rate r_ref within the range of [RMIN, RMAX]
      x_prev = x_curr & t_last = t_curr
    **/
    pub fn update_on_receive_feedback(
        &mut self,
        now: Instant,
        send_timestamp: i64,
        feedback_report: NADAFeedbackReport,
        video_fps: f64,
    ) {
        // println!("Received NADA feedback report: fps: {}, report: {:#?} ", video_fps, feedback_report );

        // self.t_curr = Utc::now().timestamp_micros();
        self.t_curr = now.duration_since(self.t0).as_micros() as i64;

        self.rmode = match feedback_report.rmode {
            0 => RateUpdateMode::AcceleratedRampUp,
            1 => RateUpdateMode::GradualUpdate,
            _ => RateUpdateMode::GradualUpdate,
        };
        self.x_curr = feedback_report.x_curr;
        self.r_recv = feedback_report.r_recv;

        //Only To Debug NADA Receiver
        self.d_queue = feedback_report.d_queue;
        self.d_tilde = feedback_report.d_tilde;

        //update estimation of rtt
        let rtt = self.t_curr - send_timestamp;
        self.rtt_history.submit_sample(rtt);

        //Measure feedback interval: delta = t_curr - t_last
        let delta_us = self.t_curr - self.t_last;
        let updated_r_ref;
        match self.rmode {
            RateUpdateMode::AcceleratedRampUp => {
                updated_r_ref = self.update_accelerated_rampup();
            }
            _ => {
                updated_r_ref = self.update_gradual(delta_us);
            }
        };

        self.r_ref = updated_r_ref.clamp(RMCAT_CC_DEFAULT_RMIN, RMCAT_CC_DEFAULT_RMAX);
        self.t_last = self.t_curr;
        self.x_prev = self.x_curr;

        //Update target (r_vin) & sending (r_send) bitrates
        self.prev_r_vin = self.r_vin;
        self.prev_r_send = self.r_send;

        let use_shaping_buffer = false;
        if use_shaping_buffer {
            self.update_target_bitrate(0.0, video_fps);
            self.update_sending_bitrate(0.0, video_fps);
        } else {
            self.r_vin = self.r_ref;
            self.r_send = self.r_ref;
        }
    }

    fn update_target_bitrate(&mut self, buffer_len: f64, video_fps: f64) {
        self.prev_r_vin = self.r_vin;
        let buffer_len_ = if buffer_len == 0.0 {
            RMCAT_CC_DEFAULT_RMAX as f64 / 8.0 / 1500.0
            // 1.0
        } else {
            buffer_len
        };

        // r_diff_v = min(0.05 * r_ref, BETA_V * 8 * buffer_len * FPS)
        let r_diff_v = f64::min(
            self.r_ref as f64 * 0.05,
            NADA_PARAM_BETA_V * 8.0 * buffer_len_ * video_fps,
        );
        self.r_vin = i64::max(
            RMCAT_CC_DEFAULT_RMIN,
            (self.r_ref as f64 - r_diff_v).round() as i64,
        );
    }

    fn update_sending_bitrate(&mut self, buffer_len: f64, video_fps: f64) {
        let buffer_len_ = if buffer_len == 0.0 {
            RMCAT_CC_DEFAULT_RMAX as f64 / 8.0 / 1500.0
            // 1.0
        } else {
            buffer_len
        };

        // r_diff_s = min(0.05 * r_ref, BETA_S * 8 * buffer_len * FPS)
        let r_diff_s = f64::min(
            self.r_ref as f64 * 0.05,
            NADA_PARAM_BETA_S * 8.0 * buffer_len_ * video_fps,
        );
        self.r_send = i64::min(
            RMCAT_CC_DEFAULT_RMAX,
            (self.r_ref as f64 + r_diff_s).round() as i64,
        );
    }

    pub fn get_target_bitrate(&mut self) -> i64 {
        self.r_vin
    }

    pub fn get_sending_bitrate(&mut self) -> i64 {
        self.r_send
    }
}

//////////// RECEIVER SIDE NADA /////////////

pub struct NadaReceiver {
    pub d_base: i64,  //Estimated baseline delay
    pub d_tilde: f64, //Equivalent delay after non-linear warping
    pub d_queue: i64, //Estimated queueing delay
    pub p_loss: f64,  //Estimated packet loss ratio
    pub p_mark: f64,  //Estimated packet ECN marking ratio
    pub r_recv: i64,  //Receiving rate  (bps)
    pub t_last: i64,  //Last time receiving a feedback
    pub d_queue_history: SlidingWindowAverageLegacy<i64>,
    pub x_curr: f64,           //Aggregate congestion signal
    pub rmode: RateUpdateMode, //Rate update mode: (0 = accelerated ramp-up | 1 = gradual)

    //last send & arrival times
    pub last_t_send: i64,    //last send timestamp
    pub last_t_arrival: i64, //last arrival timestamp

    //To compute r_recv, p_loss
    pub total_received_bytes: usize,
    total_num_packets: u32,
    total_packets_lost: u32,
    pub receive_rate_timer: Instant,

    pub t0: Instant, 
}

impl NadaReceiver {
    pub fn new( now: Instant) -> Self {
        Self {
            d_base: i64::MAX,
            d_tilde: 0.0,
            d_queue: 0,
            p_loss: 0.0,
            p_mark: 0.0,
            r_recv: 0,
            t_last: 0,
            d_queue_history: SlidingWindowAverageLegacy::new(0, 15),
            x_curr: 0.0,
            rmode: RateUpdateMode::GradualUpdate,

            //last send & arrival times
            last_t_send: 0,
            last_t_arrival: 0,

            //receive timer
            total_received_bytes: 0,
            total_num_packets: 0,
            total_packets_lost: 0,
            receive_rate_timer: Instant::now(),

            t0: now, 
        }
    }

    /****
       Obtain one-way delay measurement: d_fwd = t_curr - t_sent
       update baseline delay: d_base = min(d_base, d_fwd)
       update queuing delay:  d_queue = d_fwd - d_base
    *****/
    pub fn compute_oneway_delay(
        &mut self,
        frame_send_timestamp_MICROS: i64,
        frame_arrival_timestamp_MICROS: i64,
    ) {
        let d_fwd = frame_arrival_timestamp_MICROS - frame_send_timestamp_MICROS;
        self.d_base = std::cmp::min(self.d_base, d_fwd);
        self.d_queue = d_fwd - self.d_base;
        self.d_queue_history.submit_sample(self.d_queue);

        //initialize last timestamps
        self.last_t_send = frame_send_timestamp_MICROS;
        self.last_t_arrival = frame_arrival_timestamp_MICROS;
    }

    /** When packet losses are observed, the estimated queuing delay follows
    a non-linear warping inspired by the delay-adaptive congestion window
    backoff policy in [Budzisz-TON11]: **/
    fn compute_d_tilde(&mut self, had_packet_loss: bool) {
        if had_packet_loss {
            if self.d_queue < NADA_PARAM_QTH {
                self.d_tilde = self.d_queue as f64;
            } else if self.d_queue > NADA_PARAM_QTH && self.d_queue < NADA_PARAM_QMAX {
                let numerator = (NADA_PARAM_QTH - self.d_queue).pow(4);
                let denominator = (NADA_PARAM_QMAX - NADA_PARAM_QTH).pow(4);
                self.d_tilde = NADA_PARAM_QTH as f64 * (numerator / denominator) as f64;
            } else {
                self.d_tilde = 0.0;
            }
        }
    }

    /** On time to send a new feedback report (t_curr - t_last > DELTA)
    calculate non-linear warping of delay d_tilde if packet loss exists
    calculate current aggregate congestion signal x_curr
    determine mode of rate adaptation for sender: rmode
    send RTCP feedback report containing: rmode, x_curr, and r_recv
    update t_last = t_curr  **/

    pub fn time_to_report_feedback(
        &mut self,
        now: Instant,
        had_packet_loss: bool,
        had_packet_mark: bool,
    ) -> bool {
        // let t_curr = Utc::now().timestamp_micros();

        let t_curr = now.duration_since(self.t0).as_micros() as i64;
        // print!("[NADA]time to report?");
        let time_diff_ms = (t_curr - self.t_last) / 1000;

        // println!("time diff in ms : now (micros): {} , last: {} -> DIFF: {}", t_curr, self.t_last, time_diff_ms) ;
        if time_diff_ms > NADA_PARAM_DELTA {
            // println!("Time diff > {}", NADA_PARAM_DELTA);

            self.d_tilde = self.d_queue_history.get_average() as f64 / 1000.0; // /1000 for us to ms
                                                                               // Calculate non-linear warping of delay if packet loss exists
            if had_packet_loss {
                self.compute_d_tilde(had_packet_loss);
            }
            // Update packet marking ratio estimate
            if had_packet_mark {
                self.p_mark += 1.0; // Increment mark count (similarly, improve this logic as needed)
            }
            self.x_curr = self.d_tilde
                + self.p_mark * NADA_PARAM_DMARK as f64
                + self.p_loss * NADA_PARAM_DLOSS as f64;
            self.determine_rate_adaptation_mode(had_packet_loss);

            true
        } else {
            false
        }
    }

    pub fn update_t_last(&mut self, now: Instant) {
        // change to TaiTime
        let t_curr = now.duration_since(self.t0).as_micros() as i64;
        self.t_last = t_curr;
    }

    /** Is there build-up of queuing delay?
    Check if d_fwd-d_base < QEPS for all previous
     delay samples within the observation window LOGWIN **/
    fn exists_queuing_delay_buildup(&mut self) -> bool {
        for value in self.d_queue_history.get_history_iter() {
            if *value > NADA_PARAM_QEPS_US {
                return true;
            }
        }
        false
    }

    /**
     Determine whether the network is underutilized
     and recommend the corresponding rate adaptation mode
    **/
    fn determine_rate_adaptation_mode(&mut self, had_packet_loss: bool) {
        self.rmode = RateUpdateMode::AcceleratedRampUp;
        /* To operate in accelerated ramp-up mode:
        o  No recent packet losses within the observation window LOGWIN; and
        o  No build-up of queuing delay: d_fwd-d_base < QEPS for all previous
        delay samples within the observation window LOGWIN.*/
        if had_packet_loss {
            self.rmode = RateUpdateMode::GradualUpdate;
        }
        if self.exists_queuing_delay_buildup() {
            self.rmode = RateUpdateMode::GradualUpdate;
        }
    }

    pub fn update_receive_loss_rate(&mut self, received_bytes: usize) {
        let total_num_packets = (received_bytes as f32 / 1500.0).ceil() as u32;
        let num_packets_lost = 0; // ......OK?
                                  // received bytes in nal + size of header (VideoPacketHeader)
        let size_of_timestamp = std::mem::size_of::<Duration>();
        let size_of_send_timestamp = std::mem::size_of::<i64>();
        self.total_received_bytes += received_bytes + size_of_timestamp + size_of_send_timestamp;

        // total & lost packets to compute p_inst
        self.total_packets_lost += num_packets_lost; // Comment when replicating: THIS IS ALWAYS ZERO? So packets lost is useless as metric in this implementation??.
        self.total_num_packets += total_num_packets;

        let elapsed = self.receive_rate_timer.elapsed();
        if elapsed >= Duration::from_millis(NADA_PARAM_LOGWIN as u64) {
            /* 5.1.3.  Estimation of receiving rate (r_recv):
             * NADA maintains a recent observation window with time span of LOGWIN,
             * and simply divides the total size of packets arriving during that window */
            let interval_in_sec = NADA_PARAM_LOGWIN as f64 / 1000.0;
            let received_rate_bps = ((self.total_received_bytes * 8) as f64) / interval_in_sec;
            self.r_recv = received_rate_bps.round() as i64;

            /* 5.1.2.  Estimation of packet loss/marking ratio:
            * The instantaneous packet loss ratio p_inst is the ratio between
            * the number of missing packets over the number of total transmitted packets
            * within the recent observation window LOGWIN.  The packet loss ratio p_loss is
            *obtained after exponential smoothing:
               p_loss = ALPHA*p_inst + (1-ALPHA)*p_loss.   (10) */
            let p_inst = if self.total_num_packets > 0 {
                self.total_packets_lost as f64 / self.total_num_packets as f64
            } else {
                0.0
            };

            self.p_loss = NADA_PARAM_ALPHA * p_inst + (1.0 - NADA_PARAM_ALPHA) * self.p_loss;

            //re-initialize
            self.total_received_bytes = 0;
            self.receive_rate_timer = Instant::now();
            self.total_packets_lost = 0;
            self.total_num_packets = 0;
        }
    }
}
