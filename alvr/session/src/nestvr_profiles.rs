use crate::settings::NestVrProfile;

#[derive(Debug, Clone, Copy)]
pub struct ProfileConfig {
    pub update_interval_nestvr_s: f32,

    pub max_bitrate_mbps: f32,
    pub min_bitrate_mbps: f32,
    pub initial_bitrate_mbps: f32,

    pub bitrate_step_count: usize,
    pub bitrate_inc_steps: usize,
    pub bitrate_dec_steps: usize,

    pub rtt_adj_prob: f32,
    pub bitrate_inc_prob: f32,

    pub nfr_thresh: f32,
    pub rtt_thresh_ms: f32,

    pub capacity_scaling_factor: f32,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        ProfileConfig {
            update_interval_nestvr_s: 1.,

            max_bitrate_mbps: 100.,
            min_bitrate_mbps: 10.,
            initial_bitrate_mbps: 50.,

            bitrate_step_count: 9,
            bitrate_inc_steps: 1,
            bitrate_dec_steps: 1,

            rtt_adj_prob: 1.0,
            bitrate_inc_prob: 0.25,

            nfr_thresh: 0.99,
            rtt_thresh_ms: 22.,

            capacity_scaling_factor: 0.9,
        }
    }
}

pub fn get_profile_config(
    max_bitrate_mbps: f32,
    min_bitrate_mbps: f32,
    initial_bitrate_mbps: f32,
    nest_vr_profile: &NestVrProfile,
) -> ProfileConfig {
    let base_config = ProfileConfig {
        max_bitrate_mbps,
        min_bitrate_mbps,
        initial_bitrate_mbps,
        ..Default::default()
    };

    match nest_vr_profile {
        NestVrProfile::Custom {
            update_interval_nestvr_s,
            bitrate_step_count,
            bitrate_inc_steps,
            bitrate_dec_steps,
            rtt_adj_prob,
            bitrate_inc_prob,
            nfr_thresh,
            rtt_thresh_ms,
            capacity_scaling_factor,
        } => ProfileConfig {
            update_interval_nestvr_s: *update_interval_nestvr_s,
            bitrate_step_count: *bitrate_step_count,
            bitrate_inc_steps: *bitrate_inc_steps,
            bitrate_dec_steps: *bitrate_dec_steps,
            rtt_adj_prob: *rtt_adj_prob,
            bitrate_inc_prob: *bitrate_inc_prob,
            nfr_thresh: *nfr_thresh,
            rtt_thresh_ms: *rtt_thresh_ms,
            capacity_scaling_factor: *capacity_scaling_factor,
            ..base_config
        },
        NestVrProfile::Balanced => ProfileConfig {
            bitrate_dec_steps: 1,
            ..base_config
        },
        NestVrProfile::Speedy => ProfileConfig {
            bitrate_dec_steps: 2,
            ..base_config
        },
        NestVrProfile::Anxious => ProfileConfig {
            bitrate_dec_steps: 10,
            ..base_config
        },
    }
}
