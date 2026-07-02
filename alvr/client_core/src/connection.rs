#![allow(clippy::if_same_then_else)]

use crate::{
    decoder::{self, DECODER_INIT_CONFIG},
    logging_backend::{LogMirrorData, LOG_CHANNEL_SENDER},
    platform,
    sockets::AnnouncerSocket,
    statistics::StatisticsManager,
    storage::Config,
    ClientCoreEvent, EVENT_QUEUE, LIFECYCLE_STATE, STATISTICS_MANAGER,
};
use alvr_audio::AudioDevice;
use alvr_common::{
    debug, error,
    glam::UVec2,
    info,
    once_cell::sync::Lazy,
    parking_lot::{Condvar, RwLock},
    wait_rwlock, warn, AnyhowToCon, ConResult, ConnectionError, ConnectionState, LifecycleState,
    OptLazy, ToCon, ALVR_VERSION, NadaReceiver, RateUpdateMode, 

};
use alvr_packets::{
    ClientConnectionResult, ClientControlPacket, ClientStatistics, Haptics,
    NetworkStatisticsPacket,  ServerControlPacket, StreamConfigPacket, Tracking, VideoPacketHeader,
    VideoStreamingCapabilities, AUDIO, HAPTICS, STATISTICS, TRACKING, VIDEO,
    EverestCommand, NadaStats,  
};
use alvr_session::{settings_schema::Switch, SessionConfig, BitrateMode, };
use alvr_sockets::{
    ControlSocketSender, PeerType, ProtoControlSocket, StreamSender, StreamSocketBuilder,
    KEEPALIVE_INTERVAL, KEEPALIVE_TIMEOUT,
};
use serde_json as json;
use std::{
    collections::HashMap,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};



#[cfg(target_os = "android")]
use crate::audio;
#[cfg(not(target_os = "android"))]
use alvr_audio as audio;

const INITIAL_MESSAGE: &str = concat!(
    "Searching for streamer...\n",
    "Open ALVR on your PC then click \"Trust\"\n",
    "next to the client entry",
);
const NETWORK_UNREACHABLE_MESSAGE: &str = "Cannot connect to the internet";
// const INCOMPATIBLE_VERSIONS_MESSAGE: &str = concat!(
//     "Streamer and client have\n",
//     "incompatible types.\n",
//     "Please update either the app\n",
//     "on the PC or on the headset",
// );
const STREAM_STARTING_MESSAGE: &str = "The stream will begin soon\nPlease wait...";
const SERVER_RESTART_MESSAGE: &str = "The streamer is restarting\nPlease wait...";
const SERVER_DISCONNECTED_MESSAGE: &str = "The streamer has disconnected.";
const CONNECTION_TIMEOUT_MESSAGE: &str = "Connection timeout.";

const DISCOVERY_RETRY_PAUSE: Duration = Duration::from_millis(500);
const RETRY_CONNECT_MIN_INTERVAL: Duration = Duration::from_secs(1);
const CONNECTION_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const HANDSHAKE_ACTION_TIMEOUT: Duration = Duration::from_secs(2);
const STREAMING_RECV_TIMEOUT: Duration = Duration::from_millis(500);

const MAX_UNREAD_PACKETS: usize = 10; // Applies per stream

pub static CONNECTION_STATE: Lazy<Arc<RwLock<ConnectionState>>> =
    Lazy::new(|| Arc::new(RwLock::new(ConnectionState::Disconnected)));
pub static DISCONNECTED_NOTIF: Condvar = Condvar::new();

pub static CONTROL_SENDER: OptLazy<ControlSocketSender<ClientControlPacket>> =
    alvr_common::lazy_mut_none();
pub static TRACKING_SENDER: OptLazy<StreamSender<Tracking>> = alvr_common::lazy_mut_none();
pub static STATISTICS_SENDER: OptLazy<StreamSender<ClientStatistics>> =
    alvr_common::lazy_mut_none();

fn set_hud_message(message: &str) {
    let message = format!(
        "ALVR v{}\nhostname: {}\nIP: {}\n\n{message}",
        *ALVR_VERSION,
        Config::load().hostname,
        platform::local_ip(),
    );

    EVENT_QUEUE
        .lock()
        .push_back(ClientCoreEvent::UpdateHudMessage(message));
}

fn is_streaming() -> bool {
    *CONNECTION_STATE.read() == ConnectionState::Streaming
}

pub fn connection_lifecycle_loop(
    recommended_view_resolution: UVec2,
    supported_refresh_rates: Vec<f32>,
) {
    set_hud_message(INITIAL_MESSAGE);

    while *LIFECYCLE_STATE.read() != LifecycleState::ShuttingDown {
        if *LIFECYCLE_STATE.read() == LifecycleState::Resumed {
            if let Err(e) =
                connection_pipeline(recommended_view_resolution, supported_refresh_rates.clone())
            {
                let message = format!("Connection error:\n{e}\nCheck the PC for more details");
                set_hud_message(&message);
                error!("Connection error: {e}");
            }
        } else {
            debug!("Skip try connection because the device is sleeping");
        }

        *CONNECTION_STATE.write() = ConnectionState::Disconnected;
        DISCONNECTED_NOTIF.notify_all();

        thread::sleep(CONNECTION_RETRY_INTERVAL);
    }
}

fn connection_pipeline(
    recommended_view_resolution: UVec2,
    supported_refresh_rates: Vec<f32>,
) -> ConResult {
    let (mut proto_control_socket, server_ip) = {
        let config = Config::load();
        let announcer_socket = AnnouncerSocket::new(&config.hostname).to_con()?;
        let listener_socket =
            alvr_sockets::get_server_listener(HANDSHAKE_ACTION_TIMEOUT).to_con()?;

        loop {
            if *LIFECYCLE_STATE.write() != LifecycleState::Resumed {
                return Ok(());
            }

            if let Err(e) = announcer_socket.broadcast() {
                warn!("Broadcast error: {e:?}");

                set_hud_message(NETWORK_UNREACHABLE_MESSAGE);

                thread::sleep(RETRY_CONNECT_MIN_INTERVAL);

                set_hud_message(INITIAL_MESSAGE);

                return Ok(());
            }

            if let Ok(pair) = ProtoControlSocket::connect_to(
                DISCOVERY_RETRY_PAUSE,
                PeerType::Server(&listener_socket),
            ) {
                break pair;
            }
        }
    };

    let mut connection_state_lock = CONNECTION_STATE.write();
    let disconnect_notif = Arc::new(Condvar::new());

    *connection_state_lock = ConnectionState::Connecting;

    let microphone_sample_rate = AudioDevice::new_input(None)
        .unwrap()
        .input_sample_rate()
        .unwrap();

    proto_control_socket
        .send(&ClientConnectionResult::ConnectionAccepted {
            client_protocol_id: alvr_common::protocol_id(),
            display_name: platform::device_model(),
            server_ip,
            streaming_capabilities: Some(VideoStreamingCapabilities {
                default_view_resolution: recommended_view_resolution,
                supported_refresh_rates,
                microphone_sample_rate,
            }),
        })
        .to_con()?;
    let config_packet =
        proto_control_socket.recv::<StreamConfigPacket>(HANDSHAKE_ACTION_TIMEOUT)?;

    let settings = {
        let mut session_desc = SessionConfig::default();
        session_desc
            .merge_from_json(&json::from_str(&config_packet.session).to_con()?)
            .to_con()?;
        session_desc.to_settings()
    };

    let negotiated_config =
        json::from_str::<HashMap<String, json::Value>>(&config_packet.negotiated).to_con()?;

    let view_resolution = negotiated_config
        .get("view_resolution")
        .and_then(|v| json::from_value(v.clone()).ok())
        .unwrap_or(UVec2::ZERO);
    let refresh_rate_hint = negotiated_config
        .get("refresh_rate_hint")
        .and_then(|v| v.as_f64())
        .unwrap_or(60.0) as f32;
    let game_audio_sample_rate = negotiated_config
        .get("game_audio_sample_rate")
        .and_then(|v| v.as_u64())
        .unwrap_or(44100) as u32;

    let streaming_start_event = ClientCoreEvent::StreamingStarted {
        view_resolution,
        refresh_rate_hint,
        settings: Box::new(settings.clone()),
    };

    *STATISTICS_MANAGER.lock() = Some(StatisticsManager::new(
        settings.connection.statistics_history_size,
        Duration::from_secs_f32(1.0 / refresh_rate_hint),
        if let Switch::Enabled(config) = settings.headset.controllers {
            config.steamvr_pipeline_frames
        } else {
            0.0
        },
    ));

    let (mut control_sender, mut control_receiver) = proto_control_socket
        .split(STREAMING_RECV_TIMEOUT)
        .to_con()?;

    match control_receiver.recv(HANDSHAKE_ACTION_TIMEOUT) {
        Ok(ServerControlPacket::StartStream) => {
            info!("Stream starting");
            set_hud_message(STREAM_STARTING_MESSAGE);
        }
        Ok(ServerControlPacket::Restarting) => {
            info!("Server restarting");
            set_hud_message(SERVER_RESTART_MESSAGE);
            return Ok(());
        }
        Err(e) => {
            info!("Server disconnected. Cause: {e}");
            set_hud_message(SERVER_DISCONNECTED_MESSAGE);
            return Ok(());
        }
        _ => {
            info!("Unexpected packet");
            set_hud_message("Unexpected packet");
            return Ok(());
        }
    }

    let stream_socket_builder = StreamSocketBuilder::listen_for_server(
        Duration::from_secs(1),
        settings.connection.stream_port,
        settings.connection.stream_protocol,
        settings.connection.dscp,
        settings.connection.client_send_buffer_bytes,
        settings.connection.client_recv_buffer_bytes,
    )
    .to_con()?;

    if let Err(e) = control_sender.send(&ClientControlPacket::StreamReady) {
        info!("Server disconnected. Cause: {e:?}");
        set_hud_message(SERVER_DISCONNECTED_MESSAGE);
        return Ok(());
    }

    let mut stream_socket = stream_socket_builder.accept_from_server(
        server_ip,
        settings.connection.stream_port,
        settings.connection.packet_size as _,
        HANDSHAKE_ACTION_TIMEOUT,
    )?;

    info!("Connected to server");
    {
        let config = &mut *DECODER_INIT_CONFIG.lock();

        config.max_buffering_frames = settings.video.max_buffering_frames;
        config.buffering_history_weight = settings.video.buffering_history_weight;
        config.options = settings.video.mediacodec_extra_options;
    }

    let mut video_receiver =
        stream_socket.subscribe_to_stream::<VideoPacketHeader>(VIDEO, MAX_UNREAD_PACKETS);
    let mut game_audio_receiver = stream_socket.subscribe_to_stream(AUDIO, MAX_UNREAD_PACKETS);
    let tracking_sender = stream_socket.request_stream(TRACKING);
    let mut haptics_receiver =
        stream_socket.subscribe_to_stream::<Haptics>(HAPTICS, MAX_UNREAD_PACKETS);
    let statistics_sender = stream_socket.request_stream(STATISTICS);

    let mut last_instant_IDR_client = Instant::now();
    let interval_IDR_seconds_f32 =
        settings.connection.client_idr_refresh_interval_ms as f32 / 1000.0;

    let mut frames_dropped: u32 = 0; // number of frames dropped


    let is_everest_enabled = matches!(settings.video.bitrate.mode, BitrateMode::EverestPort{..}) ; 
    let is_nada_enabled = matches!(settings.video.bitrate.mode, BitrateMode::NadaPort{..}) ; 

    pub struct EverestObject {
        frame_size_exp_avg: f32,
        d_short_exp_avg: f32,
        d_long_exp_avg: f32,
    }
    let mut everest_receiver_object = EverestObject{ // Assuming these are kept in scope all the time during the loop
        frame_size_exp_avg: 0.0,
        d_short_exp_avg: 0.0, 
        d_long_exp_avg: 0.0,
    }; 

    let mut nada_receiver_object = if is_nada_enabled {
        Some(NadaReceiver::new(Instant::now()) ) // Assuming these are kept in scope all the time during the loop
    }   
    else{
        None
    }; 
    let video_receive_thread = thread::spawn(move || {
        let mut stream_corrupted = false;
        while is_streaming() {
            let data = match video_receiver.recv(STREAMING_RECV_TIMEOUT) {
                Ok(data) => data,
                Err(ConnectionError::TryAgain(_)) => continue,
                Err(ConnectionError::Other(_)) => return,
            };


            let mut everest_throughput: f32 = -1.0; // initialize, if negative then on rx don't count
            let mut everest_capacity: f32 = -1.0; // (only one measure per frame of either)

            let mut command_abr_everest = EverestCommand::Continue;

            if is_everest_enabled {
                if everest_receiver_object.frame_size_exp_avg == 0.0 {
                    everest_receiver_object.frame_size_exp_avg = data.get_bytes_in_frame() as f32;
                    // initialize avg only on first value
                }
                if everest_receiver_object.d_short_exp_avg == 0.0 {
                    everest_receiver_object.d_short_exp_avg = data.get_frame_span()
                        * data.get_frame_interarrival()
                        / T_SHORT_EVEREST_S;
                }
                if everest_receiver_object.d_long_exp_avg == 0.0 {
                    everest_receiver_object.d_long_exp_avg = data.get_frame_span()
                        * data.get_frame_interarrival()
                        / T_LONG_EVEREST_S;
                }
                pub const MPDU_MAX_SIZE: u32 = 1500;
                pub const THETA_EWMA: f32 = 0.01; // we want the long term expectation for comparison of individual frame sizes.

                pub const T_SHORT_EVEREST_S: f32 = 1.0;
                pub const T_LONG_EVEREST_S: f32 = 5.0;

                let frame_size_bytes = data.get_bytes_in_frame() as f32;
                let frame_span = data.get_frame_span();

                if frame_span != 0.0 {
                    // prevent division by zero
                    
                    // EVEREST-Intra
                    everest_receiver_object.frame_size_exp_avg = (THETA_EWMA * frame_size_bytes)
                        + (1.0 - THETA_EWMA) * everest_receiver_object.frame_size_exp_avg;

                    if frame_size_bytes > everest_receiver_object.frame_size_exp_avg {
                        everest_throughput = frame_size_bytes * 8.0 / frame_span;
                    } else {
                        let frame_size_mtu_portion =
                            (frame_size_bytes as u32 / MPDU_MAX_SIZE) as f32
                                * MPDU_MAX_SIZE as f32; // just the part with full packets of MTU
                        everest_capacity = (frame_size_mtu_portion * 8.0) / frame_span;

                        // print_yellow!("Capacity ev: L / deltaT = {} ({}) / {} = {}", frame_size_mtu_portion, frame_size_bytes, frame_span, everest_capacity);
                        }
                }

                let interarrival = data.get_frame_interarrival();
                everest_receiver_object.d_short_exp_avg = (interarrival / T_SHORT_EVEREST_S * frame_span)
                    + (1.0 - interarrival / T_SHORT_EVEREST_S) * everest_receiver_object.d_short_exp_avg;
                everest_receiver_object.d_long_exp_avg = (interarrival / T_LONG_EVEREST_S * frame_span)
                    + (1.0 - interarrival / T_LONG_EVEREST_S) * everest_receiver_object.d_long_exp_avg;

                //  (B2 = 2 * B1). This code assumes that every next bitrate will double the previous when going up the ladder, given the lack of the bitrate ladder info
                //  at the client. 
                //  Then: B1/B2 is always 0.5, so d_lower reduces to a constant fraction of the frame period — no live bitrate
                // or ladder knowledge needed on the client at all.
                let d_period = 1.0 / refresh_rate_hint;

                const BITRATE_STEP_RATIO: f32 = 0.5; // B1/B2 for a doubling ladder
                let d_lower_everest = BITRATE_STEP_RATIO * d_period;
                let d_upper_everest = d_period;

                const T_LOW_EVEREST_S: f32 = 0.005;
                const T_HIGH_EVEREST_S: f32 = 0.020;

                if everest_receiver_object.d_short_exp_avg >= d_upper_everest {
                    everest_receiver_object.d_short_exp_avg = T_LOW_EVEREST_S;
                    command_abr_everest = EverestCommand::SlowDown;
                }
                if everest_receiver_object.d_long_exp_avg < d_lower_everest {
                    everest_receiver_object.d_long_exp_avg = T_HIGH_EVEREST_S;
                    command_abr_everest = EverestCommand::SpeedUp;
                }

            }

            //////////////////////////////////////////////  // NADA rcv loop upon succesfully receiving a full frame.
            let mut nada_stats: NadaStats = NadaStats::default();

            if let Some(nada_receiver) = nada_receiver_object.as_mut() {
                // println!("Nada receiver exists");
                // let mut nada_receiver = nada_receiver_in.lock().unwrap();
                let now = Instant::now(); 

                let frame_send_timestamp = data.get_tx_time_first(); // as secs;
                let frame_recv_timestamp = data.get_rx_time_last(); //as secs;
                let size = data.get_bytes_in_frame() as usize;

                let micros_send_ts = (frame_send_timestamp * 1_000_000.0).round() as i64;
                let micros_rcv_ts = (frame_recv_timestamp * 1_000_000.0).round() as i64;

                nada_receiver.compute_oneway_delay(micros_send_ts, micros_rcv_ts); //inputs as micros
                nada_receiver.update_receive_loss_rate(size);
                let is_feedback_on =
                    nada_receiver.time_to_report_feedback(now, false, false);

                //if there is a feedback to report
                if is_feedback_on {
                    // println!("FEEDBACK IS ON");
                    //send RTCP feedback report containing values of: rmode, x_curr, and r_recv
                    nada_stats.nada_feedback = true;
                    nada_stats.nada_xcurr = nada_receiver.x_curr;
                    nada_stats.nada_rmode = match nada_receiver.rmode {
                        RateUpdateMode::AcceleratedRampUp => 0,
                        RateUpdateMode::GradualUpdate => 1,
                        _ => 1,
                    };
                    nada_stats.nada_recv = nada_receiver.r_recv;

                    //To Debug NADA Receiver, report values of: t_last, d_fwd, d_tilde, d_queue, p_loss
                    nada_stats.plr = nada_receiver.p_loss;
                    nada_stats.d_tilde = nada_receiver.d_tilde;
                    nada_stats.d_queue = nada_receiver.d_queue;

                    //update t_last = t_curr
                    nada_receiver.update_t_last(now);
                } else {
                    nada_stats.nada_feedback = false;
                }
            }

            // send frame and network statistics for every reconstructed video frame
            if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                sender
                    .send(&ClientControlPacket::NetworkStatistics(
                        NetworkStatisticsPacket {
                            // Frame specific metrics
                            frame_index: data.get_frame_index() as i32, // index of the current frame
                            frame_span: data.get_frame_span(), // duration of the current frame

                            bytes_in_frame: data.get_bytes_in_frame(), // bytes received for the current frame, including both prefixes and network headers
                            bytes_in_frame_app: data.get_bytes_in_frame_app(), // bytes received for the current frame, excluding both prefixes and network headers

                            // Interval specific metrics
                            frame_interarrival: data.get_frame_interarrival(), // time interval between consecutive frames

                            interarrival_jitter: data.get_interarrival_jitter(), // measure of the variability in the time between the reception of consecutive video shards
                            ow_delay: data.get_ow_delay(), // one-way delay of the received video shards
                            filtered_ow_delay: data.get_filtered_ow_delay(), // kalman filtered one-way delay of the received video shards, as GCC does

                            frames_skipped: data.get_frames_skipped(), // number of frames skipped

                            rx_bytes: data.get_rx_bytes(), // bytes received in the interval between the consecutive frames, including any prefixes and network headers

                            rx_shard_counter: data.get_rx_shard_counter(), // non-duplicated video shards received during the interval between consecutive frames
                            duplicated_shard_counter: data.get_duplicated_shard_counter(), // duplicated video shards received during the interval between consecutive frames

                            highest_rx_frame_index: data.get_highest_rx_frame_index(), // index of the highest video frame received during the interval between consecutive frames
                            highest_rx_shard_index: data.get_highest_rx_shard_index(), // index of the highest video shard received during the interval between consecutive frames
                        

                            everest_capacity_update: everest_capacity,
                            everest_throughput_update: everest_throughput,
                            everest_dshort: everest_receiver_object.d_short_exp_avg,
                            everest_dlong: everest_receiver_object.d_long_exp_avg,
                            everest_command: command_abr_everest,
                            // edca_ac: EdcaAc::Video, // Explanation: Given we're computing the VF-RTT of video packets based on arrivals, let's assume this AC for UL to get the same 'treatment' by EDCA.
                            nada_stats,                      
                        
                        },
                    ))
                    .ok();
            }

            let Ok((header, nal)) = data.get() else {
                return;
            };
            if let Some(stats) = &mut *STATISTICS_MANAGER.lock() {
                stats.report_video_packet_received(header.timestamp);
            }

            // periodically request an IDR frame using the settings' client_idr_refresh_interval_ms

            if settings.connection.idr_periodic_bool {
                if Instant::now()
                    .saturating_duration_since(last_instant_IDR_client)
                    .as_secs_f32()
                    >= interval_IDR_seconds_f32
                {
                    if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                        sender.send(&ClientControlPacket::RequestIdr).ok();
                    }
                    last_instant_IDR_client = Instant::now();   
                }
            }

            if header.is_idr {
                stream_corrupted = false;
            } else if data.had_packet_loss() {
                stream_corrupted = true;
                if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                    sender.send(&ClientControlPacket::RequestIdr).ok();
                }
                warn!(
                    "Network skipped {} video packets",
                    data.get_frames_skipped()
                );
            }
            if !stream_corrupted || !settings.connection.avoid_video_glitching {
                if !decoder::push_nal(header.timestamp, nal) {
                    stream_corrupted = true;
                    if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                        sender.send(&ClientControlPacket::RequestIdr).ok();
                    }
                    if let Some(stats) = &mut *STATISTICS_MANAGER.lock() {
                        stats.report_video_packet_dropped(data.get_frame_index());
                    }
                    warn!(
                        "Dropped video packet {}. Reason: Decoder saturation",
                        data.get_frame_index()
                    );
                    frames_dropped += 1;
                } else {
                    // frame is decoded correctly
                    if let Some(stats) = &mut *STATISTICS_MANAGER.lock() {
                        stats.report_video_packet_data(
                            header.timestamp,
                            data.get_frame_index(),
                            frames_dropped,
                        );
                    }
                    frames_dropped = 0;
                }
            } else {
                if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                    sender.send(&ClientControlPacket::RequestIdr).ok();
                }
                if let Some(stats) = &mut *STATISTICS_MANAGER.lock() {
                    stats.report_video_packet_dropped(data.get_frame_index());
                }
                warn!(
                    "Dropped video packet {}. Reason: Waiting for IDR frame",
                    data.get_frame_index()
                );
                frames_dropped += 1;
            }
        }
    });

    let game_audio_thread = if let Switch::Enabled(config) = settings.audio.game_audio {
        let device = AudioDevice::new_output(None, None).to_con()?;

        thread::spawn(move || {
            while is_streaming() {
                alvr_common::show_err(audio::play_audio_loop(
                    is_streaming,
                    &device,
                    2,
                    game_audio_sample_rate,
                    config.buffering.clone(),
                    &mut game_audio_receiver,
                ));
            }
        })
    } else {
        thread::spawn(|| ())
    };

    let microphone_thread = if matches!(settings.audio.microphone, Switch::Enabled(_)) {
        let device = AudioDevice::new_input(None).to_con()?;

        let microphone_sender = stream_socket.request_stream(AUDIO);

        thread::spawn(move || {
            while is_streaming() {
                match audio::record_audio_blocking(
                    Arc::new(is_streaming),
                    microphone_sender.clone(),
                    &device,
                    1,
                    false,
                ) {
                    Ok(()) => break,
                    Err(e) => {
                        error!("Audio record error: {e}");

                        continue;
                    }
                }
            }
        })
    } else {
        thread::spawn(|| ())
    };

    let haptics_receive_thread = thread::spawn(move || {
        while is_streaming() {
            let data = match haptics_receiver.recv(STREAMING_RECV_TIMEOUT) {
                Ok(packet) => packet,
                Err(ConnectionError::TryAgain(_)) => continue,
                Err(ConnectionError::Other(_)) => return,
            };
            let Ok(haptics) = data.get_header() else {
                return;
            };

            EVENT_QUEUE.lock().push_back(ClientCoreEvent::Haptics {
                device_id: haptics.device_id,
                duration: haptics.duration,
                frequency: haptics.frequency,
                amplitude: haptics.amplitude,
            });
        }
    });

    let (log_channel_sender, log_channel_receiver) = mpsc::channel();

    let control_send_thread = thread::spawn({
        let disconnect_notif = Arc::clone(&disconnect_notif);
        move || {
            let mut keepalive_deadline = Instant::now();

            #[cfg(target_os = "android")]
            let mut battery_deadline = Instant::now();

            while is_streaming() && *LIFECYCLE_STATE.read() == LifecycleState::Resumed {
                if let (Ok(packet), Some(sender)) = (
                    log_channel_receiver.recv_timeout(STREAMING_RECV_TIMEOUT),
                    &mut *CONTROL_SENDER.lock(),
                ) {
                    if let Err(e) = sender.send(&packet) {
                        info!("Server disconnected. Cause: {e:?}");
                        set_hud_message(SERVER_DISCONNECTED_MESSAGE);

                        break;
                    }
                }

                if Instant::now() > keepalive_deadline {
                    if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                        sender.send(&ClientControlPacket::KeepAlive).ok();

                        keepalive_deadline = Instant::now() + KEEPALIVE_INTERVAL;
                    }
                }

                #[cfg(target_os = "android")]
                if Instant::now() > battery_deadline {
                    let (gauge_value, is_plugged) = platform::get_battery_status();
                    if let Some(sender) = &mut *CONTROL_SENDER.lock() {
                        sender
                            .send(&ClientControlPacket::Battery(crate::BatteryPacket {
                                device_id: *alvr_common::HEAD_ID,
                                gauge_value,
                                is_plugged,
                            }))
                            .ok();
                    }

                    battery_deadline = Instant::now() + Duration::from_secs(5);
                }
            }

            disconnect_notif.notify_one();
        }
    });

    let control_receive_thread = thread::spawn({
        let disconnect_notif = Arc::clone(&disconnect_notif);
        move || {
            let mut disconnection_deadline = Instant::now() + KEEPALIVE_TIMEOUT;
            while is_streaming() {
                let maybe_packet = control_receiver.recv(STREAMING_RECV_TIMEOUT);

                match maybe_packet {
                    Ok(ServerControlPacket::InitializeDecoder(config)) => {
                        decoder::create_decoder(config, settings.video.force_software_decoder);
                    }
                    Ok(ServerControlPacket::Restarting) => {
                        info!("{SERVER_RESTART_MESSAGE}");
                        set_hud_message(SERVER_RESTART_MESSAGE);
                        disconnect_notif.notify_one();
                    }
                    Ok(_) => (),
                    Err(ConnectionError::TryAgain(_)) => {
                        if Instant::now() > disconnection_deadline {
                            info!("{CONNECTION_TIMEOUT_MESSAGE}");
                            set_hud_message(CONNECTION_TIMEOUT_MESSAGE);
                            disconnect_notif.notify_one();
                        } else {
                            continue;
                        }
                    }
                    Err(e) => {
                        info!("{SERVER_DISCONNECTED_MESSAGE} Cause: {e}");
                        set_hud_message(SERVER_DISCONNECTED_MESSAGE);
                        disconnect_notif.notify_one();
                    }
                }

                disconnection_deadline = Instant::now() + KEEPALIVE_TIMEOUT;
            }
        }
    });

    let stream_receive_thread = thread::spawn({
        let disconnect_notif = Arc::clone(&disconnect_notif);
        move || {
            while is_streaming() {
                match stream_socket.recv() {
                    Ok(()) => (),
                    Err(ConnectionError::TryAgain(_)) => continue,
                    Err(e) => {
                        info!("Client disconnected. Cause: {e}");
                        set_hud_message(SERVER_DISCONNECTED_MESSAGE);
                        disconnect_notif.notify_one();
                    }
                }
            }
        }
    });

    *CONTROL_SENDER.lock() = Some(control_sender);
    *TRACKING_SENDER.lock() = Some(tracking_sender);
    *STATISTICS_SENDER.lock() = Some(statistics_sender);
    if let Switch::Enabled(filter_level) = settings.logging.client_log_report_level {
        *LOG_CHANNEL_SENDER.lock() = Some(LogMirrorData {
            sender: log_channel_sender,
            filter_level,
        });
    }
    EVENT_QUEUE.lock().push_back(streaming_start_event);

    *connection_state_lock = ConnectionState::Streaming;

    // Unlock CONNECTION_STATE and block thread
    wait_rwlock(&disconnect_notif, &mut connection_state_lock);

    *connection_state_lock = ConnectionState::Disconnecting;

    *CONTROL_SENDER.lock() = None;
    *TRACKING_SENDER.lock() = None;
    *STATISTICS_SENDER.lock() = None;
    *LOG_CHANNEL_SENDER.lock() = None;

    EVENT_QUEUE
        .lock()
        .push_back(ClientCoreEvent::StreamingStopped);

    #[cfg(target_os = "android")]
    {
        *crate::decoder::DECODER_SINK.lock() = None;
        *crate::decoder::DECODER_SOURCE.lock() = None;
    }

    // Remove lock to allow threads to properly exit:
    drop(connection_state_lock);

    video_receive_thread.join().ok();
    game_audio_thread.join().ok();
    microphone_thread.join().ok();
    haptics_receive_thread.join().ok();
    control_send_thread.join().ok();
    control_receive_thread.join().ok();
    stream_receive_thread.join().ok();

    Ok(())
}
