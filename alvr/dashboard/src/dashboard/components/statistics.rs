use crate::{dashboard::theme::graph_colors, dashboard::ServerRequest};
use alvr_events::{GraphNetworkStatistics, GraphStatistics, StatisticsSummary};
use alvr_gui_common::theme;
use eframe::{
    egui::{
        popup, pos2, vec2, Align2, Color32, FontId, Frame, Id, Painter, Rect, RichText, Rounding,
        ScrollArea, Shape, Stroke, Ui,
    },
    emath::RectTransform,
    epaint::Pos2,
};
use statrs::statistics::{self, OrderStatistics};
use std::{collections::VecDeque, ops::RangeInclusive};

const GRAPH_HISTORY_SIZE: usize = 1000;
const UPPER_QUANTILE: f64 = 0.80;
// const LOWER_QUANTILE: f64 = 0.2;
// const MIDDLE_QUANTILE: f64 = 0.5;
fn draw_lines(painter: &Painter, points: Vec<Pos2>, color: Color32) {
    painter.add(Shape::line(points, Stroke::new(1.0, color)));
}

pub struct StatisticsTab {
    history: VecDeque<GraphStatistics>,
    history_network: VecDeque<GraphNetworkStatistics>,
    last_statistics_summary: Option<StatisticsSummary>,
}

impl StatisticsTab {
    pub fn new() -> Self {
        Self {
            history: vec![GraphStatistics::default(); GRAPH_HISTORY_SIZE]
                .into_iter()
                .collect(),
            history_network: vec![GraphNetworkStatistics::default(); GRAPH_HISTORY_SIZE]
                .into_iter()
                .collect(),
            last_statistics_summary: None,
        }
    }

    pub fn update_statistics(&mut self, statistics: StatisticsSummary) {
        self.last_statistics_summary = Some(statistics);
    }

    pub fn update_graph_statistics(&mut self, statistics: GraphStatistics) {
        self.history.pop_front();
        self.history.push_back(statistics);
    }

    pub fn update_graph_network_statistics(&mut self, statistics: GraphNetworkStatistics) {
        self.history_network.pop_front();
        self.history_network.push_back(statistics);
    }

    pub fn ui(&mut self, ui: &mut Ui) -> Option<ServerRequest> {
        if let Some(stats) = &self.last_statistics_summary {
            ScrollArea::new([false, true]).show(ui, |ui| {
                let available_width = ui.available_width();
                self.draw_latency_graph(ui, available_width);
                self.draw_fps_graph(ui, available_width);
                self.draw_bitrate_graph(ui, available_width);
                self.draw_throughput_graphs(ui, available_width);
                self.draw_jitter(ui, available_width);
                self.draw_frameloss(ui, available_width);
                self.draw_frame_span_interarrival(ui, available_width);
                self.draw_statistics_overview(ui, stats);
            });
        } else {
            ui.heading("No statistics available");
        }

        None
    }

    fn draw_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &GraphStatistics),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);

            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();
            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.0}", min),
                FontId::monospace(20.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.0}", max),
                FontId::monospace(20.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE - 1);

            popup::show_tooltip(ui.ctx(), Id::new("popup"), |ui| {
                tooltip_content(ui, self.history.get(history_index).unwrap())
            });
        }
    }

    fn draw_network_graph(
        &self,
        ui: &mut Ui,
        available_width: f32,
        title: &str,
        data_range: RangeInclusive<f32>,
        graph_content: impl FnOnce(&Painter, RectTransform),
        tooltip_content: impl FnOnce(&mut Ui, &GraphNetworkStatistics),
    ) {
        ui.add_space(10.0);
        ui.label(RichText::new(title).size(20.0));

        let canvas_response = Frame::canvas(ui.style()).show(ui, |ui| {
            ui.ctx().request_repaint();
            let size = available_width * vec2(1.0, 0.2);

            let (_id, canvas_rect) = ui.allocate_space(size);

            let max = *data_range.end();
            let min = *data_range.start();
            let data_rect = Rect::from_x_y_ranges(0.0..=GRAPH_HISTORY_SIZE as f32, max..=min);
            let to_screen = RectTransform::from_to(data_rect, canvas_rect);

            let painter = ui.painter().with_clip_rect(canvas_rect);

            graph_content(&painter, to_screen);

            ui.painter().text(
                to_screen * pos2(0.0, min),
                Align2::LEFT_BOTTOM,
                format!("{:.0}", min),
                FontId::monospace(20.0),
                Color32::GRAY,
            );
            ui.painter().text(
                to_screen * pos2(0.0, max),
                Align2::LEFT_TOP,
                format!("{:.0}", max),
                FontId::monospace(20.0),
                Color32::GRAY,
            );

            data_rect
        });

        if let Some(pos) = canvas_response.response.hover_pos() {
            let graph_pos =
                RectTransform::from_to(canvas_response.response.rect, canvas_response.inner) * pos;
            let history_index = (graph_pos.x as usize).clamp(0, GRAPH_HISTORY_SIZE - 1);

            popup::show_tooltip(ui.ctx(), Id::new("popup"), |ui| {
                tooltip_content(ui, self.history_network.get(history_index).unwrap())
            });
        }
    }

    fn draw_latency_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history
                .iter()
                .map(|stats| stats.total_pipeline_latency_s as f64)
                .collect::<Vec<_>>(),
        );

        self.draw_graph(
            ui,
            available_width,
            "Latency (ms)",
            0.0..=(data.quantile(UPPER_QUANTILE)) as f32 * 1000.0,
            |painter, to_screen_trans| {
                for i in 0..GRAPH_HISTORY_SIZE {
                    let stats = self.history.get(i).unwrap();
                    let mut offset = 0.0;
                    for (value, color) in &[
                        (stats.game_time_s, graph_colors::RENDER_VARIANT),
                        (stats.server_compositor_s, graph_colors::RENDER),
                        (stats.encoder_s, graph_colors::TRANSCODE),
                        (stats.network_s, graph_colors::NETWORK),
                        (stats.decoder_s, graph_colors::TRANSCODE),
                        (stats.decoder_queue_s, graph_colors::IDLE),
                        (stats.client_compositor_s, graph_colors::RENDER),
                        (stats.vsync_queue_s, graph_colors::IDLE),
                    ] {
                        painter.rect_filled(
                            Rect {
                                min: to_screen_trans * pos2(i as f32, offset + value * 1000.0),
                                max: to_screen_trans * pos2(i as f32 + 2.0, offset),
                            },
                            Rounding::ZERO,
                            *color,
                        );
                        offset += value * 1000.0;
                    }
                }
            },
            |ui, stats| {
                use graph_colors::*;

                fn label(ui: &mut Ui, text: &str, value_s: f32, color: Color32) {
                    ui.colored_label(color, &format!("{text}: {:.2} ms", value_s * 1000.0));
                }
                label(
                    ui,
                    "Total latency",
                    stats.total_pipeline_latency_s,
                    theme::FG,
                );
                label(ui, "Client VSync", stats.vsync_queue_s, IDLE);
                label(ui, "Client compositor", stats.client_compositor_s, RENDER);
                label(ui, "Decoder queue", stats.decoder_queue_s, IDLE);
                label(ui, "Decode", stats.decoder_s, TRANSCODE);
                label(ui, "Network", stats.network_s, NETWORK);
                label(ui, "Encode", stats.encoder_s, TRANSCODE);
                label(ui, "Streamer compositor", stats.server_compositor_s, RENDER);
                label(ui, "Game render", stats.game_time_s, RENDER_VARIANT);
            },
        );
    }

    fn draw_fps_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.client_fps)
                .chain(self.history_network.iter().map(|stats| stats.server_fps))
                .map(|v| v as f64)
                .collect::<Vec<_>>(),
        );
        let upper_quantile = data.quantile(UPPER_QUANTILE);
        let lower_quantile = data.quantile(1.0 - UPPER_QUANTILE);

        let max = upper_quantile + (upper_quantile - lower_quantile);
        let min = 0.0;

        self.draw_network_graph(
            ui,
            available_width,
            "Framerate",
            min as f32..=max as f32,
            |painter, to_screen_trans| {
                let (server_fps_points, client_fps_points) = (0..GRAPH_HISTORY_SIZE)
                    .map(|i| {
                        (
                            to_screen_trans * pos2(i as f32, self.history_network[i].server_fps),
                            to_screen_trans * pos2(i as f32, self.history_network[i].client_fps),
                        )
                    })
                    .unzip();

                draw_lines(painter, server_fps_points, graph_colors::SERVER_FPS);
                draw_lines(painter, client_fps_points, graph_colors::CLIENT_FPS);
            },
            |ui, stats| {
                ui.colored_label(
                    graph_colors::SERVER_FPS,
                    format!("Streamer FPS: {:.2}", stats.server_fps),
                );
                ui.colored_label(
                    graph_colors::CLIENT_FPS,
                    format!("Client FPS: {:.2}", stats.client_fps),
                );
            },
        );
    }

    fn draw_jitter(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.ow_delay_ms as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Shards Jitter Graph",
            -5.0..=(data.quantile(UPPER_QUANTILE) * 5.0) as f32,
            |painter, to_screen_trans| {
                let mut interarrival_jitter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut ow_delay = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut filtered_ow_delay = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                // let mut threshold_gcc = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let value_fowd = pointer_graphstatistics.filtered_ow_delay_ms;
                    filtered_ow_delay.push(to_screen_trans * pos2(i as f32, value_fowd));

                    let value_jitt = pointer_graphstatistics.interarrival_jitter_ms;
                    interarrival_jitter.push(to_screen_trans * pos2(i as f32, value_jitt));

                    let value_owd = pointer_graphstatistics.ow_delay_ms;
                    ow_delay.push(to_screen_trans * pos2(i as f32, value_owd));

                    // let value_thr = pointer_graphstatistics.threshold_gcc;
                    // threshold_gcc.push(to_screen_trans * pos2(i as f32, value_thr));
                }
                draw_lines(painter, filtered_ow_delay, Color32::LIGHT_YELLOW);
                draw_lines(painter, interarrival_jitter, Color32::RED);
                draw_lines(painter, ow_delay, Color32::BLUE);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.7} ms", value));
                    }
                }
                maybe_label(
                    ui,
                    "Filtered OW Delay",
                    Some(stats.filtered_ow_delay_ms),
                    Color32::LIGHT_YELLOW,
                );
                maybe_label(
                    ui,
                    "Shard Interarrival Jitter",
                    Some(stats.interarrival_jitter_ms),
                    Color32::RED,
                );
                maybe_label(ui, "OW Delay", Some(stats.ow_delay_ms), Color32::BLUE);

                // maybe_label(
                //     ui,
                //     "Threshold from GCC",
                //     Some(stats.threshold_gcc),
                //     Color32::GOLD,
                // );
            },
        )
    }

    fn draw_frameloss(&self, ui: &mut Ui, available_width: f32) {
        self.draw_network_graph(
            ui,
            available_width,
            "Frames Skipped, Shards Lost and Shards Duplicated Graph",
            -20.0..=20.0 as f32,
            |painter, to_screen_trans| {
                let mut frameskipped = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut shardloss = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut dup_shards = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let val_fs = pointer_graphstatistics.frames_skipped;
                    frameskipped.push(to_screen_trans * pos2(i as f32, val_fs as f32));

                    let val_sl = pointer_graphstatistics.shards_lost;
                    shardloss.push(to_screen_trans * pos2(i as f32, val_sl as f32));

                    let val_dups = pointer_graphstatistics.shards_duplicated;
                    dup_shards.push(to_screen_trans * pos2(i as f32, val_dups as f32));
                }

                draw_lines(painter, frameskipped, Color32::LIGHT_BLUE);
                draw_lines(painter, shardloss, Color32::LIGHT_RED);
                draw_lines(painter, dup_shards, Color32::DARK_GREEN);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.0} ", value));
                    }
                }
                let graphstats = stats;
                maybe_label(
                    ui,
                    "Frames Skipped",
                    Some(graphstats.frames_skipped as f32),
                    Color32::LIGHT_BLUE,
                );
                maybe_label(
                    ui,
                    "Shards Lost",
                    Some(graphstats.shards_lost as f32),
                    Color32::LIGHT_RED,
                );
                maybe_label(
                    ui,
                    "Shards Duplicated",
                    Some(graphstats.shards_duplicated as f32),
                    Color32::DARK_GREEN,
                );
            },
        )
    }

    fn draw_frame_span_interarrival(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.frame_interarrival_ms as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Frame Span and Frame Interarrival Graph",
            0.0..=(data.quantile(UPPER_QUANTILE) * 2.0) as f32,
            |painter, to_screen_trans| {
                let mut frame_span = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut frame_interarrival = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut frame_jitter = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];

                    let fs = pointer_graphstatistics.frame_span_ms;
                    frame_span.push(to_screen_trans * pos2(i as f32, fs as f32));

                    let fi = pointer_graphstatistics.frame_interarrival_ms;
                    frame_interarrival.push(to_screen_trans * pos2(i as f32, fi as f32));

                    let val_std = pointer_graphstatistics.frame_jitter_ms;
                    frame_jitter.push(to_screen_trans * pos2(i as f32, val_std));
                }
                draw_lines(painter, frame_interarrival, Color32::LIGHT_RED);
                draw_lines(painter, frame_span, Color32::LIGHT_BLUE);
                draw_lines(painter, frame_jitter, Color32::LIGHT_YELLOW);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.6} ms", value));
                    }
                }
                let graphstats = stats;

                maybe_label(
                    ui,
                    "Frame span",
                    Some(graphstats.frame_span_ms as f32),
                    Color32::LIGHT_BLUE,
                );
                maybe_label(
                    ui,
                    "Frame Interarrival",
                    Some(graphstats.frame_interarrival_ms as f32),
                    Color32::LIGHT_RED,
                );
                maybe_label(
                    ui,
                    "Frame interarrival std (frame jitter).",
                    Some(stats.frame_jitter_ms),
                    Color32::LIGHT_YELLOW,
                );
            },
        )
    }

    fn draw_throughput_graphs(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history_network
                .iter()
                .map(|stats| stats.network_throughput_bps as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_network_graph(
            ui,
            available_width,
            "Video Network Throughput",
            0.0..=150.0 as f32,
            |painter, to_screen_trans| {
                let mut network_throughput_bps: Vec<Pos2> = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                let mut requested = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let pointer_graphstatistics = &self.history_network[i];
                    let nom_br = &self.history_network[i].nominal_bitrate;

                    let value_nw = pointer_graphstatistics.network_throughput_bps;
                    network_throughput_bps.push(to_screen_trans * pos2(i as f32, value_nw / 1e6));

                    requested.push(to_screen_trans * pos2(i as f32, nom_br.requested_bps / 1e6));
                }
                draw_lines(painter, network_throughput_bps, Color32::BLUE);
                draw_lines(painter, requested, theme::OK_GREEN);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.4} Mbps", value / 1e6));
                    }
                }
                let graphstats = stats;
                let n = &stats.nominal_bitrate;

                maybe_label(
                    ui,
                    "Network Throughput",
                    Some(graphstats.network_throughput_bps),
                    Color32::BLUE,
                );
                maybe_label(
                    ui,
                    "Requested Bitrate",
                    Some(n.requested_bps),
                    theme::OK_GREEN,
                );
            },
        )
    }

    fn draw_bitrate_graph(&self, ui: &mut Ui, available_width: f32) {
        let mut data = statistics::Data::new(
            self.history
                .iter()
                .map(|stats| stats.actual_bitrate_bps as f64)
                .collect::<Vec<_>>(),
        );
        self.draw_graph(
            ui,
            available_width,
            "Bitrate (ALVR's computation)",
            0.0..=(data.quantile(UPPER_QUANTILE) * 2.0) as f32 / 1e6,
            |painter, to_screen_trans| {
                let mut scaled_calculated = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut decoder_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut network_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut encoder_latency_limiter = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut manual_max = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut manual_min = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut requested = Vec::with_capacity(GRAPH_HISTORY_SIZE);
                let mut actual = Vec::with_capacity(GRAPH_HISTORY_SIZE);

                for i in 0..GRAPH_HISTORY_SIZE {
                    let nom_br = &self.history[i].nominal_bitrate;

                    if let Some(value) = nom_br.scaled_calculated_bps {
                        scaled_calculated.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.decoder_latency_limiter_bps {
                        decoder_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.network_latency_limiter_bps {
                        network_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.encoder_latency_limiter_bps {
                        encoder_latency_limiter.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.manual_max_bps {
                        manual_max.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }
                    if let Some(value) = nom_br.manual_min_bps {
                        manual_min.push(to_screen_trans * pos2(i as f32, value / 1e6))
                    }

                    requested.push(to_screen_trans * pos2(i as f32, nom_br.requested_bps / 1e6));
                    actual.push(
                        to_screen_trans * pos2(i as f32, self.history[i].actual_bitrate_bps / 1e6),
                    );
                }

                draw_lines(painter, scaled_calculated, Color32::GRAY);
                draw_lines(painter, encoder_latency_limiter, graph_colors::TRANSCODE);
                draw_lines(painter, network_latency_limiter, graph_colors::NETWORK);
                draw_lines(painter, decoder_latency_limiter, graph_colors::TRANSCODE);
                draw_lines(painter, manual_max, graph_colors::RENDER);
                draw_lines(painter, manual_min, graph_colors::RENDER);
                draw_lines(painter, requested, theme::OK_GREEN);
                draw_lines(painter, actual, theme::FG);
            },
            |ui, stats| {
                fn maybe_label(
                    ui: &mut Ui,
                    text: &str,
                    maybe_value_bps: Option<f32>,
                    color: Color32,
                ) {
                    if let Some(value) = maybe_value_bps {
                        ui.colored_label(color, &format!("{text}: {:.2} Mbps", value / 1e6));
                    }
                }

                let n = &stats.nominal_bitrate;

                maybe_label(
                    ui,
                    "Initial calculated",
                    n.scaled_calculated_bps,
                    Color32::GRAY,
                );
                maybe_label(
                    ui,
                    "Encoder latency limiter",
                    n.encoder_latency_limiter_bps,
                    graph_colors::TRANSCODE,
                );
                maybe_label(
                    ui,
                    "Network latency limiter",
                    n.network_latency_limiter_bps,
                    graph_colors::NETWORK,
                );
                maybe_label(
                    ui,
                    "Decoder latency limiter",
                    n.decoder_latency_limiter_bps,
                    graph_colors::TRANSCODE,
                );
                maybe_label(ui, "Manual max", n.manual_max_bps, graph_colors::RENDER);
                maybe_label(ui, "Manual min", n.manual_min_bps, graph_colors::RENDER);
                maybe_label(ui, "Requested", Some(n.requested_bps), theme::OK_GREEN);
                maybe_label(
                    ui,
                    "Actual recorded",
                    Some(stats.actual_bitrate_bps),
                    theme::FG,
                );
            },
        )
    }

    fn draw_statistics_overview(&self, ui: &mut Ui, statistics: &StatisticsSummary) {
        ui.add_space(10.0);

        ui.columns(2, |ui| {
            ui[0].label("Total packets:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.video_packets_total, statistics.video_packets_per_sec
            ));

            ui[0].label("Total sent:");
            ui[1].label(&format!("{} MB", statistics.video_mbytes_total));

            ui[0].label("Bitrate:");
            ui[1].label(&format!("{:.1} Mbps", statistics.video_mbits_per_sec));

            ui[0].label("Throughput:");
            ui[1].label(&format!(
                "{:.1} Mbps",
                statistics.video_throughput_mbits_per_sec
            ));

            ui[0].label("Game delay:");
            ui[1].label(&format!("{:.2} ms", statistics.game_delay_average_ms));

            ui[0].label("Server compositor delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.server_compositor_delay_average_ms
            ));

            ui[0].label("Encoder delay:");
            ui[1].label(&format!("{:.2} ms", statistics.encode_delay_average_ms));

            ui[0].label("Network delay:");
            ui[1].label(&format!("{:.2} ms", statistics.network_delay_average_ms));

            ui[0].label("Decoder delay:");
            ui[1].label(&format!("{:.2} ms", statistics.decode_delay_average_ms));

            ui[0].label("Decoder queue delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.decoder_queue_delay_average_ms
            ));

            ui[0].label("Client compositor delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.client_compositor_average_ms
            ));

            ui[0].label("Vsync delay:");
            ui[1].label(&format!(
                "{:.2} ms",
                statistics.vsync_queue_delay_average_ms
            ));

            ui[0].label("Total latency:");
            ui[1].label(&format!(
                "{:.0} ms",
                statistics.total_pipeline_latency_average_ms
            ));

            ui[0].label("Frame jitter:");
            ui[1].label(&format!("{:.0} ms", statistics.frame_jitter_ms));

            ui[0].label("Total packets dropped:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.packets_dropped_total, statistics.packets_dropped_per_sec
            ));

            ui[0].label("Total packets skipped:");
            ui[1].label(&format!(
                "{} packets ({} packets/s)",
                statistics.packets_skipped_total, statistics.packets_skipped_per_sec
            ));

            ui[0].label("Client FPS:");
            ui[1].label(&format!("{} FPS", statistics.client_fps));

            ui[0].label("Streamer FPS:");
            ui[1].label(&format!("{} FPS", statistics.server_fps));

            ui[0].label("Headset battery");
            ui[1].label(&format!(
                "{}% ({})",
                statistics.battery_hmd,
                if statistics.hmd_plugged {
                    "plugged"
                } else {
                    "unplugged"
                }
            ));
        });
    }
}