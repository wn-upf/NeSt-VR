

# ALVR - Air Light VR - Wireless Networking UPF

This is a fork of ALVR v20.6.0, where several metrics have been added in order to characterize the state of the network during a streaming session, plus our ABR contribution -Network-aware Step-wise ABR algorithm for VR streaming (NeSt-VR)- which serves as an alternative ABR scheme to the native `Adaptive` mode of ALVR. We introduce our development process and results in the paper [**Experimenting with Adaptive Bitrate Algorithms for Virtual Reality Streaming over Wi-Fi**](https://arxiv.org/abs/2407.15614)

The following metrics have been added to our fork of ALVR to monitor the arrival of **Video Frames (VF)**. We log every metric for each video frame, while showing the ones we deem most important in the `Statistics` tab of the ALVR dashboard in real time, as is shown next:

<div style="text-align:center">
  <img src="https://github.com/wn-upf/ALVR_ABR_UPF/blob/master/images/Metrics_dashboard.png" alt="Alt Text" width="200" />
</div>

The metrics have been validated via a set of experiments, involving parsing wireshark captures of the generated VR traffic during the stream for an independent comparison. For traces of the experiments, plus a comparison between CBR, `Adaptive` and `NeST-VR` in two different scenarios head to our [Zenodo](https://doi.org/10.5281/zenodo.12773446)

## Metrics: 
### Time-related metrics
* **Client-side frame span**: time interval between the reception of the first packet to the reception of the last packet of a VF. Since ALVR sends packets in bursts, this is indicative of the time the network takes for delivering traffic to the HMD. 
* **Frame inter-arrival time**: time interval between the reception of the last packet received in a complete VF and the actual one, if it's completely received too (with no losses).
Indicates the time interval between the instants where each of the two VF is correctly received and can be decoded.
* **Video Frame Round-Trip Time (VF-RTT)**: time it takes for a complete VF to travel from the server to the client and for our supplementary UL packet —promptly sent upon the complete reception of the VF— to reach the server. ALVR already provided originally a `network_s` metric for network latency that resembles the VF-RTT interval through [substraction of other pipeline delays from the total](https://github.com/alvr-org/ALVR/blob/386a1a8d2de636c3d15d8aa3bbb5810c82c1922c/alvr/server_core/src/statistics.rs#L211), while ours is an independent measurement of the RTT between Server and HMD.  

### Reliability metrics 
* **Packet loss**: number of packets lost in the interval between two VF correct receptions. We add a sequence number to each video packet received in order to estimate the number of packets sent during the interval. Thus packet losses can be accurately measured at the HMD. 
* **Frames skipped and dropped**: To distinguish between frame losses coming from the network or decoder issues respectively. 
* **Packets duplicated**: While they have no effect on the quality of the stream, duplicated packets are accounted for. 
### Data rate metrics 
* **Instantaneous video network throughput**: rate at which video data is received by the client, measured in the interval between two VFs receptions. It reflects
the network’s capability to sustain the desired video quality according to the encoder’s target bitrate. 
* **Peak network throughput**: is computed as the ratio between the VF’s size and its client-side frame span. It is used by us as an estimate of the network capacity, since ALVR sends each VF in a quick burst, we use a sliding window filter over the peak throughput in order to estimate the average maximum speed of delivery of the network.

### Network Stability metrics
* **VF jitter**: variation in VF time deliveries, providing insight into the smoothness of video playback. It is computed as the sample standard deviation of frame inter-arrival times over a 256 sample sliding window.
* **Video packet jitter**: variability in video packet arrival times, using the formulas defined in [RFC 3550](https://datatracker.ietf.org/doc/html/rfc3550), providing insight into the consistency of packet delivery.
* **Filtered one-way delay gradient**: computed using the formulas specified in the [GCC design paper](https://dl.acm.org/doi/10.1145/2910017.2910605), indicates the rate of change and its direction in queueing and transmission delays between two consecutive VFs. We implement the Kalman filter with state noise variance of `Q = 10^(−7)`, which we judged appropiate for 90 FPS streams. 
### Extras
In order to facilitate testing, we make two small changes to ALVR's pipeline:
* An extra uplink (HMD to server) packet is sent when a full VF is received. The original uplink packet sent in ALVR when the same VF is visualized is mantained. This packet is used to compute the VF-RTT metric mentioned previously, while fields with metrics are included inside the 56 byte packet, to effectively reduce at the server the delay of feedback from the HMD.
* A setting to enable a periodic timer at the HMD to periodically request IDR frames with configurable frequency, in order to mimic the Group of Pictures structure for video streaming.

## NeST-VR: 

In order to ensure that the encoder bitrate doesn't exceed the capacity of the network, we propose our algorithm 'Network-aware Step-wise ABR algorithm for VR streaming' (NeST-VR) to dynamically change the bitrate of the stream based on network conditions through a simple decision-making process.
The algorithm operates in intervals of τ seconds, adjusting the target bitrate (B_v) in steps of β Mbps to mitigate abrupt quality shifts. Its logic diagram can be found next: 
<div style="text-align:center">
  <img src="https://github.com/wn-upf/ALVR_ABR_UPF/blob/master/images/stepwise-abr-nest.png" alt="Alt Text" width="300" />
</div>

At each update, the Network Frame Ratio (NFR) and Video Frame Round-Trip Time (VF-RTT) metrics are passed through an `n`-sample sliding window acting as low-pass filters for the network metrics. 

* The NFR is defined as `(fps_rx/fps_tx)`, where `fps_rx` is the rate of correct VF receival over the HMD, while `fps_tx` is the rate of VF transmission from the server. Thus, the NFR indicates network reliability in delivering complete Video Frames (VFs) to the HMD. 

* VF-RTT measures the latency associated to the network in delivering VFs. Excessive latency is unencouraged, so we simply put thresholds for these two metrics to vary the bitrate accordingly.
While jitter or packet loss metrics are not explicitly used by NeST-VR, they have a direct effect on the NFR if their levels are excessive, and so are accounted for implicitly. 

The decision-making process of NeST-VR involves the following steps:  
- **NFR Below Threshold ρ:** Reduces bitrate to minimize packet loss and improve frame delivery.
- **NFR Above ρ and VF-RTT Below Threshold σ:** May increase bitrate with probability `γ` to exploit available network capacity.
- **Both NFR and VF-RTT Above Thresholds:** Decreases bitrate to reduce the perceived latency, with probability `1 - γ`.

* Lastly, the bitrate is bounded by `C_NeST` (computed as the 256-sample average over the peak network throughput), multiplied by a scaling factor `m` so we ensure that the bitrate is conservatively kept under the estimated capacity. This mechanism ensures the encoder bitrate is quickly reduced under capacity drops in order to not stall the stream.
Otherwise, video freezes would be common, as the network couldn't handle the throughput. We illustrate this effect next, where an emulated capacity drop is applied at the network for both CBR and NeST-VR.
![Alt Text](https://github.com/wn-upf/ALVR_ABR_UPF/blob/master/images/MaxR-video-July.gif)

 CBR (left) freezes the video stream at the HMD due to losses since the encoder bitrate is mantained at 100 Mbps (over the emulated capacity), while NeST-VR (or ALVR's Adaptive, for instance) reduces the bitrate accordingly. There is a short episode of losses plus a latency peak (visualized black borders when head moves) at first when the effect is applied on NeST-VR, but the stream recovers eventually through decreasing the encoder bitrate. 

The experiments done with NeST-VR in the paper use the settings outlined in the following tables. We originally tested for drops in capacity through emulation, and experiments under mobility through a hallway.
However, we leave most parameters configurable in the dashboard, as in the next image. 

<div style="text-align:center">
  <img src="https://github.com/wn-upf/ALVR_ABR_UPF/blob/master/images/Settings_NeST-VR.png" alt="Alt Text2" width="300" />
</div>


| Parameter | Symbol |
|-----------|--------|
| Adjustment period | τ |
| Sliding window size | n |
| min Bitrate | Bmin |
| max Bitrate | Bmax |
| initial Bitrate | B0 |
| Step size | β |
| Est. Capacity scaling factor | m |
| VF-RTT Exploration Prob. | γ |
| NFR thresh. | ρ |
| VF-RTT thresh. scaling factor | ς |



| τ  | 1 s     | B_min  | 10 Mbps  | B_0   | 30 Mbps | m  | 0.90 | ρ  | 0.95 |
|----|---------|--------|----------|-------|---------|----|------|----|------|
| n  | 256     | B_max  | 100 Mbps | β     | 10 Mbps | γ  | 0.25 | ζ  | 2.0  |

## Logging: 

ALVR has an option to store raw JSON logs in .txt files, as `session_log.txt`, that we leave on by default. 
Through examining the `GraphStatistics` events in the text file, the range of original ALVR statistics for each frame can be obtained. In a similar manner, we add an event for `GraphNetworkStatistics` that logs our new metrics computed for each video frame in the stream, logged every time the extra uplink packet is received.  
When using NeST-VR, an additional `HeuristicStats` event is generated for each update in the algorithm, with each of the metrics involved in the decision making process.  

For instance, `GraphNetworkStatistics` includes the following fields: 
```rust 
pub struct GraphNetworkStatistics {
    pub frame_index: u32,
    pub client_fps: f32,
    pub server_fps: f32,
    pub frame_span_ms: f32,
    pub interarrival_jitter_ms: f32,
    pub ow_delay_ms: f32,
    pub filtered_ow_delay_ms: f32,
    pub rtt_ms: f32,
    pub frame_interarrival_ms: f32,
    pub frame_jitter_ms: f32,
    pub frames_skipped: u32,
    pub shards_lost: isize,
    pub shards_duplicated: u32,
    pub instant_network_throughput_bps: f32,
    pub peak_network_throughput_bps: f32,
    pub nominal_bitrate: NominalBitrateStats,
    pub interval_avg_plot_throughput: f32,
}
```

## Requirements

-   A supported standalone VR headset - Meta Quest 2 was used by us. 

-   SteamVR

-   High-end gaming PC
    -   See OS compatibility table above.
    -   NVIDIA GPU that supports NVENC (1000 GTX Series or higher) (or with an AMD GPU that supports AMF VCE) with the latest driver.
    -   Laptops with an onboard (Intel HD, AMD iGPU) and an additional dedicated GPU (NVidia GTX/RTX, AMD HD/R5/R7): you should assign the dedicated GPU or "high performance graphics adapter" to the applications ALVR, SteamVR for best performance and compatibility. (NVidia: Nvidia control panel->3d settings->application settings; AMD: similiar way)

-   802.11ac/ax 5Ghz wireless or ethernet wired connection  
    -   It is recommended to use 802.11ac 5Ghz for the headset and ethernet for PC  
    -   You need to connect both the PC and the headset to same router (or use a routed connection as described [here](https://github.com/alvr-org/ALVR/wiki/ALVR-v14-and-Above))

