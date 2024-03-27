// Note: for StreamSocket, the client uses a server socket, the server uses a client socket.
// This is because of certificate management. The server needs to trust a client and its certificate
//
// StreamSender and StreamReceiver endpoints allow for convenient conversion of the header to/from
// bytes while still handling the additional byte buffer with zero copies and extra allocations.

// Performance analysis:
// We want to minimize the transmission time for various sizes of packets.
// The current code locks the write socket *per shard* and not *per packet*. This leds to the best
// performance outcome given that the possible packets can be either very small (one shard) or very
// large (hundreds/thousands of shards, for video). if we don't allow interleaving shards, a very
// small packet will need to wait a long time before getting received if there was an ongoing
// transmission of a big packet before. If we allow interleaving shards, small packets can be
// transmitted quicker, with only minimal latency increase for the ongoing transmission of the big
// packet.
// Note: We can't clone the underlying socket for each StreamSender and the mutex around the socket
// cannot be removed. This is because we need to make sure at least shards are written whole.
use crate::backend::{tcp, udp, SocketReader, SocketWriter};
use alvr_common::{
    anyhow::Result, debug, parking_lot::Mutex, AnyhowToCon, ConResult, HandleTryAgain, ToCon,
};
use alvr_packets::VIDEO;
use alvr_session::{DscpTos, SocketBufferSize, SocketProtocol};
use core::cmp::Reverse;
use serde::{de::DeserializeOwned, Serialize};
use socket2::Socket;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    marker::PhantomData,
    mem,
    net::{IpAddr, TcpListener, UdpSocket},
    sync::{mpsc, Arc},
    time::{Duration, Instant},
};

const Q_KALMAN: f32 = 10E-5;

pub struct KalmanFilter {
    ow_delay: f32,
    m_current: f32,
    p_current: f32,
    noise_prev: f32,
    residual_z: f32,
    noise_estimation: f32,
    m_prev: f32,
    p_prev: f32,
    k_gain: f32,
    measured_delay: f32,
    abs_m: f32,
}
impl Default for KalmanFilter {
    fn default() -> Self {
        KalmanFilter {
            ow_delay: 0.0,
            m_current: 0.0,
            p_current: 0.0,
            noise_prev: 0.0,
            residual_z: 0.0,
            noise_estimation: 0.0,
            m_prev: 0.0,
            p_prev: 0.0,
            k_gain: 0.0,
            measured_delay: 0.0,
            abs_m: 0.0,
        }
    }
}

const SHARD_PREFIX_SIZE: usize = mem::size_of::<u32>() // packet length - field itself (4 bytes)
    + mem::size_of::<u16>() // stream ID
    + mem::size_of::<u32>() // packet index
    + mem::size_of::<u32>() // shards count
    + mem::size_of::<u32>() // shards index
    + mem::size_of::<f32>();

/// Memory buffer that contains a hidden prefix
#[derive(Default)]
pub struct Buffer<H = ()> {
    inner: Vec<u8>,
    hidden_offset: usize, // this corresponds to prefix + header
    length: usize,
    _phantom: PhantomData<H>,
}

impl<H> Buffer<H> {
    /// Length of payload (without prefix)
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the whole payload of the buffer
    pub fn get(&self) -> &[u8] {
        &self.inner[self.hidden_offset..][..self.length]
    }

    /// If the range is outside the valid range, new space will be allocated
    /// NB: the offset parameter is applied on top of the internal offset of the buffer
    pub fn get_range_mut(&mut self, offset: usize, size: usize) -> &mut [u8] {
        let required_size = self.hidden_offset + offset + size;
        if required_size > self.inner.len() {
            self.inner.resize(required_size, 0);
        }

        self.length = self.length.max(offset + size);

        &mut self.inner[self.hidden_offset + offset..][..size]
    }

    /// If length > current length, allocate more space
    pub fn set_len(&mut self, length: usize) {
        self.inner.resize(self.hidden_offset + length, 0);
        self.length = length;
    }
}

#[derive(Clone)]
pub struct StreamSender<H> {
    inner: Arc<Mutex<Box<dyn SocketWriter>>>,
    stream_id: u16,
    max_packet_size: usize,
    // if the packet index overflows the worst that happens is a false positive packet loss
    next_packet_index: u32,
    used_buffers: Vec<Vec<u8>>,
    _phantom: PhantomData<H>,

    last_shard_instant: Option<Instant>,
    shards_count: usize,
}

impl<H> StreamSender<H> {
    pub fn get_shards_count(&self) -> usize {
        self.shards_count
    }
    pub fn get_last_packet_id(&self) -> u32 {
        self.next_packet_index - 1
    }
    /// Shard and send a buffer with zero copies and zero allocations.
    /// The prefix of each shard is written over the previously sent shard to avoid reallocations.
    pub fn send(&mut self, mut buffer: Buffer<H>) -> Result<()> {
        let max_shard_data_size = self.max_packet_size - SHARD_PREFIX_SIZE;
        let actual_buffer_size = buffer.hidden_offset + buffer.length;
        let data_size = actual_buffer_size - SHARD_PREFIX_SIZE;
        let shards_count = (data_size as f32 / max_shard_data_size as f32).ceil() as usize;

        let mut duration: f32 = Duration::ZERO.as_secs_f32();

        for idx in 0..shards_count {
            // this overlaps with the previous shard, this is intended behavior and allows to
            // reduce allocations
            let packet_start_position = idx * max_shard_data_size;
            let sub_buffer = &mut buffer.inner[packet_start_position..];

            // NB: true shard length (account for last shard that is smaller)
            let packet_length = usize::min(
                self.max_packet_size,
                actual_buffer_size - packet_start_position,
            );

            let now = Instant::now();
            if let Some(shard_instant) = self.last_shard_instant {
                duration = now.duration_since(shard_instant).as_secs_f32();
            }
            // todo: switch to little endian
            // todo: do not remove sizeof<u32> for packet length
            sub_buffer[0..4]
                .copy_from_slice(&((packet_length - mem::size_of::<u32>()) as u32).to_be_bytes());
            sub_buffer[4..6].copy_from_slice(&self.stream_id.to_be_bytes());
            sub_buffer[6..10].copy_from_slice(&self.next_packet_index.to_be_bytes());
            sub_buffer[10..14].copy_from_slice(&(shards_count as u32).to_be_bytes());
            sub_buffer[14..18].copy_from_slice(&(idx as u32).to_be_bytes());
            sub_buffer[18..22].copy_from_slice(&duration.to_be_bytes());

            self.inner.lock().send(&sub_buffer[..packet_length])?;

            self.last_shard_instant = Some(now);
        }

        self.shards_count = shards_count;

        self.last_shard_instant = None;

        self.next_packet_index += 1;

        self.used_buffers.push(buffer.inner);

        Ok(())
    }
}

impl<H: Serialize> StreamSender<H> {
    pub fn get_buffer(&mut self, header: &H) -> Result<Buffer<H>> {
        let mut buffer = self.used_buffers.pop().unwrap_or_default();

        let header_size = bincode::serialized_size(header)? as usize;
        let hidden_offset = SHARD_PREFIX_SIZE + header_size;

        if buffer.len() < hidden_offset {
            buffer.resize(hidden_offset, 0);
        }

        bincode::serialize_into(&mut buffer[SHARD_PREFIX_SIZE..hidden_offset], header)?;

        Ok(Buffer {
            inner: buffer,
            hidden_offset,
            length: 0,
            _phantom: PhantomData,
        })
    }

    pub fn send_header(&mut self, header: &H) -> Result<()> {
        let buffer = self.get_buffer(header)?;
        self.send(buffer)
    }
}

pub struct ReceiverData<H> {
    buffer: Option<Vec<u8>>,
    size: usize, // counting the prefix
    used_buffer_queue: mpsc::Sender<Vec<u8>>,
    had_packet_loss: bool,
    _phantom: PhantomData<H>,

    frame_index: u32,

    frame_span: f32,
    frame_interarrival: f32,

    jitter_avg_frame: f32,
    filtered_ow_delay: f32,

    rx_bytes: u32,
    bytes_in_frame: u32,
    bytes_in_frame_app: u32,

    frames_skipped: u32,

    rx_shard_counter: u32,
    duplicated_shard_counter: u32,

    highest_rx_frame_index: i32,
    highest_rx_shard_index: i32,
}

impl<H> ReceiverData<H> {
    pub fn had_packet_loss(&self) -> bool {
        self.had_packet_loss
    }
    pub fn get_frame_index(&self) -> u32 {
        self.frame_index
    }
    pub fn get_frame_span(&self) -> f32 {
        self.frame_span
    }
    pub fn get_frame_interarrival(&self) -> f32 {
        self.frame_interarrival
    }
    pub fn get_jitter_avg_frame(&self) -> f32 {
        self.jitter_avg_frame
    }
    pub fn get_filtered_ow_delay(&self) -> f32 {
        self.filtered_ow_delay
    }
    pub fn get_rx_bytes(&self) -> u32 {
        self.rx_bytes
    }
    pub fn get_bytes_in_frame(&self) -> u32 {
        self.bytes_in_frame
    }
    pub fn get_bytes_in_frame_app(&self) -> u32 {
        self.bytes_in_frame_app
    }
    pub fn get_frames_skipped(&self) -> u32 {
        self.frames_skipped
    }
    pub fn get_rx_shard_counter(&self) -> u32 {
        self.rx_shard_counter
    }
    pub fn get_duplicated_shard_counter(&self) -> u32 {
        self.duplicated_shard_counter
    }
    pub fn get_highest_rx_frame_index(&self) -> i32 {
        self.highest_rx_frame_index
    }
    pub fn get_highest_rx_shard_index(&self) -> i32 {
        self.highest_rx_shard_index
    }
}

impl<H: DeserializeOwned> ReceiverData<H> {
    pub fn get(&self) -> Result<(H, &[u8])> {
        let mut data: &[u8] = &self.buffer.as_ref().unwrap()[SHARD_PREFIX_SIZE..self.size];
        // This will partially consume the slice, leaving only the actual payload
        let header = bincode::deserialize_from(&mut data)?;

        Ok((header, data))
    }
    pub fn get_header(&self) -> Result<H> {
        Ok(self.get()?.0)
    }
}

impl<H> Drop for ReceiverData<H> {
    fn drop(&mut self) {
        self.used_buffer_queue
            .send(self.buffer.take().unwrap())
            .ok();
    }
}

struct ReconstructedPacket {
    index: u32,
    buffer: Vec<u8>,
    size: usize, // contains prefix

    frame_index: u32,

    frame_span: f32,
    frame_interarrival: f32,

    jitter_avg_frame: f32,
    filtered_ow_delay: f32,

    rx_bytes: u32,
    bytes_in_frame: u32,
    bytes_in_frame_app: u32,

    rx_shard_counter: u32,
    duplicated_shard_counter: u32,

    highest_rx_frame_index: i32,
    highest_rx_shard_index: i32,
}

pub struct StreamReceiver<H> {
    packet_receiver: mpsc::Receiver<ReconstructedPacket>,
    used_buffer_queue: mpsc::Sender<Vec<u8>>,
    last_packet_index: Option<u32>,
    _phantom: PhantomData<H>,

    frame_interarrival: f32,

    rx_bytes: u32,

    rx_shard_counter: u32,
    duplicated_shard_counter: u32,
}

fn wrapping_cmp(lhs: u32, rhs: u32) -> Ordering {
    let diff = lhs.wrapping_sub(rhs);
    if diff == 0 {
        Ordering::Equal
    } else if diff < u32::MAX / 2 {
        Ordering::Greater
    } else {
        // if diff > u32::MAX / 2, it means the sub operation wrapped
        Ordering::Less
    }
}

/// Get next packet reconstructing from shards.
/// Returns true if a packet has been recontructed and copied into the buffer.
impl<H: DeserializeOwned + Serialize> StreamReceiver<H> {
    pub fn recv(&mut self, timeout: Duration) -> ConResult<ReceiverData<H>> {
        let packet = self
            .packet_receiver
            .recv_timeout(timeout)
            .handle_try_again()?;

        self.frame_interarrival += packet.frame_interarrival;

        self.rx_bytes += packet.rx_bytes;

        self.rx_shard_counter += packet.rx_shard_counter;
        self.duplicated_shard_counter += packet.duplicated_shard_counter;

        let mut had_packet_loss = false;
        let mut frames_skipped: u32 = 0;

        if let Some(last_idx) = self.last_packet_index {
            // Use wrapping arithmetics
            match wrapping_cmp(packet.index, last_idx.wrapping_add(1)) {
                Ordering::Equal => (),
                Ordering::Greater => {
                    // Skipped some indices
                    frames_skipped = packet.index - last_idx.wrapping_add(1);
                    had_packet_loss = true
                }
                Ordering::Less => {
                    // Old packet, discard
                    self.used_buffer_queue.send(packet.buffer).to_con()?;
                    return alvr_common::try_again();
                }
            }
        }
        let interarrival = self.frame_interarrival;
        let rx_bytes_val = self.rx_bytes;
        let rx_counter = self.rx_shard_counter;
        let duplicated_counter = self.duplicated_shard_counter;

        self.frame_interarrival = 0.0;
        self.rx_bytes = 0;
        self.rx_shard_counter = 0;
        self.duplicated_shard_counter = 0;

        self.last_packet_index = Some(packet.index);

        Ok(ReceiverData {
            buffer: Some(packet.buffer),
            size: packet.size,
            used_buffer_queue: self.used_buffer_queue.clone(),
            had_packet_loss,
            _phantom: PhantomData,

            frame_index: packet.frame_index,

            frame_span: packet.frame_span,
            frame_interarrival: interarrival,

            jitter_avg_frame: packet.jitter_avg_frame,
            filtered_ow_delay: packet.filtered_ow_delay,

            rx_bytes: rx_bytes_val,
            bytes_in_frame: packet.bytes_in_frame,
            bytes_in_frame_app: packet.bytes_in_frame_app,

            frames_skipped: frames_skipped,

            rx_shard_counter: rx_counter,
            duplicated_shard_counter: duplicated_counter,

            highest_rx_frame_index: packet.highest_rx_frame_index,
            highest_rx_shard_index: packet.highest_rx_shard_index,
        })
    }
}

pub enum StreamSocketBuilder {
    Tcp(TcpListener),
    Udp(UdpSocket),
}

impl StreamSocketBuilder {
    pub fn listen_for_server(
        timeout: Duration,
        port: u16,
        stream_socket_config: SocketProtocol,
        stream_tos_config: Option<DscpTos>,
        send_buffer_bytes: SocketBufferSize,
        recv_buffer_bytes: SocketBufferSize,
    ) -> Result<Self> {
        Ok(match stream_socket_config {
            SocketProtocol::Udp => StreamSocketBuilder::Udp(udp::bind(
                port,
                stream_tos_config,
                send_buffer_bytes,
                recv_buffer_bytes,
            )?),
            SocketProtocol::Tcp => StreamSocketBuilder::Tcp(tcp::bind(
                timeout,
                port,
                stream_tos_config,
                send_buffer_bytes,
                recv_buffer_bytes,
            )?),
        })
    }

    pub fn accept_from_server(
        self,
        server_ip: IpAddr,
        port: u16,
        max_packet_size: usize,
        timeout: Duration,
    ) -> ConResult<StreamSocket> {
        let protocol: SocketProtocol;
        let (send_socket, receive_socket): (Box<dyn SocketWriter>, Box<dyn SocketReader>) =
            match self {
                StreamSocketBuilder::Udp(socket) => {
                    let (send_socket, receive_socket) =
                        udp::connect(&socket, server_ip, port, timeout).to_con()?;
                    protocol = SocketProtocol::Udp;
                    (Box::new(send_socket), Box::new(receive_socket))
                }
                StreamSocketBuilder::Tcp(listener) => {
                    let (send_socket, receive_socket) =
                        tcp::accept_from_server(&listener, Some(server_ip), timeout)?;
                    protocol = SocketProtocol::Tcp;

                    (Box::new(send_socket), Box::new(receive_socket))
                }
            };

        Ok(StreamSocket {
            // +4 is a workaround to retain compatibilty with old protocol
            // todo: remove +4
            max_packet_size: max_packet_size + 4,
            send_socket: Arc::new(Mutex::new(send_socket)),
            receive_socket,
            shard_recv_state: None,
            stream_recv_components: HashMap::new(),

            transport_protocol: protocol,

            map_rx: HashMap::new(),
            rx_bytes: 0,

            kalman: KalmanFilter::default(),
            last_shard_in_frame_instant: Instant::now(),

            rx_shard_counter: 0,
            duplicated_shard_counter: 0,

            highest_rx_frame_index: -1,
            highest_rx_shard_index: -1,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn connect_to_client(
        timeout: Duration,
        client_ip: IpAddr,
        port: u16,
        protocol: SocketProtocol,
        dscp: Option<DscpTos>,
        send_buffer_bytes: SocketBufferSize,
        recv_buffer_bytes: SocketBufferSize,
        max_packet_size: usize,
    ) -> ConResult<StreamSocket> {
        let (send_socket, receive_socket): (Box<dyn SocketWriter>, Box<dyn SocketReader>) =
            match protocol {
                SocketProtocol::Udp => {
                    let socket =
                        udp::bind(port, dscp, send_buffer_bytes, recv_buffer_bytes).to_con()?;
                    let (send_socket, receive_socket) =
                        udp::connect(&socket, client_ip, port, timeout).to_con()?;

                    (Box::new(send_socket), Box::new(receive_socket))
                }
                SocketProtocol::Tcp => {
                    let (send_socket, receive_socket) = tcp::connect_to_client(
                        timeout,
                        &[client_ip],
                        port,
                        send_buffer_bytes,
                        recv_buffer_bytes,
                    )?;

                    (Box::new(send_socket), Box::new(receive_socket))
                }
            };

        Ok(StreamSocket {
            // +4 is a workaround to retain compatibilty with old protocol
            // todo: remove +4
            max_packet_size: max_packet_size + 4,
            send_socket: Arc::new(Mutex::new(send_socket)),
            receive_socket,
            shard_recv_state: None,
            stream_recv_components: HashMap::new(),

            transport_protocol: protocol,

            map_rx: HashMap::new(),
            rx_bytes: 0,

            kalman: KalmanFilter::default(),
            last_shard_in_frame_instant: Instant::now(),

            rx_shard_counter: 0,
            duplicated_shard_counter: 0,

            highest_rx_frame_index: -1,
            highest_rx_shard_index: -1,
        })
    }
}

struct RecvState {
    shard_length: usize, // contains prefix length itself
    stream_id: u16,
    packet_index: u32,
    shards_count: usize,
    shard_index: usize,
    packet_cursor: usize, // counts also the prefix bytes
    overwritten_data_backup: Option<[u8; SHARD_PREFIX_SIZE]>,
    should_discard: bool,
}

struct InProgressPacket {
    buffer: Vec<u8>,
    buffer_length: usize,
    received_shard_indices: HashSet<usize>,
}

struct StreamRecvComponents {
    used_buffer_sender: mpsc::Sender<Vec<u8>>,
    used_buffer_receiver: mpsc::Receiver<Vec<u8>>,
    packet_queue: mpsc::Sender<ReconstructedPacket>,
    in_progress_packets: HashMap<u32, InProgressPacket>,
    discarded_shards_sink: InProgressPacket,
}

// Note: used buffers don't *have* to be split by stream ID, but doing so improves memory usage
// todo: impose cap on number of created buffers to avoid OOM crashes
pub struct StreamSocket {
    max_packet_size: usize,
    send_socket: Arc<Mutex<Box<dyn SocketWriter>>>,
    receive_socket: Box<dyn SocketReader>,
    shard_recv_state: Option<RecvState>,
    stream_recv_components: HashMap<u16, StreamRecvComponents>,

    transport_protocol: SocketProtocol,

    map_rx: HashMap<u32, HashMap<usize, ShardMapStats>>,
    rx_bytes: u32,

    kalman: KalmanFilter,
    last_shard_in_frame_instant: Instant,

    rx_shard_counter: u32,
    duplicated_shard_counter: u32,

    highest_rx_shard_index: i32,
    highest_rx_frame_index: i32,
}
#[derive(Clone)]
struct ShardMapStats {
    tx_duration: f32,
    rx_instant: Instant,
    rx_bytes: u32,
    rx_bytes_app: u32,
}

impl StreamSocket {
    pub fn request_stream<T>(&self, stream_id: u16) -> StreamSender<T> {
        StreamSender {
            inner: Arc::clone(&self.send_socket),
            stream_id,
            max_packet_size: self.max_packet_size,
            next_packet_index: 0,
            used_buffers: vec![],
            _phantom: PhantomData,
            last_shard_instant: None,
            shards_count: 0,
        }
    }

    // max_concurrent_buffers: number of buffers allocated by this call which will be reused to
    // receive packets for this stream ID. If packets are not read fast enough, the shards received
    // for this particular stream will be discarded
    pub fn subscribe_to_stream<T>(
        &mut self,
        stream_id: u16,
        max_concurrent_buffers: usize,
    ) -> StreamReceiver<T> {
        let (packet_sender, packet_receiver) = mpsc::channel();
        let (used_buffer_sender, used_buffer_receiver) = mpsc::channel();

        for _ in 0..max_concurrent_buffers {
            used_buffer_sender.send(vec![]).ok();
        }

        self.stream_recv_components.insert(
            stream_id,
            StreamRecvComponents {
                used_buffer_sender: used_buffer_sender.clone(),
                used_buffer_receiver,
                packet_queue: packet_sender,
                in_progress_packets: HashMap::new(),
                discarded_shards_sink: InProgressPacket {
                    buffer: vec![],
                    buffer_length: 0,
                    received_shard_indices: HashSet::new(),
                },
            },
        );

        StreamReceiver {
            packet_receiver,
            used_buffer_queue: used_buffer_sender,
            _phantom: PhantomData,
            last_packet_index: None,

            frame_interarrival: 0.,

            rx_bytes: 0,

            rx_shard_counter: 0,
            duplicated_shard_counter: 0,
        }
    }

    pub fn recv(&mut self) -> ConResult {
        let shard_recv_state_mut = if let Some(state) = &mut self.shard_recv_state {
            state
        } else {
            let mut bytes = [0; SHARD_PREFIX_SIZE];
            let count = self.receive_socket.peek(&mut bytes)?;
            if count < SHARD_PREFIX_SIZE {
                return alvr_common::try_again();
            }

            // todo: switch to little endian
            // todo: do not remove sizeof<u32> for packet length
            let shard_length = mem::size_of::<u32>()
                + u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
            let stream_id = u16::from_be_bytes(bytes[4..6].try_into().unwrap());
            let packet_index = u32::from_be_bytes(bytes[6..10].try_into().unwrap());
            let shards_count = u32::from_be_bytes(bytes[10..14].try_into().unwrap()) as usize;
            let shard_index = u32::from_be_bytes(bytes[14..18].try_into().unwrap()) as usize;
            let tx_duration = f32::from_be_bytes(bytes[18..22].try_into().unwrap());

            if stream_id == VIDEO {
                if self.highest_rx_frame_index == packet_index as i32 {
                    if self.highest_rx_shard_index < shard_index as i32 {
                        self.highest_rx_shard_index = shard_index as i32;
                    }
                } else if self.highest_rx_frame_index < packet_index as i32 {
                    self.highest_rx_frame_index = packet_index as i32;
                    self.highest_rx_shard_index = shard_index as i32;
                }

                let header_bytes_transport: u32 = match self.transport_protocol {
                    SocketProtocol::Udp => 42,
                    SocketProtocol::Tcp => 54,
                };
                let packet = ShardMapStats {
                    tx_duration,
                    rx_instant: Instant::now(),
                    rx_bytes: shard_length as u32 + header_bytes_transport,
                    rx_bytes_app: (shard_length - SHARD_PREFIX_SIZE) as u32,
                };
                let shards_map = self.map_rx.entry(packet_index).or_insert(HashMap::new());
                shards_map.insert(shard_index, packet);

                self.rx_bytes += shard_length as u32 + header_bytes_transport;
            }
            self.shard_recv_state.insert(RecvState {
                shard_length,
                stream_id,
                packet_index,
                shards_count,
                shard_index,
                packet_cursor: 0,
                overwritten_data_backup: None,
                should_discard: false,
            })
        };

        let Some(components) = self
            .stream_recv_components
            .get_mut(&shard_recv_state_mut.stream_id)
        else {
            debug!(
                "Received packet from stream {} before subscribing!",
                shard_recv_state_mut.stream_id
            );
            return alvr_common::try_again();
        };

        let in_progress_packet = if shard_recv_state_mut.should_discard {
            &mut components.discarded_shards_sink
        } else if let Some(packet) = components
            .in_progress_packets
            .get_mut(&shard_recv_state_mut.packet_index)
        {
            packet
        } else if let Some(buffer) = components.used_buffer_receiver.try_recv().ok().or_else(|| {
            // By default, try to dequeue a used buffer. In case none were found, recycle one of the
            // in progress packets, chances are these buffers are "dead" because one of their shards
            // has been dropped by the network.
            let idx = *components.in_progress_packets.iter().next()?.0;
            Some(components.in_progress_packets.remove(&idx).unwrap().buffer)
        }) {
            // NB: Can't use entry pattern because we want to allow bailing out on the line above
            components.in_progress_packets.insert(
                shard_recv_state_mut.packet_index,
                InProgressPacket {
                    buffer,
                    buffer_length: 0,
                    // todo: find a way to skipping this allocation
                    received_shard_indices: HashSet::with_capacity(
                        shard_recv_state_mut.shards_count,
                    ),
                },
            );
            components
                .in_progress_packets
                .get_mut(&shard_recv_state_mut.packet_index)
                .unwrap()
        } else {
            // This branch may be hit in case the thread related to the stream hangs for some reason
            shard_recv_state_mut.should_discard = true;
            shard_recv_state_mut.packet_cursor = 0; // reset cursor from old shards
                                                    // always write at the start of the packet so the buffer doesn't grow much
            shard_recv_state_mut.shard_index = 0;

            &mut components.discarded_shards_sink
        };

        if !in_progress_packet
            .received_shard_indices
            .contains(&shard_recv_state_mut.shard_index)
        {
            if shard_recv_state_mut.stream_id == VIDEO {
                self.rx_shard_counter += 1;
            }
        } else {
            if shard_recv_state_mut.stream_id == VIDEO {
                self.duplicated_shard_counter += 1;
            }
        }

        let max_shard_data_size = self.max_packet_size - SHARD_PREFIX_SIZE;
        // Note: there is no prefix offset, since we want to write the prefix too.
        let packet_start_index = shard_recv_state_mut.shard_index * max_shard_data_size;

        // Prepare buffer to accomodate receiving shard
        {
            // Note: this contains the prefix offset
            in_progress_packet.buffer_length = usize::max(
                in_progress_packet.buffer_length,
                packet_start_index + shard_recv_state_mut.shard_length,
            );

            if in_progress_packet.buffer.len() < in_progress_packet.buffer_length {
                in_progress_packet
                    .buffer
                    .resize(in_progress_packet.buffer_length, 0);
            }
        }

        let sub_buffer = &mut in_progress_packet.buffer[packet_start_index..];

        // Read shard into the single contiguous buffer
        {
            // Backup the small section of bytes that will be overwritten by reading from socket.
            if shard_recv_state_mut.overwritten_data_backup.is_none() {
                shard_recv_state_mut.overwritten_data_backup =
                    Some(sub_buffer[..SHARD_PREFIX_SIZE].try_into().unwrap())
            }

            // This loop may bail out at any time if a timeout is reached. This is correctly handled by
            // the previous code.
            while shard_recv_state_mut.packet_cursor < shard_recv_state_mut.shard_length {
                let size = self.receive_socket.recv(
                    &mut sub_buffer
                        [shard_recv_state_mut.packet_cursor..shard_recv_state_mut.shard_length],
                )?;
                shard_recv_state_mut.packet_cursor += size;
            }

            // Restore backed up bytes
            // Safety: overwritten_data_backup is always set just before receiving the packet
            sub_buffer[..SHARD_PREFIX_SIZE]
                .copy_from_slice(&shard_recv_state_mut.overwritten_data_backup.take().unwrap());
        }

        if !shard_recv_state_mut.should_discard {
            if !in_progress_packet
                .received_shard_indices
                .contains(&shard_recv_state_mut.shard_index)
            {
                in_progress_packet
                    .received_shard_indices
                    .insert(shard_recv_state_mut.shard_index);
            }
        }

        let mut jitter_avg = 0.0;

        let mut frame_span = 0.0;
        let mut frame_interarrival: f32 = 0.0;

        let mut all_bytes_in_frame: u32 = 0;
        let mut all_bytes_in_frame_app: u32 = 0;

        // Check if packet is complete and send
        if in_progress_packet.received_shard_indices.len() == shard_recv_state_mut.shards_count {
            if shard_recv_state_mut.stream_id == VIDEO {
                if let Some(inner_map) = self.map_rx.remove(&shard_recv_state_mut.packet_index) {
                    let values: Vec<&ShardMapStats> = inner_map.values().collect();
                    let min_time = values.iter().map(|shard| shard.rx_instant).min().unwrap();
                    let max_time = values.iter().map(|shard| shard.rx_instant).max().unwrap();

                    frame_span = max_time.saturating_duration_since(min_time).as_secs_f32();
                    frame_interarrival = max_time
                        .saturating_duration_since(self.last_shard_in_frame_instant)
                        .as_secs_f32();

                    self.last_shard_in_frame_instant = max_time;

                    all_bytes_in_frame = values.iter().map(|shard| shard.rx_bytes).sum();
                    all_bytes_in_frame_app = values.iter().map(|shard| shard.rx_bytes_app).sum();

                    // Jitter computation
                    {
                        let mut sorted_keys: Vec<usize> = inner_map.keys().copied().collect();
                        sorted_keys.sort_unstable_by_key(|&k| Reverse(k));

                        let ordered_values: Vec<&ShardMapStats> = sorted_keys
                            .iter()
                            .filter_map(|&key| inner_map.get(&key))
                            .collect();

                        let mut prev_instant_rx = Instant::now();
                        let mut index = 0;
                        let mut time_duration_rx;

                        for shards_stats in ordered_values {
                            if index == 0 {
                                prev_instant_rx = shards_stats.rx_instant;
                            } else {
                                time_duration_rx =
                                    shards_stats.rx_instant.duration_since(prev_instant_rx);
                                // delay_gradient += shards_stats.tx_duration - time_duration_rx.as_secs_f32() ;
                                self.kalman.ow_delay =
                                    shards_stats.tx_duration - time_duration_rx.as_secs_f32();
                                self.kalman.k_gain = (self.kalman.p_prev + Q_KALMAN)
                                    / (self.kalman.p_prev
                                        + Q_KALMAN
                                        + self.kalman.noise_estimation);
                                self.kalman.m_current = (1.0 - self.kalman.k_gain)
                                    * self.kalman.m_prev
                                    + self.kalman.k_gain * self.kalman.ow_delay;

                                self.kalman.residual_z = self.kalman.ow_delay - self.kalman.m_prev;
                                self.kalman.noise_estimation = (0.95 * self.kalman.noise_prev)
                                    + self.kalman.residual_z.powf(2.0) * 0.05;
                                self.kalman.p_current =
                                    (1.0 - self.kalman.k_gain) * (self.kalman.p_prev + Q_KALMAN);
                                self.kalman.abs_m = self.kalman.m_current.abs(); // absolute value in rust? ;

                                self.kalman.p_prev = self.kalman.p_current;
                                self.kalman.m_prev = self.kalman.m_current;
                                self.kalman.noise_prev = self.kalman.noise_estimation;

                                self.kalman.measured_delay += self.kalman.m_current;

                                prev_instant_rx = shards_stats.rx_instant;
                            }
                            index += 1;
                        }
                        jitter_avg =
                            self.kalman.ow_delay / shard_recv_state_mut.shards_count as f32;
                    }
                }
            }
            let size = in_progress_packet.buffer_length;
            components
                .packet_queue
                .send(ReconstructedPacket {
                    index: shard_recv_state_mut.packet_index,
                    buffer: components
                        .in_progress_packets
                        .remove(&shard_recv_state_mut.packet_index)
                        .unwrap()
                        .buffer,
                    size,

                    frame_index: shard_recv_state_mut.packet_index,

                    frame_span: frame_span,
                    frame_interarrival: frame_interarrival,

                    jitter_avg_frame: jitter_avg,
                    filtered_ow_delay: self.kalman.m_current,

                    rx_bytes: self.rx_bytes,
                    bytes_in_frame: all_bytes_in_frame,
                    bytes_in_frame_app: all_bytes_in_frame_app,

                    rx_shard_counter: self.rx_shard_counter,
                    duplicated_shard_counter: self.duplicated_shard_counter,

                    highest_rx_frame_index: self.highest_rx_frame_index,
                    highest_rx_shard_index: self.highest_rx_shard_index,
                })
                .ok();
            if shard_recv_state_mut.stream_id == VIDEO {
                self.rx_bytes = 0;
                self.rx_shard_counter = 0;
                self.duplicated_shard_counter = 0;

                // Keep only shards with later packet index (using wrapping logic)
                for idx in components
                    .in_progress_packets
                    .iter()
                    .filter_map(|(idx, _)| {
                        if wrapping_cmp(*idx, shard_recv_state_mut.packet_index) == Ordering::Less {
                            Some(*idx)
                        } else {
                            None
                        }
                    })
                {
                    // Remove the stored data from previous frame indexes. Those frames won't be reconstructed given the while below...
                    self.map_rx.remove(&idx);
                }
            }

            // Keep only shards with later packet index (using wrapping logic)
            while let Some((idx, _)) = components.in_progress_packets.iter().find(|(idx, _)| {
                wrapping_cmp(**idx, shard_recv_state_mut.packet_index) == Ordering::Less
            }) {
                let idx = *idx; // fix borrow rule
                let packet = components.in_progress_packets.remove(&idx).unwrap();

                // Recycle buffer
                components.used_buffer_sender.send(packet.buffer).ok();
            }
        }

        // Mark current shard as read and allow for a new shard to be read
        self.shard_recv_state = None;

        Ok(())
    }
}
