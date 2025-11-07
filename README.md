# AIVISION Privacy - De-Identification MVP

A comprehensive privacy protection system that provides real-time de-identification of sensitive information in video streams, including human faces and license plates. The system supports multiple input sources including RTSP IP cameras, uploaded videos, and webcam feeds.

## Overview

This project implements an advanced video privacy protection system that can:
- **Face De-identification**: Swap detected faces with anonymized source faces
- **License Plate De-identification**: Replace detected license plates with anonymized templates
- **Real-time Streaming**: Process and stream video in real-time with low latency
- **Multiple Input Sources**: Support for RTSP cameras, video files, and webcam feeds

## Key Features

- 🎥 **Real-time Video Processing**: Stream and process video frames in real-time
- 🔒 **Privacy Protection**: Automatically detect and anonymize faces and license plates
- 🌐 **WebRTC Streaming**: Low-latency video streaming to web browsers using WebRTC
- 📹 **RTSP Camera Support**: Direct integration with IP cameras via RTSP protocol
- 🎨 **Modern Web Interface**: User-friendly Gradio-based web interface
- ⚡ **GPU Acceleration**: Supports CUDA for fast processing

## Project Structure

```
aivision_privacy/
├── gradio_video_privacy_rtsp_cam.py    # RTSP camera streaming application
├── gradio_video_privacy_batch_multiusers.py  # Batch processing for multiple users
├── gradio_video_swapface_stream.py     # Face swapping streaming
├── gradio_image_swapper.py             # Image-based face swapping
├── modules/
│   ├── face_swapper.py                 # Face detection and swapping logic
│   ├── lp_swapper.py                   # License plate detection and swapping
│   ├── face_restoration.py            # Face restoration utilities
│   └── utils.py                        # Utility functions
├── models/                             # AI model files
│   ├── swapface/                      # Face swapping models
│   └── lp_detect/                     # License plate detection models
└── data/                               # Source faces and license plates
```

## RTSP Camera Streaming Application

### `gradio_video_privacy_rtsp_cam.py`

This is the main application for streaming video from RTSP IP cameras with real-time privacy protection. The application provides a web interface where users can:

1. **Connect to RTSP Cameras**: Input RTSP camera addresses (e.g., `rtsp://172.19.192.1:554/live`)
2. **Configure Settings**: Adjust frame size, FPS, and select objects to protect (faces, license plates, or both)
3. **Real-time Processing**: View both the original and privacy-protected video streams side-by-side
4. **Start/Stop Streaming**: Control the video stream with a simple button interface

### How WebRTC Streaming Works

The application leverages **Gradio's built-in WebRTC capabilities** to stream video from RTSP cameras to web browsers with minimal latency. Here's how it works:

#### 1. **RTSP Stream Acquisition**
```python
input_cap = cv2.VideoCapture(self.stream_video_source[session_id])
```
- The application uses OpenCV's `VideoCapture` to connect to the RTSP camera stream
- RTSP (Real-Time Streaming Protocol) is a network protocol used by IP cameras to deliver video streams
- The RTSP URL format: `rtsp://[IP_ADDRESS]:[PORT]/[STREAM_PATH]`

#### 2. **Frame Processing Pipeline**
```python
while self.is_input_streaming[session_id]:
    ret, frame = input_cap.read()
    # Resize frame
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    # Process frame (license plate swapping)
    temp_frame = swap_lp_image(self.lp_segment_engine, resized_frame, self.source_lp_dict)
    # Write to video segment
    segment_writer.write(temp_frame)
```

The processing pipeline:
- Captures frames from the RTSP stream
- Resizes frames to the configured dimensions
- Applies privacy protection (license plate/face swapping)
- Writes processed frames to video segments

#### 3. **WebRTC Streaming via Gradio**

Gradio's `gr.Video` component with `streaming=True` automatically handles WebRTC conversion:

```python
input_display = gr.Video(label="Input Video", streaming=True, autoplay=False)
output_display = gr.Video(label="Privacy Protected", streaming=True, autoplay=False)
```

**How WebRTC Works in This Context:**

1. **Segment-based Streaming**: The application creates short video segments (2 seconds worth of frames) and yields them to Gradio
   ```python
   stream_batch_len = 2 * desired_fps  # 2 seconds of video
   yield segment_name  # Gradio receives the segment
   ```

2. **Gradio's WebRTC Bridge**: Gradio automatically:
   - Converts the video segments to WebRTC-compatible format
   - Establishes a WebRTC peer connection with the browser
   - Streams the video using SRTP (Secure Real-time Transport Protocol)
   - Handles adaptive bitrate and network conditions

3. **Browser Playback**: The browser receives the WebRTC stream and plays it using native HTML5 video capabilities

#### 4. **Low-Latency Architecture**

The application implements several optimizations for low-latency streaming:

- **Batch Processing**: Frames are processed in batches (2 seconds) to balance latency and efficiency
- **Circular Buffer**: Old segments are automatically deleted to manage memory:
  ```python
  if len(stream_buffer_mp4) == self.stream_bufer_length:
      os.remove(stream_buffer_mp4[0])  # Remove oldest segment
  ```
- **Configurable FPS**: Users can adjust the frame rate to balance quality and performance
- **Frame Subsampling**: Optional frame skipping for lower computational load

#### 5. **Session Management**

The application supports multiple concurrent sessions:
- Each user session has its own video source, settings, and processing state
- Session-specific dictionaries track:
  - Video source (RTSP URL or file path)
  - Frame size and FPS settings
  - Selected objects to protect
  - Streaming state

### Technical Details

#### Video Segment Format
- **Codec**: MP4V (MPEG-4 Part 2)
- **Container**: MP4
- **Segment Duration**: 2 seconds (configurable via `stream_batch_len`)
- **Storage**: Temporary files in `output/stream_output/` directory

#### WebRTC Benefits
- **Low Latency**: Typically 100-200ms end-to-end delay
- **Browser Native**: No plugins required
- **Secure**: Uses DTLS and SRTP for encryption
- **Adaptive**: Automatically adjusts to network conditions
- **Peer-to-Peer**: Direct connection between server and browser

## Installation

### Prerequisites

- Python 3.10
- CUDA-capable GPU (recommended) or CPU
- FFmpeg (for video processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aivision_privacy
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yaml
   conda activate Faceswap_image
   ```

3. **Install additional dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models**
   - Place face swapping models in `models/swapface/`
   - Place license plate detection model in `models/lp_detect/lp_detect.pt`
   - Ensure source faces are in `data/source_faces/`
   - Ensure source license plates are in `data/source_lp/`

5. **Create output directory**
   ```bash
   mkdir -p output/stream_output
   ```

## Usage

### Running the RTSP Camera Application

```bash
python gradio_video_privacy_rtsp_cam.py
```

The application will launch a web interface (typically at `http://127.0.0.1:7860`).

### Using the Web Interface

1. **Configure Source**:
   - Select "RTSP" from the Source radio buttons
   - Enter your RTSP camera address (e.g., `rtsp://172.19.192.1:554/live`)
   - Select objects to protect: "Human Face", "License Plate", or "Face + LP"

2. **Adjust Settings**:
   - Set Frame Height (default: 600 pixels)
   - Set FPS (default: 30)
   - Click "Apply" to save settings

3. **Start Streaming**:
   - Click "Start Streaming" button
   - The input video will appear in the left panel
   - Click the swap button (middle) to start processing
   - The privacy-protected video will appear in the right panel

4. **Stop Streaming**:
   - Click "Stop Streaming" to halt the stream

### RTSP Camera URL Format

The RTSP URL format varies by camera manufacturer:

- **Generic**: `rtsp://[username]:[password]@[IP_ADDRESS]:[PORT]/[STREAM_PATH]`
- **Example**: `rtsp://admin:password123@192.168.1.100:554/stream1`
- **No Auth**: `rtsp://192.168.1.100:554/live`

Common RTSP paths:
- `/live` - Live stream
- `/stream1` - Primary stream
- `/h264` - H.264 encoded stream
- `/cam/realmonitor` - Some camera models

## Architecture

### Processing Flow

```
RTSP Camera → OpenCV VideoCapture → Frame Processing → Privacy Protection
                                                           ↓
Web Browser ← WebRTC Stream ← Gradio Video Component ← Video Segments
```

### Key Components

1. **Video Capture Layer**: OpenCV handles RTSP stream acquisition
2. **Processing Layer**: YOLO for license plate detection, InsightFace for face detection
3. **Privacy Protection Layer**: Face/plate swapping algorithms
4. **Streaming Layer**: Gradio's WebRTC implementation
5. **Web Interface Layer**: Gradio Blocks UI

## Configuration

### Adjustable Parameters

In `gradio_video_privacy_rtsp_cam.py`:

- `SUBSAMPLE`: Frame subsampling rate (default: 1, no subsampling)
- `stream_bufer_length`: Maximum number of segments in buffer (default: 1800)
- `streaming_period`: Maximum streaming duration in seconds (default: 3600 = 1 hour)
- `stream_batch_len`: Frames per segment (default: 2 * fps)

### Model Paths

- License Plate Detection: `models/lp_detect/lp_detect.pt`
- Face Swapping: `models/swapface/aivision_swapface_v1.onnx`
- Source License Plates: `data/source_lp/`
- Source Faces: `data/source_faces/`

## Performance Considerations

- **GPU Acceleration**: Ensure CUDA is properly configured for faster processing
- **Network Bandwidth**: RTSP streams require stable network connection
- **Frame Rate**: Lower FPS reduces computational load but may affect smoothness
- **Resolution**: Lower frame height reduces processing time
- **Memory Management**: Old segments are automatically cleaned up to prevent memory issues

## Troubleshooting

### RTSP Connection Issues

- Verify camera IP address and port
- Check network connectivity
- Ensure RTSP is enabled on the camera
- Try different RTSP URL formats
- Check firewall settings

### Streaming Issues

- Ensure `output/stream_output/` directory exists and is writable
- Check available disk space (segments are temporarily stored)
- Verify GPU/CUDA setup if using GPU acceleration
- Monitor system resources (CPU, memory, GPU)

### WebRTC Issues

- Use modern browsers (Chrome, Firefox, Edge)
- Check browser console for WebRTC errors
- Ensure HTTPS or localhost (WebRTC requires secure context)
- Check network firewall for WebRTC ports

## Future Enhancements

- [ ] Webcam support (currently in development)
- [ ] Multiple camera support
- [ ] Full body de-identification
- [ ] Custom privacy templates
- [ ] Recording functionality
- [ ] Advanced analytics

## License

[Add your license information here]

## Contact

[Add contact information here]

---

**Note**: This is an MVP (Minimum Viable Product) demonstration. For production use, additional security, error handling, and optimization may be required.
