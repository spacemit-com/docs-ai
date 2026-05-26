---
sidebar_position: 3
---

# Seewise

**Seewise** is an intelligent video search engine that supports both local video uploads and RTSP camera streams. It automatically analyzes video content and enables users to locate specific clips quickly through natural-language search.

## Key Features

- **Intelligent Video Understanding**: Automatically analyzes video frames and generates natural-language descriptions for each frame.
- **Multiple Search Modes**: Supports semantic search, keyword search, and hybrid search.
- **Flexible Video Sources**: Works with both uploaded local videos and live RTSP streams.
- **Real-time Processing Feedback**: Displays processing progress directly in the user interface.
- **Bilingual Support**: Provides both Chinese and English interfaces, with multilingual search prompts.

## Platform Support

| Platform & OS              | Supported          |
|----------------------------|--------------------|
| K1 Buildroot               | ❌ Not supported   |
| K1 OpenHarmony             | ❌ Not supported   |
| K1 Bianbu LXQT/GNOME       | ❌ Not supported   |
| K3 Buildroot               | ❌ Not supported   |
| K3 OpenHarmony             | ❌ Not supported   |
| K3 Bianbu LXQT/GNOME       | ✅ Supported       |

## Technical Architecture

Seewise uses a client-server architecture with three layers:

- **Frontend Layer**: A modern web interface for video upload, search, and real-time progress monitoring.
- **Application Service Layer**: Handles business logic for video processing, search, and data management.
- **AI Model Layer**: Performs image analysis, text vectorization, and search result reranking.

### System Architecture Diagram

![](../static/seewise-1.png)

## Installation and Deployment

### Install the Debian Package

For production environments, installing the local package with `apt` is recommended:

```bash
sudo apt update
sudo apt install seewise-2
```

If dependency issues occur, run:

```bash
sudo apt-get install -f
```

![](../static/seewise-7.png)

After installation, `seewise-2.service` is created and enabled automatically.

- Default root directory: `~/.seewise-2`
- Runtime dependencies include `llama.cpp-tools-spacemit`, `spacemit-onnxruntime`, and `python3-spacemit-ort`

### Quick Start

- **Web access**: After installation, access the service in a browser at `http://<board-ip>:8084`
- **Desktop access**: Open the application menu in the lower-left corner, search for **seewise**, and click the icon to open the corresponding web page

![](../static/seewise-2.png)

## Model Download and Parameter Configuration

### Download Models

For production use, download models from the **Settings** page in the Seewise web UI.

Default model directories:

- **VLM**: `~/.seewise-2/models/vlm/fastvlm-mm-0.5b-q4_1/`
- **Embedding**: `~/.seewise-2/models/embedding/`
- **Rerank**: `~/.seewise-2/models/rerank/`

Use the recommended models whenever possible.

![](../static/seewise-5.png)

### Configure Parameters

The main configurable settings are frame extraction and retrieval parameters:

- **Frame extraction settings**: Define the frame sampling interval, which determines how often a frame is extracted from the video.
- **Retrieval settings**: Control the search strategy, including whether reranking is enabled.

![](../static/seewise-6.png)

## Video Upload and RTSP Retrieval Workflow

### Local Video Upload Workflow

1. Select **Upload Video** on the frontend page.
2. The backend saves the file to `data/videos/`.
3. FFmpeg extracts frames and generates output in `data/frames/` and `data/thumbnails/`.
4. The VLM generates frame-level semantic descriptions.
5. The embedding service generates vectors and writes them to the vector index.
6. The video is now indexed and ready to search.

### RTSP Live Stream Workflow

1. Enter the RTSP address in the frontend and start the connection.
2. The backend uses `RTSPCapture` to record the stream and package it as an MP4 file.
3. FFmpeg then extracts frames and generates thumbnails.
4. Processing progress is pushed to the frontend in real time over WebSocket.
5. When processing is complete, frame data is written to both the database and the vector index.

![](../static/seewise-3.png)

### Search Workflow

1. Upload a video or connect to an RTSP stream and wait for processing to complete.
2. While processing, a progress indicator appears in the lower-right corner of the video player and keyframe previews populate below the search box.
3. Enter a natural-language query in the search box and press **Enter**.

> For best results, use the retrieval configuration options available on the Settings page.

![](../static/seewise-4.jpg)

## Logs and Cache Management

### Log Locations

- **Backend log**: `~/.seewise-2/logs/backend.log`
- **Frontend log**: `~/.seewise-2/logs/frontend.log`
- **Model logs**: `~/.seewise-2/logs/vlm.log`, `~/.seewise-2/logs/embedding.log`, `~/.seewise-2/logs/rerank.log`
- **systemd log streaming**: `journalctl -u seewise-2.service -f`

### Cache and Data Directories

- **Video files**: `~/.seewise-2/data/videos/`
- **Extracted frames**: `~/.seewise-2/data/frames/`
- **Thumbnails**: `~/.seewise-2/data/thumbnails/`
- **SQLite database**: `~/.seewise-2/data/video_search.db`
- **Vector index**: `~/.seewise-2/data/vector_store.db`

To clear cached data:

1. Stop the service:
   ```bash
   sudo systemctl stop seewise-2.service
   ```
2. Back up or remove the cache directories:
   ```bash
   rm -rf ~/.seewise-2/data/videos/*
   rm -rf ~/.seewise-2/data/frames/*
   rm -rf ~/.seewise-2/data/thumbnails/*
   rm -f ~/.seewise-2/data/video_search.db
   rm -f ~/.seewise-2/data/vector_store.db
   ```
3. To download the models again, delete the model directory and repeat the model download process.

## FAQ

### Q: Why does the service fail to start after installation?

A: First check `systemctl status seewise-2.service`. If the issue is dependency-related, confirm that `llama.cpp-tools-spacemit`, `spacemit-onnxruntime`, and `python3-spacemit-ort` are installed.

### Q: Why can’t the models be downloaded, or why is the model service not starting?

A: Check the following logs to identify the cause:

- `~/.seewise-2/logs/vlm_8071.log`
- `~/.seewise-2/logs/embedding_8072.log`
- `~/.seewise-2/logs/rerank_8073.log`

### Q: Why does the RTSP connection fail?

A: Verify the RTSP URL, network connectivity, and camera status. If RTSP recording fails, review the backend log and check whether the required port is already in use.

### Q: Why does video upload fail, or why is processing slow?

A: The maximum size for a single file is 500 MB. Large videos may require more processing time. Also verify that there is enough disk space available in `data/videos/` and `data/frames/`.

### Q: Why are the search results inaccurate?

A: Try switching the retrieval mode between semantic, keyword, and hybrid search, and enable reranking if needed. If the issue appears to be model-related, re-download the models and restart the model services.

### Q: How do I clear the cache and start over?

A: Stop the service first, then delete `~/.seewise-2/data/video_search.db`, `~/.seewise-2/data/vector_store.db`, `data/frames/`, `data/thumbnails/`, and `~/.seewise-2/data/videos/`.