sidebar_position: 12

# 多路视频分析（YOLO Demo）

**多路视频分析（YOLO Demo）** 是一个 SpacemiT K3 RISC-V 平台的本地实时视频分析应用。应用可同时处理 4、9 或 12 路视频，利用 K3 的 VPU、GPU 和 AI 推理能力完成视频解码、图像前处理、YOLO 推理与结果展示。所有推理数据均在设备本地处理，适用于智慧园区、交通监控、零售分析和工业巡检等边缘视觉场景。

## 产品特点

- **多路并行分析**：支持 4、9、12 路视频网格显示和逐通道统计
- **硬件加速**：使用 MPP/VPU 解码、GPU OpenCL 前处理和 SpaceMIT EP 推理
- **低复制流水线**：OpenCL 直接导入 NV12 dma-buf，并通过预分配 tensor 槽向 ORT 交付数据
- **过载保护**：每通道采用 latest-only 队列和跨通道轮询，负载较高时优先处理新帧
- **多模型支持**：支持目标检测、人脸、手势、火焰、姿态估计和实例分割模型
- **实时状态监控**：显示各通道 FPS、检测数和端到端延迟
- **本地运行**：模型和推理结果无需上传云端

## 平台支持

| 平台 & 系统 | 支持情况 |
| --- | --- |
| K1 Buildroot | ❌ 不支持 |
| K1 OpenHarmony | ❌ 不支持 |
| K1 Bianbu LXQt/GNOME | ❌ 不支持 |
| K3 Buildroot | ❌ 不支持 |
| K3 OpenHarmony | ❌ 不支持 |
| K3 Bianbu LXQt/GNOME | ✅ 支持 |

## 技术架构

### 核心技术栈

- **视频解码**
  - SpacemiT MPP/VPU
  - 支持 H.264、H.265、MJPEG 视频
  - 解码输出为 640×360 NV12 dma-buf
- **图像前处理**
  - GPU OpenCL
  - 完成 NV12 转 RGB/BGR、letterbox、归一化和 NCHW 排布
- **AI 推理**
  - ONNX Runtime
  - SpaceMIT Execution Provider
  - 默认创建 8 个并行 ORT Session
- **界面显示**
  - Qt5 + OpenGL
  - 4、9、12 路网格和逐通道运行统计

### 系统架构图

![多路视频分析系统架构图](../static/multi_stream_vision/system-architecture.png)

系统以 `PipelineManager` 为中心管理多个通道实例。每个通道共享 K3 的 VPU、GPU、AI 计算资源和 DDR 缓冲区，但拥有独立的视频处理状态；界面层通过 Qt 信号与槽接收视频帧、分析结果和性能统计。

### 工作流程

![多路视频分析工作流程图](../static/multi_stream_vision/workflow.png)

> **图例**：实线表示当前仓库已实现的数据通路；虚线表示尚未支持的网络视频流接入路径。网络流接入完成后，同样进入 MPP/VPU 进行硬件解码。

应用为每个通道保留最新帧，并以轮询方式把任务分配给多个推理 Session。当系统负载较高时，会优先处理新帧，避免队列持续堆积导致画面延迟不断增加。

### 性能数据

本次数据为 12 路实测结果，使用1920×1080 H.264 YUV420P 的视频文件作为输入源：

| 模型 | VPU+GPU 带宽 | DDR 总带宽 | 平均每路帧率 | 总帧率 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 2438.97 MB/s | 21716.34 MB/s | 11.19 FPS | 134.29 FPS |
| YOLOv8s | 2137.83 MB/s | 20712.64 MB/s | 6.48 FPS | 77.74 FPS |
| YOLOv8m | 1947.73 MB/s | 18988.03 MB/s | 3.27 FPS | 39.23 FPS |
| YOLOv11n | 2289.96 MB/s | 19085.08 MB/s | 8.48 FPS | 101.81 FPS |
| YOLOv11s | 2083.27 MB/s | 19678.14 MB/s | 5.28 FPS | 63.38 FPS |
| YOLOv12n | 2077.22 MB/s | 18389.45 MB/s | 5.06 FPS | 60.76 FPS |
| YOLO26n | 2304.99 MB/s | 18774.13 MB/s | 8.68 FPS | 104.18 FPS |
| YOLOv5n | 2445.23 MB/s | 20476.51 MB/s | 11.28 FPS | 135.35 FPS |
| YOLOv5s | 2171.69 MB/s | 20613.31 MB/s | 6.87 FPS | 82.46 FPS |
| YOLOv5n-face  | 2033.63 MB/s | 21465.51 MB/s | 4.83 FPS | 57.90 FPS |
| YOLOv5-gesture | 2262.16 MB/s | 18901.93 MB/s | 8.11 FPS | 97.33 FPS |
| YOLOv8-fire | 2129.87 MB/s | 19231.10 MB/s | 6.01 FPS | 72.06 FPS |
| YOLOv8n-seg | 2245.16 MB/s | 19178.61 MB/s | 8.21 FPS | 98.56 FPS |
| YOLOv8s-seg | 2053.98 MB/s | 19468.00 MB/s | 4.86 FPS | 58.29 FPS |
| YOLOv8n-pose | 2308.58 MB/s | 21477.32 MB/s | 9.32 FPS | 111.78 FPS |
| YOLOv8s-pose | 2106.10 MB/s | 21119.24 MB/s | 5.80 FPS | 69.59 FPS |

## 安装

### 方式一：apt 安装（推荐）

在 K3 Bianbu 桌面系统中打开终端，执行：

```bash
sudo apt update
sudo apt install yolo-demo
```

安装完成后，程序、默认配置、模型和示例视频位于以下目录：

```text
/usr/bin/yolo-demo
/usr/share/yolo-demo/configs/default.yaml
/usr/share/yolo-demo/
```

### 方式二：源码安装（开发者）

下载源码：

```bash
# ssh 下载，需要配置 ssh 密钥
git clone git@github.com:spacemit-com/multi-stream-vision.git
# http 下载
git clone https://github.com/spacemit-com/multi-stream-vision.git
```

安装依赖：

```bash
sudo apt install -y \
  build-essential cmake \
  qtbase5-dev libqt5opengl5-dev \
  libopencv-dev libyaml-cpp-dev \
  opencl-headers ocl-icd-opencl-dev \
  libgles-dev mpp2 \
  spacemit-onnxruntime
```

进入源码目录后执行：

```bash
cd /root/release/yolo-demo
cmake -S . -B build
cmake --build build -j8
```

编译完成后，可执行文件位于：

```text
/root/release/yolo-demo/build/yolo-demo
```

## 快速开始

### 1. 启动应用

点击左下角应用菜单，搜索 **YOLO Demo**，然后点击应用图标。应用窗口打开后会自动启动视频分析，无需再点击启动按钮。

如果设备刚运行过其他 AI 推理任务，建议先确认这些任务已经退出，再清理遗留的 TCM 占用：

```bash
spacemit-tcm-smi -c
```

> ⚠️ **重要提示**：不要在 `yolo-demo` 或其他推理程序运行时清理 TCM。

也可以在终端中直接运行：

```bash
yolo-demo
```

### 2. 查看分析结果

默认界面左侧显示多路视频和检测结果，右侧包含控制面板与运行统计：

- **通道**：视频通道编号
- **FPS**：当前通道完成后处理并回调结果的帧率
- **检测数**：当前帧保留的检测目标数量
- **延迟（ms）**：当前帧在解码队列、前处理、推理和后处理各阶段的累计耗时

![](../static/multi_stream_vision/mul-stream-video.png)

> 💡 **提示**：统计值会随模型、视频内容、通道数、显示模式、系统负载和散热状态变化，截图中的数据仅表示当次实机运行状态。

## 功能使用

### 切换通道数

在右侧 **路数配置** 中选择需要的布局：

- **4 路（2×2）**
- **9 路（3×3）**
- **12 路（4×3）**

切换路数后，应用会自动停止当前流水线，并按新布局重新启动。

### 切换分析模型

在右侧 **模型选择** 下拉框中选择模型。模型切换时，应用会重新创建推理流水线并加载对应模型。

| 能力 | 可选模型 |
| --- | --- |
| 通用目标检测 | YOLOv5 Nano/Small、YOLOv8 Nano/Small/Medium、YOLOv11 Nano/Small、YOLOv12 Nano、YOLO26 Nano |
| 人脸检测 | YOLOv5n Face |
| 手势识别 | YOLOv5 Gesture |
| 火焰检测 | YOLOv8 Fire |
| 姿态估计 | YOLOv8 Nano Pose、YOLOv8 Small Pose |
| 实例分割 | YOLOv8 Nano Seg、YOLOv8 Small Seg |

应用内置三个模型：yolov8m.q.onnx、yolov8n.q.onnx、yolov8s.q.onnx，即 YOLOv8 Nano/Small/Medium。

其他模型可以从`https://archive.spacemit.com/spacemit-ai/model_zoo/vision/`获取或通过下载脚本下载到指定目录：

```bash
sudo ./scripts/download_models.sh /usr/share/yolo-demo
```

> ⚠️ **重要提示**：切换模型前应确认对应 ONNX 模型已经安装在 `/usr/share/yolo-demo` 下。较大的模型通常具有更高精度，但会降低多路总帧率。

### 使用纯统计模式

取消勾选 **启用视频显示** 后，应用将隐藏视频网格，仅显示控制面板和运行统计。推理流水线会继续运行，但不再构造显示帧或上传 OpenGL 纹理，可减少 GPU 和 DDR 带宽占用。

![](../static/multi_stream_vision/mul-stream-stats.png)

需要恢复视频时，再次勾选 **启用视频显示**。

## 命令行使用

### 指定配置、路数和模型

```bash
yolo-demo \
  -c /usr/share/yolo-demo/configs/default.yaml \
  -l 12 \
  -m yolov8n
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-c, --config <path>` | 指定 YAML 配置文件 |
| `-l, --layout <num>` | 指定 4、9 或 12 路布局 |
| `-m, --model <id>` | 指定模型 ID |
| `-h, --help` | 查看帮助 |
| `-v, --version` | 查看版本 |

通过 SSH 在已有 Wayland 桌面会话中启动时，可使用：

```bash
QT_QPA_PLATFORM=wayland \
QT_WAYLAND_SHELL_INTEGRATION=xdg-shell \
WAYLAND_DISPLAY=wayland-0 \
XDG_RUNTIME_DIR=/run/user/1000 \
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
yolo-demo -c configs/default.yaml
```

如果桌面用户 UID 不是 1000，需要同步修改 `XDG_RUNTIME_DIR` 和 DBus 地址。

## 高级设置

默认配置文件为：

```text
/usr/share/yolo-demo/configs/default.yaml
```

建议先复制配置文件，再修改副本：

```bash
cp /usr/share/yolo-demo/configs/default.yaml ~/yolo-demo.yaml
yolo-demo -c ~/yolo-demo.yaml
```

### 视频源配置

```yaml
streams:
  layout: 12
  sources:
    - /usr/share/yolo-demo/1.mp4
    - /usr/share/yolo-demo/2.mp4
```

本地视频播放结束后会自动循环。`sources` 数量应不少于所选布局的通道数；同一个文件可以在多个通道中复用。

### 推理配置

```yaml
model:
  id: yolov8n
  conf_threshold: 0.25
  iou_threshold: 0.45

inference:
  num_sessions: 8
  intra_threads: 1
  infer_fps: 25
```

- **conf_threshold**：置信度阈值；调高可减少低置信度结果
- **iou_threshold**：NMS 的 IoU 阈值
- **num_sessions**：并行 ORT Session 数量
- **intra_threads**：每个 Session 使用的 SpaceMIT EP 线程数
- **infer_fps**：算法目标帧率；设为 0 时每个解码帧都进入算法通路

SpaceMIT EP 会为推理线程申请 TCM，当前配置应满足：

```text
num_sessions × intra_threads <= 8
```

默认推荐使用 8 个 Session、每个 Session 1 个线程。

### 前处理配置

```yaml
preprocess:
  image_size: [640, 640]
  pixel_format: RGB
  normalize: true
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
  letterbox: true
  letterbox_color: [114, 114, 114]
```

- **image_size**：OpenCL 输出和 ONNX 模型输入尺寸
- **pixel_format**：模型需要的 `RGB` 或 `BGR` 通道顺序
- **normalize**：是否执行归一化
- **mean/std**：三通道归一化参数
- **letterbox**：是否保持输入图像宽高比
- **letterbox_color**：letterbox padding 颜色

### 显示与性能配置

```yaml
display:
  mode: video
  draw_detections: true

performance:
  decode_fps: 25
  display_fps: 25
  stats_interval_ms: 1000
```

- **display.mode**：`video` 显示视频；`stats` 仅显示统计
- **draw_detections**：是否在视频上叠加检测结果
- **decode_fps**：本地文件解码节奏
- **display_fps**：视频显示刷新上限
- **stats_interval_ms**：统计日志刷新周期

### 当前保留但未生效的配置

以下字段可能仍出现在 YAML 中，但当前执行路径尚未使用：

- `inference.num_threads`
- `inference.providers`：当前始终尝试初始化 SpaceMIT EP
- `inference.queue_size`：实际输入和 ready 队列采用固定的 latest-only 深度
- `inference.skip_frames`
- `postprocess.nms_type`：当前使用 OpenCV class-aware `NMSBoxes`
- `postprocess.tracker.*`
- `display.fps_overlay`、`display.bbox_thickness`、`display.label_font_size`、`display.color_palette`

## 性能日志

程序会输出 `PIPE` 和 `PERF` 两类性能日志。`PIPE` 用于观察解码和算法送帧节奏：

```text
PIPE ch=0 decoded_fps=... alg_in_fps=... infer_stride=... frame_id=...
```

- **decoded_fps**：VDEC 输出帧率
- **alg_in_fps**：送入算法通路的帧率
- **infer_stride**：算法抽帧步长

`PERF` 用于观察前处理、推理和后处理各阶段：

```text
PERF ch=0 fps=... decode_q=... pre_wait=... pre=...
infer_q=... run_active=... infer=... post=... total=... det=...
```

- **fps**：该通道完成后处理并回调结果的帧率
- **decode_q / pre_wait**：解码帧进入算法通路后的排队等待
- **pre**：OpenCL 前处理耗时
- **infer_q**：ready 队列等待可用 Session 的时间
- **run_active**：本次推理开始时的并发推理数量
- **infer**：ORT tensor setup 与 `Session::Run` 总耗时
- **post**：阈值过滤、NMS 和坐标还原耗时
- **total**：各阶段累计端到端时延
- **det**：当前帧最终检测框数量

12 路总推理 FPS 应对 12 个通道最新的 `PERF fps` 求和，不能使用解码帧率代替。

## 常见问题

### 启动后提示找不到模型？

确认模型文件已经安装，并检查 `model.cache_dir`。例如 YOLOv8 Nano 的默认路径为：

```text
/usr/share/yolo-demo/yolov8/yolov8n.q.onnx
```

### 界面没有视频？

- 确认 **启用视频显示** 已勾选
- 确认配置中的本地视频路径存在
- 确认当前运行在 Bianbu 图形桌面会话中
- 通过 SSH 启动时，检查 `WAYLAND_DISPLAY` 和 `XDG_RUNTIME_DIR`

### 多路延迟持续升高？

- 使用更小的模型
- 降低 `infer_fps` 或 `decode_fps`
- 关闭视频显示，观察是否为显示带宽造成的瓶颈
- 检查系统中是否还有其他 AI 推理任务占用 TCM

### 出现 `tcm buffer acquire failed`？

先查看 TCM 状态：

```bash
spacemit-tcm-smi
```

清理 TCM：

```bash
spacemit-tcm-smi -c
```

不要在 `yolo-demo` 或其他推理程序运行时清理 TCM。

### MPP 打印多个 `/dev/videoX is not a M2M device`？

MPP 启动时会扫描视频节点。只要后续能够找到有效解码设备并成功创建 VDEC channel，这类扫描日志不一定表示启动失败。
