# 🚦 Traffic Monitoring System

An AI-powered traffic monitoring solution with real-time object detection, region-of-interest analysis, and interactive video controls.

## ✨ Features

### 🎯 Object Detection
- **YOLOv8-powered detection** for high accuracy and speed
- **Multi-class support**: Cars, bikes/motorcycles, pedestrians, trucks/buses
- **Real-time bounding boxes** with confidence scores
- **Object center point tracking** for precise ROI analysis

### 📍 Region of Interest (ROI) Management
- **Interactive polygon drawing** to define custom areas
- **Visual feedback**: Green overlay (no objects) → Red overlay (objects detected)
- **Real-time object counting** by type within each ROI
- **Multiple ROI support** with individual naming and tracking

### 🎥 Video Processing
- **YouTube video support** with automatic downloading
- **Local video file upload** (MP4, AVI, MOV)
- **Frame-by-frame navigation** with slider control
- **Play/Pause functionality** with automatic frame progression
- **Step-through controls** (forward/backward single frame)

### 🖥️ User Interface
- **Modern web-based interface** using Streamlit
- **Responsive design** with sidebar controls
- **Real-time metrics display** (frame count, FPS, object counts)
- **Interactive ROI management** with expandable details

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

#### Option A: Automated Setup (Recommended)
1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd TrafficMonitoringITS
```

2. **Run the setup script:**
```bash
python setup.py
```

3. **Launch the application:**
```bash
python launch.py
```

#### Option B: Manual Setup
1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd TrafficMonitoringITS
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run traffic_monitor.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### 1. Loading a Video

#### Option A: YouTube Video
1. Copy the YouTube video URL
2. Paste it in the "YouTube Video URL" field in the sidebar
3. Click "Load YouTube Video"
4. Wait for the download to complete

#### Option B: Local File
1. Click "Browse files" in the file uploader
2. Select your video file (MP4, AVI, or MOV)
3. The video will load automatically

### 2. Defining Regions of Interest (ROI)

#### Method A: Coordinate Input (Recommended)
1. **Name your ROI:** Enter a descriptive name in the "ROI Name" field
2. **Select "Coordinate Input"** from the ROI Input Method dropdown
3. **Enter coordinates:** Type polygon points as "x1,y1 x2,y2 x3,y3 ..."
   - Example: `100,200 300,200 300,400 100,400`
4. **Create ROI:** Click "Create ROI from Coordinates"

#### Method B: Quick Presets
1. **Load a video first** to enable presets
2. **Click "Left Lane" or "Right Lane"** for automatic lane ROIs
3. **Customize as needed** using coordinate editing

#### Method C: Import Configuration
1. **Use sample config:** Import `sample_roi_config.json` for demo ROIs
2. **Custom config:** Export your ROIs and reuse them later

#### Method D: Visual Drawing (Advanced)
1. **Select "Visual Drawing"** from the dropdown
2. **External window:** Use the OpenCV window for precise drawing
3. **Interactive controls:** Click points, press F to finish

### 3. Video Controls

- **🎮 Frame Navigation:** Use the slider to jump to specific frames
- **⏮️ Step Back:** Move one frame backward
- **▶️ Play/⏸️ Pause:** Start or stop automatic playback
- **⏭️ Step Forward:** Move one frame forward

### 4. Monitoring Traffic

Once you have ROIs defined and video loaded:

1. **Object Detection:** Watch as objects are automatically detected with bounding boxes
2. **ROI Status:** Green overlays indicate no objects, red indicates objects detected
3. **Real-time Counts:** View object counts by type in the sidebar ROI panels
4. **Visual Feedback:** Object counts are displayed directly on the ROI overlays

## 🛠️ Technical Details

### Object Detection Model
- **Model:** YOLOv8 Nano (yolov8n.pt)
- **Classes Detected:** 
  - `person` → `pedestrian`
  - `bicycle` → `bike`
  - `motorcycle` → `bike`
  - `car` → `car`
  - `bus` → `truck`
  - `truck` → `truck`

### ROI Detection Algorithm
- **Point-in-polygon testing** using OpenCV's `pointPolygonTest`
- **Center-point based detection** (uses object bounding box center)
- **Real-time counting** with automatic reset per frame

### Performance Optimizations
- **Efficient video processing** with frame caching
- **Optimized model inference** using YOLOv8 Nano
- **Selective class filtering** for traffic-relevant objects only
- **Smart video resolution limiting** (720p max for YouTube)

## 📁 Project Structure

```
TrafficMonitoringITS/
├── traffic_monitor.py         # Main application
├── roi_drawer.py             # ROI drawing utility
├── setup.py                  # Automated setup script
├── launch.py                 # Convenient launcher (created by setup)
├── sample_roi_config.json    # Sample ROI configuration
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── videos/                  # Downloaded videos (created by setup)
└── models/                  # YOLO models (created by setup)
└── roi_configs/             # Saved ROI configurations (created by setup)
```

## 🔧 Configuration Options

### Model Configuration
You can modify the YOLO model in `traffic_monitor.py`:
```python
self.model = YOLO('yolov8n.pt')  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
```

### Target Classes
Modify the detected object types:
```python
self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
```

### Class Mapping
Customize how detected classes are displayed:
```python
self.class_mapping = {
    'person': 'pedestrian',
    'bicycle': 'bike',
    # Add more mappings...
}
```

## 🎨 ROI Drawing Tool

For advanced ROI creation, you can use the standalone ROI drawer:

```bash
python roi_drawer.py
```

This opens an interactive drawing interface with:
- **Left click:** Add polygon points
- **F:** Finish current ROI
- **C:** Cancel current ROI
- **U:** Undo last point
- **ESC:** Exit

## 🚨 Troubleshooting

### Common Issues

1. **"Model not found" error:**
   - The YOLOv8 model will download automatically on first run
   - Ensure you have internet connection for initial download

2. **YouTube download fails:**
   - Check if the URL is valid and accessible
   - Some videos may be region-restricted or have download limitations

3. **Poor detection performance:**
   - Try using a larger YOLO model (yolov8s, yolov8m)
   - Ensure good video quality and lighting

4. **Slow performance:**
   - Use smaller video resolution
   - Switch to yolov8n (nano) model for speed
   - Close other applications to free up resources

### Performance Tips

- **Video Resolution:** Lower resolution videos process faster
- **ROI Complexity:** Simpler polygon shapes perform better
- **Model Size:** Nano model (yolov8n) is fastest, X model (yolov8x) is most accurate

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **Streamlit** for the user-friendly web interface framework
- **OpenCV** for computer vision capabilities
- **yt-dlp** for YouTube video downloading functionality

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed description and steps to reproduce

---

**Happy Monitoring!** 🚦✨ 