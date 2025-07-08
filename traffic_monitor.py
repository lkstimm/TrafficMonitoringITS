import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
import tempfile
import os
from PIL import Image
import pandas as pd
from typing import List, Dict, Tuple, Optional
import time
import threading
from collections import defaultdict

class ROI:
    def __init__(self, name: str, points: List[Tuple[int, int]], color=(0, 255, 0)):
        self.name = name
        self.points = np.array(points, np.int32)
        self.color = color
        self.active_color = (0, 0, 255)  # Red when objects detected
        self.object_counts = defaultdict(int)
        self.is_active = False
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the ROI polygon"""
        return cv2.pointPolygonTest(self.points, point, False) >= 0
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the ROI on the frame"""
        color = self.active_color if self.is_active else self.color
        
        # Draw filled polygon with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], color)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw border
        cv2.polylines(frame, [self.points], True, color, 2)
        
        # Draw ROI name and counts
        x, y = self.points[0]
        y_offset = 0
        cv2.putText(frame, f"ROI: {self.name}", (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        for obj_type, count in self.object_counts.items():
            if count > 0:
                cv2.putText(frame, f"{obj_type}: {count}", (x, y + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_offset += 20
        
        return frame

class TrafficMonitor:
    def __init__(self):
        self.model = None
        self.roi_list = []
        self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        self.class_mapping = {
            'person': 'pedestrian',
            'bicycle': 'bike',
            'car': 'car',
            'motorcycle': 'bike',
            'bus': 'truck',
            'truck': 'truck'
        }
        self.current_frame = None
        self.video_cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.frame_skip = 20  # Process every 20th frame for performance
        
    def load_model(self):
        """Load YOLO model"""
        if self.model is None:
            self.model = YOLO('yolov8n.pt')  # Using nano version for speed
    
    def download_youtube_video(self, url: str) -> Optional[str]:
        """Download YouTube video and return local path"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, 'video.%(ext)s')
                
                ydl_opts = {
                    'format': 'best[height<=720]',  # Limit resolution for performance
                    'outtmpl': output_path,
                    'noplaylist': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    downloaded_file = ydl.prepare_filename(info)
                    
                # Copy to permanent location
                permanent_path = 'downloaded_video.mp4'
                os.rename(downloaded_file, permanent_path)
                return permanent_path
                
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")
            return None
    
    def load_video(self, video_path: str):
        """Load video file"""
        if video_path:
            self.video_cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.current_frame_idx = 0
    
    def get_frame(self, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """Get specific frame from video"""
        if self.video_cap is None:
            return None
            
        if frame_idx is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_frame_idx = frame_idx
        
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            return frame
        return None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame"""
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return []
        
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    if class_name in self.target_classes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Calculate center point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        mapped_class = self.class_mapping.get(class_name, class_name)
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': (center_x, center_y),
                            'class': mapped_class,
                            'confidence': float(confidence)
                        })
        
        return detections
    
    def update_roi_counts(self, detections: List[Dict]):
        """Update object counts for each ROI"""
        # Reset all ROI counts and active status
        for roi in self.roi_list:
            roi.object_counts.clear()
            roi.is_active = False
        
        # Count objects in each ROI
        for detection in detections:
            center = detection['center']
            obj_class = detection['class']
            
            for roi in self.roi_list:
                if roi.contains_point(center):
                    roi.object_counts[obj_class] += 1
                    roi.is_active = True
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            bbox = detection['bbox']
            center = detection['center']
            obj_class = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, center, 3, (0, 255, 255), -1)
            
            # Draw label
            label = f"{obj_class}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), (255, 0, 0), -1)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Draw all ROIs on frame"""
        for roi in self.roi_list:
            frame = roi.draw(frame)
        return frame
    
    def add_roi(self, name: str, points: List[Tuple[int, int]]):
        """Add a new ROI"""
        roi = ROI(name, points)
        self.roi_list.append(roi)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with object detection and ROI analysis"""
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Update ROI counts
        self.update_roi_counts(detections)
        
        # Draw everything
        frame = self.draw_detections(frame, detections)
        frame = self.draw_rois(frame)
        
        return frame

def main():
    st.set_page_config(page_title="Traffic Monitoring System", layout="wide")
    
    st.title("🚦 Traffic Monitoring System")
    st.write("An AI-powered traffic monitoring solution with object detection and region-of-interest analysis")
    
    # Initialize monitor
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TrafficMonitor()
        st.session_state.video_loaded = False
        st.session_state.playing = False
        st.session_state.roi_drawing_mode = False
        st.session_state.roi_points = []
        st.session_state.next_frame = None
    
    monitor = st.session_state.monitor
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Video Controls")
        
        # YouTube URL input
        youtube_url = st.text_input("YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        
        if st.button("Load YouTube Video") and youtube_url:
            with st.spinner("Downloading video..."):
                video_path = monitor.download_youtube_video(youtube_url)
                if video_path:
                    monitor.load_video(video_path)
                    st.session_state.video_loaded = True
                    st.success("Video loaded successfully!")
        
        # File upload option
        uploaded_file = st.file_uploader("Or upload a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                monitor.load_video(tmp_file.name)
                st.session_state.video_loaded = True
                st.success("Video loaded successfully!")
        
        if st.session_state.video_loaded:
            st.header("Playback Controls")
            
            # Frame slider
            frame_idx = st.slider("Frame", 0, monitor.total_frames - 1, monitor.current_frame_idx)
            
            # Handle auto-play frame updates
            if st.session_state.next_frame is not None:
                frame_idx = st.session_state.next_frame
                st.session_state.next_frame = None
            
            # Performance controls
            st.subheader("Performance Settings")
            frame_skip = st.selectbox("Frame Skip (for performance):", 
                                    [1, 5, 10, 20, 30], 
                                    index=3,  # Default to 20
                                    help="Process every Nth frame. Higher values = faster processing but less smooth.")
            monitor.frame_skip = frame_skip
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("⏮️ Step Back"):
                    frame_idx = max(0, monitor.current_frame_idx - monitor.frame_skip)
            with col2:
                play_button_text = "⏸️ Pause" if st.session_state.playing else "▶️ Play"
                if st.button(play_button_text):
                    st.session_state.playing = not st.session_state.playing
            with col3:
                if st.button("⏭️ Step Forward"):
                    frame_idx = min(monitor.total_frames - 1, monitor.current_frame_idx + monitor.frame_skip)
            
            # ROI Management
            st.header("Region of Interest")
            
            roi_name = st.text_input("ROI Name:", placeholder="Enter ROI name")
            
            # ROI input method selection
            roi_method = st.selectbox("ROI Input Method:", 
                                    ["Coordinate Input", "Visual Drawing (Advanced)"])
            
            if roi_method == "Coordinate Input":
                st.subheader("Define ROI by Coordinates")
                st.write("Enter polygon coordinates (x,y) separated by commas:")
                st.write("Example: 100,200 300,200 300,400 100,400")
                
                coordinates_text = st.text_area("Coordinates:", 
                                               placeholder="x1,y1 x2,y2 x3,y3 ...",
                                               height=100)
                
                if st.button("Create ROI from Coordinates") and roi_name and coordinates_text:
                    try:
                        # Parse coordinates
                        coord_pairs = coordinates_text.strip().split()
                        points = []
                        for pair in coord_pairs:
                            x, y = map(int, pair.split(','))
                            points.append((x, y))
                        
                        if len(points) >= 3:
                            monitor.add_roi(roi_name, points)
                            st.success(f"ROI '{roi_name}' created with {len(points)} points!")
                        else:
                            st.error("Need at least 3 points to create an ROI")
                    except ValueError:
                        st.error("Invalid coordinate format. Use: x1,y1 x2,y2 x3,y3 ...")
            
            else:  # Visual Drawing
                st.subheader("Visual ROI Drawing")
                st.info("⚠️ Visual drawing requires external window. Use coordinate input for easier setup.")
                
                if st.button("Start Drawing ROI"):
                    st.session_state.roi_drawing_mode = True
                    st.session_state.roi_points = []
                    st.info("Click on the video to define ROI points. Click 'Finish ROI' when done.")
                
                if st.button("Finish ROI") and roi_name and len(st.session_state.roi_points) >= 3:
                    monitor.add_roi(roi_name, st.session_state.roi_points)
                    st.session_state.roi_drawing_mode = False
                    st.session_state.roi_points = []
                    st.success(f"ROI '{roi_name}' added successfully!")
            
            # Quick ROI presets for common areas
            st.subheader("Quick ROI Presets")
            if st.session_state.video_loaded and monitor.current_frame is not None:
                frame_height, frame_width = monitor.current_frame.shape[:2]
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Left Lane"):
                        points = [(0, int(frame_height*0.4)), 
                                (int(frame_width*0.3), int(frame_height*0.4)),
                                (int(frame_width*0.3), frame_height),
                                (0, frame_height)]
                        monitor.add_roi("Left Lane", points)
                        st.success("Left Lane ROI added!")
                
                with col2:
                    if st.button("Right Lane"):
                        points = [(int(frame_width*0.7), int(frame_height*0.4)),
                                (frame_width, int(frame_height*0.4)),
                                (frame_width, frame_height),
                                (int(frame_width*0.7), frame_height)]
                        monitor.add_roi("Right Lane", points)
                        st.success("Right Lane ROI added!")
            
            # Display current ROIs
            if monitor.roi_list:
                st.subheader("Current ROIs:")
                for i, roi in enumerate(monitor.roi_list):
                    with st.expander(f"ROI: {roi.name}"):
                        total_objects = sum(roi.object_counts.values())
                        st.write(f"**Total objects:** {total_objects}")
                        
                        if roi.object_counts:
                            df = pd.DataFrame(list(roi.object_counts.items()), 
                                            columns=['Object Type', 'Count'])
                            st.dataframe(df, use_container_width=True)
                        
                        # ROI management buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Delete {roi.name}", key=f"delete_{i}"):
                                monitor.roi_list.remove(roi)
                                st.rerun()
                        with col2:
                            # Show coordinates
                            points_str = " ".join([f"{x},{y}" for x, y in roi.points])
                            st.text_input("Coordinates:", value=points_str, 
                                        key=f"coords_{i}", disabled=True)
            
            # Export/Import ROI configurations
            st.subheader("ROI Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export ROIs") and monitor.roi_list:
                    import json
                    roi_data = []
                    for roi in monitor.roi_list:
                        roi_data.append({
                            'name': roi.name,
                            'points': roi.points.tolist()
                        })
                    
                    json_str = json.dumps(roi_data, indent=2)
                    st.download_button(
                        label="Download ROI Config",
                        data=json_str,
                        file_name="roi_config.json",
                        mime="application/json"
                    )
            
            with col2:
                uploaded_roi_file = st.file_uploader("Import ROI Config", type=['json'])
                if uploaded_roi_file:
                    import json
                    try:
                        roi_data = json.load(uploaded_roi_file)
                        for roi_info in roi_data:
                            monitor.add_roi(roi_info['name'], roi_info['points'])
                        st.success(f"Imported {len(roi_data)} ROIs!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing ROI config: {str(e)}")
    
    # Main content area
    if st.session_state.video_loaded:
        # Get current frame
        frame = monitor.get_frame(frame_idx)
        
        if frame is not None:
            # Process frame with object detection
            processed_frame = monitor.process_frame(frame)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            st.image(processed_frame_rgb, caption=f"Frame {monitor.current_frame_idx}/{monitor.total_frames}", 
                    use_container_width=True)
            
            # Frame info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Frame", monitor.current_frame_idx)
            with col2:
                st.metric("Total Frames", monitor.total_frames)
            with col3:
                st.metric("Original FPS", f"{monitor.fps:.2f}")
            with col4:
                effective_fps = monitor.fps / monitor.frame_skip
                st.metric("Effective FPS", f"{effective_fps:.1f}")
            
            # Auto-play functionality with frame skipping
            if st.session_state.playing and monitor.current_frame_idx < monitor.total_frames - monitor.frame_skip:
                # Skip to next processing frame (every 20th frame)
                next_frame = min(monitor.current_frame_idx + monitor.frame_skip, monitor.total_frames - 1)
                # Shorter delay for smoother playback since we're skipping frames
                time.sleep(0.5)  # 2 FPS effective playback speed
                # Force rerun to update to next frame
                st.session_state.next_frame = next_frame
                st.rerun()
    
    else:
        st.info("Please load a video to start monitoring traffic.")
        
        # Show demo instructions
        st.markdown("""
        ### How to use:
        1. **Load a video**: Enter a YouTube URL or upload a video file
        2. **Define ROIs**: Use the sidebar to create regions of interest
        3. **Monitor traffic**: Watch as objects are detected and counted in each ROI
        4. **Control playback**: Use play/pause and step controls
        
        ### Supported objects:
        - 🚗 Cars
        - 🚲 Bikes/Motorcycles  
        - 🚶 Pedestrians
        - 🚛 Trucks/Buses
        
        ### ROI Features:
        - **Green overlay**: No objects detected
        - **Red overlay**: Objects detected in the region
        - **Real-time counting**: See object counts by type
        """)

if __name__ == "__main__":
    main() 