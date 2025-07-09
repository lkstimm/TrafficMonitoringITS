import cv2
import numpy as np
from typing import List, Tuple, Callable, Optional

Print ("hello world, this is a bug")

class ROIDrawer:
    def __init__(self, window_name: str = "ROI Drawer"):
        self.window_name = window_name
        self.points = []
        self.current_roi_points = []
        self.temp_point = None
        self.drawing = False
        self.roi_list = []
        self.current_roi_name = ""
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.drawing:
                self.current_roi_points.append((x, y))
                print(f"Point added: ({x}, {y})")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_point = (x, y)
    
    def draw_current_roi(self, frame: np.ndarray) -> np.ndarray:
        """Draw the currently being drawn ROI"""
        if not self.drawing or len(self.current_roi_points) < 1:
            return frame
        
        # Draw points
        for point in self.current_roi_points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)
        
        # Draw lines between consecutive points
        if len(self.current_roi_points) > 1:
            for i in range(len(self.current_roi_points) - 1):
                cv2.line(frame, self.current_roi_points[i], 
                        self.current_roi_points[i + 1], (0, 255, 0), 2)
        
        # Draw line from last point to current mouse position
        if self.temp_point and len(self.current_roi_points) > 0:
            cv2.line(frame, self.current_roi_points[-1], self.temp_point, (0, 255, 0), 1)
        
        # Draw closing line if we have enough points
        if len(self.current_roi_points) > 2:
            cv2.line(frame, self.current_roi_points[-1], 
                    self.current_roi_points[0], (0, 255, 0), 1)
        
        return frame
    
    def draw_saved_rois(self, frame: np.ndarray) -> np.ndarray:
        """Draw all saved ROIs"""
        for roi_data in self.roi_list:
            points = np.array(roi_data['points'], np.int32)
            cv2.polylines(frame, [points], True, (255, 0, 0), 2)
            
            # Draw ROI name
            if len(points) > 0:
                x, y = points[0]
                cv2.putText(frame, roi_data['name'], (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def start_roi(self, name: str):
        """Start drawing a new ROI"""
        self.current_roi_name = name
        self.current_roi_points = []
        self.drawing = True
        print(f"Started drawing ROI: {name}")
        print("Click points to define the region. Press 'f' to finish, 'c' to cancel, 'u' to undo last point")
    
    def finish_roi(self) -> Optional[dict]:
        """Finish the current ROI and return it"""
        if len(self.current_roi_points) >= 3:
            roi_data = {
                'name': self.current_roi_name,
                'points': self.current_roi_points.copy()
            }
            self.roi_list.append(roi_data)
            self.drawing = False
            self.current_roi_points = []
            print(f"ROI '{self.current_roi_name}' finished with {len(roi_data['points'])} points")
            return roi_data
        else:
            print("Need at least 3 points to create an ROI")
            return None
    
    def cancel_roi(self):
        """Cancel the current ROI drawing"""
        self.drawing = False
        self.current_roi_points = []
        print("ROI drawing cancelled")
    
    def undo_point(self):
        """Remove the last point"""
        if self.current_roi_points:
            removed_point = self.current_roi_points.pop()
            print(f"Removed point: {removed_point}")
    
    def draw_interactive(self, frame: np.ndarray, roi_name: str = "") -> Optional[List[Tuple[int, int]]]:
        """
        Interactive ROI drawing interface
        Returns the points of the completed ROI, or None if cancelled
        """
        self.start_roi(roi_name)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            display_frame = frame.copy()
            
            # Draw saved ROIs
            display_frame = self.draw_saved_rois(display_frame)
            
            # Draw current ROI being drawn
            display_frame = self.draw_current_roi(display_frame)
            
            # Add instructions
            instructions = [
                "Left click: Add point",
                "F: Finish ROI",
                "C: Cancel",
                "U: Undo last point",
                "ESC: Exit"
            ]
            
            y_offset = 30
            for instruction in instructions:
                cv2.putText(display_frame, instruction, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Show current ROI info
            if self.drawing:
                status = f"Drawing: {self.current_roi_name} ({len(self.current_roi_points)} points)"
                cv2.putText(display_frame, status, (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('f') or key == ord('F'):
                roi_data = self.finish_roi()
                if roi_data:
                    return roi_data['points']
            
            elif key == ord('c') or key == ord('C'):
                self.cancel_roi()
                return None
            
            elif key == ord('u') or key == ord('U'):
                self.undo_point()
            
            elif key == 27:  # ESC key
                break
        
        cv2.destroyWindow(self.window_name)
        return None

def demo_roi_drawer():
    """Demo function to test ROI drawing"""
    # Create a test image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Add some text
    cv2.putText(img, "ROI Drawing Demo", (250, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Draw polygons by clicking points", (200, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    drawer = ROIDrawer()
    
    while True:
        roi_name = input("Enter ROI name (or 'quit' to exit): ")
        if roi_name.lower() == 'quit':
            break
        
        points = drawer.draw_interactive(img.copy(), roi_name)
        if points:
            print(f"ROI '{roi_name}' created with points: {points}")
        else:
            print("ROI creation cancelled")
    
    print("All ROIs created:")
    for roi in drawer.roi_list:
        print(f"- {roi['name']}: {roi['points']}")

if __name__ == "__main__":
    demo_roi_drawer() 
