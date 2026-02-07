
import cv2
import numpy as np
import os
from shapely.geometry import LineString, Point

# Mock classes to match map_server.py dependencies
class FeatureType:
    HIGHWAY = "highway"
    LOCAL_ROAD = "road"
    MAIN_ROAD = "main_road"
    BUILDING = "building"

class Style:
    def __init__(self, priority):
        self.priority = priority

class StyleManager:
    @staticmethod
    def get_style(type):
        return Style(1)

class MapFeature:
    def __init__(self, id, type, geometry, style):
        self.id = id
        self.type = type
        self.geometry = geometry
        self.style = style

def process_image(image_path):
    print(f"Processing image: {image_path}")
    features = []
    
    try:
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise ValueError("Could not read image")
        
        # 1. Resize for Speed (Work on a copy, scale back later)
        max_dim = 1500
        h_orig, w_orig = img_original.shape[:2]
        scale = 1.0
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            img = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = img_original.copy()
        
        h, w = img.shape[:2]
        print(f"Original Dimensions: {w_orig}x{h_orig}")
        print(f"Processing Dimensions: {w}x{h} (Scale: {scale:.4f})")
        
        # Denoise
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # --- STRATEGY 1: Color-Specific Extraction (Arcadia/OpenStreetMap Style) ---
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
        
        # Orange Roads (Main Highways)
        lower_orange = np.array([5, 120, 120]) # Tuned S/V
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        orange_pixels = cv2.countNonZero(mask_orange)
        has_orange_roads = orange_pixels > (w * h * 0.001) # Adaptive threshold
        
        print(f"Orange pixels: {orange_pixels} (Threshold: {w * h * 0.001}) - Has Orange Roads: {has_orange_roads}")

        road_count = 0
        lines_mask = np.zeros((h, w), dtype=np.uint8)

        if has_orange_roads:
            # Cleaner mask
            kernel = np.ones((5,5), np.uint8)
            mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)
            mask_orange = cv2.dilate(mask_orange, kernel, iterations=1)
            
            # Extract Orange Lines
            lines_orange = cv2.HoughLinesP(mask_orange, 1, np.pi/180, 
                                         threshold=50, minLineLength=40, maxLineGap=20)
            
            if lines_orange is not None:
                print(f"Found {len(lines_orange)} orange line segments")
                for line in lines_orange:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 5)
                    features.append("orange_road")
                    road_count += 1
            else:
                 print("Found 0 orange line segments despite having pixels")

        # --- Local Roads (White/Grey lines) ---
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        
        if has_orange_roads:
            mask_orange_dilated = cv2.dilate(mask_orange, np.ones((7,7), np.uint8), iterations=2)
            gray_masked = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask_orange_dilated))
        else:
            gray_masked = gray

        edges = cv2.Canny(gray_masked, 40, 120, apertureSize=3)
        edge_pixels = cv2.countNonZero(edges)
        print(f"Canny Edges pixels: {edge_pixels}")
        
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
        
        lines_local = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=60, minLineLength=30, maxLineGap=15)
                                    
        if lines_local is not None:
            print(f"Found {len(lines_local)} local line segments")
            raw_local_count = len(lines_local)
            filtered_local_count = 0
            for line in lines_local:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length < 25: continue
                filtered_local_count += 1
                
                cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 3)
                features.append("local_road")
                road_count += 1
            print(f"Kept {filtered_local_count} local roads after filtering")
        else:
            print("Found 0 local line segments")

        # --- Building Detection ---
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 4)
        
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        lines_mask_dilated = cv2.dilate(lines_mask, np.ones((5,5), np.uint8), iterations=2)
        thresh_clean = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(lines_mask_dilated))
        
        contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} initial contours")
        
        building_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = (h * w) * 0.00005 
            max_area = (h * w) * 0.01   
            
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = float(cw)/ch
                if aspect_ratio > 4 or aspect_ratio < 0.25:
                    continue
                    
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area if hull_area > 0 else 0
                if solidity < 0.7: 
                    continue
                
                features.append("building")
                building_count += 1
        
        print(f"Final Count - Roads: {road_count}, Buildings: {building_count}")
        return features
            
    except Exception as e:
        print(f"Error extracting features: {e}")
        return []

if __name__ == "__main__":
    # Find latest screenshot
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        files = [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.startswith("Screenshot")]
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(f"Analyzing latest file: {latest_file}")
            process_image(latest_file)
        else:
            print("No screenshots found.")
    else:
        print("Uploads dir not found.")
