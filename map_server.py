import os
import cv2
import numpy as np
import json
from flask import Flask, request, render_template, send_file, make_response
from werkzeug.utils import secure_filename
from shapely.geometry import LineString, Point, Polygon

# Import the existing system
from map_declutter_system import MapDeclutterSystem, MapFeature, FeatureType, StyleManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MapImageProcessor:
    """
    Extracts map features from an image using Computer Vision.
    """
    @staticmethod
    def _merge_lines(lines, distance_threshold=40, angle_threshold=15):
        """
        Merge collinear and nearby lines to reduce fragmentation.
        lines: List of [x1, y1, x2, y2]
        """
        if lines is None or len(lines) == 0:
            return []
            
        # Convert to list of dicts for easier handling
        line_objs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Normalize direction (left to right)
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            
            line_objs.append({
                'coords': [x1, y1, x2, y2],
                'length': length,
                'angle': angle,
                'merged': False
            })
            
        # Sort by length (descending)
        line_objs.sort(key=lambda x: x['length'], reverse=True)
        
        merged_lines = []
        
        for i, l1 in enumerate(line_objs):
            if l1['merged']:
                continue
                
            current_merge = l1['coords']
            l1['merged'] = True
            
            # Look for others to merge - Multiple passes or just greedy approach
            # Greedy approach: keep trying to merge into current_merge
            any_merged = True
            while any_merged:
                any_merged = False
                for j, l2 in enumerate(line_objs):
                    if i == j or l2['merged']:
                        continue
                    
                    # Check angle difference
                    angle_diff = abs(l1['angle'] - l2['angle'])
                    if angle_diff > 180: angle_diff = 360 - angle_diff
                    
                    if angle_diff < angle_threshold:
                        # Check distance between endpoints
                        xa, ya, xb, yb = current_merge
                        xc, yc, xd, yd = l2['coords']
                        
                        # Distances between all pairs of endpoints
                        d_ac = np.sqrt((xa-xc)**2 + (ya-yc)**2)
                        d_ad = np.sqrt((xa-xd)**2 + (ya-yd)**2)
                        d_bc = np.sqrt((xb-xc)**2 + (yb-yc)**2)
                        d_bd = np.sqrt((xb-xd)**2 + (yb-yd)**2)
                        
                        min_gap = min(d_ac, d_ad, d_bc, d_bd)
                        
                        # Also check perpendicular distance if they are parallel but side-by-side
                        # For now, end-point proximity is enough for road segments
                        
                        if min_gap < distance_threshold:
                            # MERGE
                            # New coords are extremas of the combined set
                            points = [(xa,ya), (xb,yb), (xc,yc), (xd,yd)]
                            
                            max_dist = 0
                            best_pair = current_merge 
                            
                            for p1_idx in range(4):
                                for p2_idx in range(p1_idx+1, 4):
                                    px1, py1 = points[p1_idx]
                                    px2, py2 = points[p2_idx]
                                    d = np.sqrt((px1-px2)**2 + (py1-py2)**2)
                                    if d > max_dist:
                                        max_dist = d
                                        best_pair = [px1, py1, px2, py2]
                            
                            current_merge = best_pair
                            l2['merged'] = True
                            any_merged = True # Keep looking for more to add to this super-line
            
            merged_lines.append([current_merge])
            
        return merged_lines

    @staticmethod
    def process_image(image_path):
        print(f"Processing image: {image_path}")
        features = []
        
        try:
            img_original = cv2.imread(image_path)
            if img_original is None:
                raise ValueError("Could not read image")
            
            # 1. Resize for consistent processing
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
            
            # 2. Heavy Blur to obscure text labels (The main source of "waste lines")
            # We want to keep large blocks of color (roads, rivers) but kill thin text strokes
            img_blur = cv2.medianBlur(img, 7)
            img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
            
            occupied_mask = np.zeros((h, w), dtype=np.uint8)
            
            # helper to add features
            def add_feature(geometry, ftype):
                nonlocal road_count, river_count, bldg_count
                if ftype == FeatureType.HIGHWAY:
                    fid = f"hwy_{road_count:03d}"
                    road_count += 1
                elif ftype == FeatureType.MAIN_ROAD or ftype == FeatureType.LOCAL_ROAD:
                    fid = f"road_{road_count:03d}"
                    road_count += 1
                elif ftype == FeatureType.RIVER:
                    fid = f"river_{river_count:02d}"
                    river_count += 1
                elif ftype == FeatureType.BUILDING:
                    fid = f"bldg_{bldg_count:03d}"
                    bldg_count += 1
                elif ftype == FeatureType.PARK:
                    fid = f"park_{bldg_count:03d}"
                    bldg_count += 1
                    
                features.append(MapFeature(
                    id=fid,
                    type=ftype,
                    geometry=geometry,
                    style=StyleManager.get_style(ftype)
                ))

            road_count = 0
            river_count = 0
            bldg_count = 0

            # --- 1. RIVERS (Blue) ---
            # Very specific blue range for maps
            lower_blue = np.array([90, 40, 150])
            upper_blue = np.array([130, 255, 255])
            mask_river = cv2.inRange(img_hsv, lower_blue, upper_blue)
            
            # Clean up river mask
            mask_river = cv2.morphologyEx(mask_river, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            mask_river = cv2.morphologyEx(mask_river, cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
            
            contours_river, _ = cv2.findContours(mask_river, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_river:
                if cv2.contourArea(cnt) > (w * h * 0.005): # Only big rivers
                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    poly_coords = []
                    for p in approx:
                        px, py = p[0]
                        poly_coords.append((px/scale, h_orig - py/scale))
                    
                    if len(poly_coords) > 2:
                        add_feature(Polygon(poly_coords), FeatureType.RIVER)
                        cv2.drawContours(occupied_mask, [cnt], -1, 255, -1)

            # --- 2. HIGHWAYS (Yellow/Orange) ---
            # Combined warm colors
            lower_yellow = np.array([15, 70, 150])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
            
            lower_orange = np.array([5, 100, 150])
            upper_orange = np.array([15, 255, 255])
            mask_orange = cv2.inRange(img_hsv, lower_orange, upper_orange)
            
            mask_warm = cv2.bitwise_or(mask_yellow, mask_orange)
            
            # Remove small dots (likely text/icons)
            mask_warm = cv2.morphologyEx(mask_warm, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            mask_warm = cv2.dilate(mask_warm, np.ones((3,3),np.uint8))
            
            # Hough Transform for lines
            lines_warm = cv2.HoughLinesP(mask_warm, 1, np.pi/180, 
                                       threshold=50, minLineLength=80, maxLineGap=40)
            
            if lines_warm is not None:
                lines_warm = MapImageProcessor._merge_lines(lines_warm, distance_threshold=40)
                for line in lines_warm:
                    x1, y1, x2, y2 = line[0]
                    
                    # Store
                    fx1, fy1 = x1/scale, y1/scale
                    fx2, fy2 = x2/scale, y2/scale
                    add_feature(LineString([(fx1, h_orig - fy1), (fx2, h_orig - fy2)]), FeatureType.HIGHWAY)
                    
                    # Update mask
                    cv2.line(occupied_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 15)

            # --- 3. ROADS (White/Light Grey) ---
            # This is tricky. Roads are usually high value, low saturation.
            # But so is the background.
            # We look for "Light" pixels that are NOT the background color?
            # Or we look for edges of white pixels?
            
            # Strict White filter
            lower_white = np.array([0, 0, 200])   # Very bright
            upper_white = np.array([180, 30, 255]) # Low saturation
            mask_white = cv2.inRange(img_hsv, lower_white, upper_white)
            
            # Remove text (black pixels in white background, or white pixels in dark)
            # Since we are looking for white roads, text (dark) creates "holes" in the white mask.
            # We can Close these holes.
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            
            # Now we have a blob of "Background + Roads".
            # Roads are usually thin connected components. Background is huge blocks.
            # We'll use Skeletonization or simple line detection on the white mask.
            
            # BUT, to avoid "waste lines" from text, we must remove small noise first
            mask_white_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            # Mask out what we already found
            mask_white_clean = cv2.bitwise_and(mask_white_clean, mask_white_clean, mask=cv2.bitwise_not(occupied_mask))
            
            # Detect lines in the WHITE areas
            # Use Canny ONLY on the masked white area to find edges of roads
            edges_white = cv2.Canny(mask_white_clean, 50, 150, apertureSize=3)
            
            # Dialte edges to connect broken parts
            edges_white = cv2.dilate(edges_white, np.ones((3,3),np.uint8))
            
            # Standard Hough lines on the edges of white areas
            # VERY STRICT minLineLength to ignore text
            lines_white = cv2.HoughLinesP(edges_white, 1, np.pi/180,
                                        threshold=70, minLineLength=100, maxLineGap=25)
            
            if lines_white is not None:
                lines_white = MapImageProcessor._merge_lines(lines_white, distance_threshold=20)
                for line in lines_white:
                    x1, y1, x2, y2 = line[0]
                    # Filter short again just in case
                    if np.sqrt((x2-x1)**2 + (y2-y1)**2) < 100: continue
                    
                    fx1, fy1 = x1/scale, y1/scale
                    fx2, fy2 = x2/scale, y2/scale
                    add_feature(LineString([(fx1, h_orig - fy1), (fx2, h_orig - fy2)]), FeatureType.LOCAL_ROAD)
                    cv2.line(occupied_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)

            # --- 4. GREEN AREAS (Parks) ---
            lower_green = np.array([35, 40, 100])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
            
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))
            
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_green:
                if cv2.contourArea(cnt) > (w * h * 0.002):
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    poly_coords = []
                    for p in approx:
                        px, py = p[0]
                        poly_coords.append((px/scale, h_orig - py/scale))
                    if len(poly_coords) > 2:
                         add_feature(Polygon(poly_coords), FeatureType.PARK)

            print(f"Extracted {len(features)} features (Scale: {scale:.2f})")
            
            # Demo Fallback ONLY if absolutely nothing found
            if len(features) < 2:
                print("Extraction failed to find enough features. Using demo data.")
                from map_declutter_system import MapGenerator, WKTParser, create_homestead_demo_data
                wkt_file = create_homestead_demo_data()
                wkt_data = WKTParser.parse_file(wkt_file)
                features = MapGenerator.create_from_wkt(wkt_data)

        except Exception as e:
            print(f"Error extracting features: {e}")
            features = []

        return features

def serialize_features(features):
    """Convert MapFeatures to JSON-serializable list of dicts"""
    data = []
    for f in features:
        item = {
            "id": f.id,
            "type": f.type.value,
            "priority": f.style.priority,
        }
        
        if isinstance(f.geometry, LineString):
            item["geometry_type"] = "LineString"
            # shapely coords to list
            item["coords"] = list(f.geometry.coords)
        elif isinstance(f.geometry, Point):
            item["geometry_type"] = "Point"
            item["coords"] = list(f.geometry.coords)
        elif isinstance(f.geometry, Polygon):
            item["geometry_type"] = "Polygon"
            # Polygon exterior coords
            item["coords"] = list(f.geometry.exterior.coords)
            
        data.append(item)
    return data

@app.route('/', methods=['GET'])
def index():
    # Load default demo data (The "Present One")
    from map_declutter_system import create_sample_wkt_file
    
    # Init system
    system = MapDeclutterSystem()
    
    # Create and load sample WKT (streets_ugen.wkt)
    wkt_file = create_sample_wkt_file()
    
    print(f"Loading default demo map from {wkt_file}...")
    if system.load_from_wkt_file(wkt_file):
        # Run the pipeline on the demo data
        system.detect_overlaps()
        system.resolve_overlaps()
        
        # Serialize
        feature_data_list = []
        for f in system.features:
            item = {
                "id": f.id,
                "type": f.type.value,
                "priority": f.style.priority,
            }
            if isinstance(f.geometry, LineString):
                item["geometry_type"] = "LineString"
                item["coords"] = list(f.geometry.coords)
            elif isinstance(f.geometry, Point):
                item["geometry_type"] = "Point"
                item["coords"] = list(f.geometry.coords)
            feature_data_list.append(item)
            
        return render_template('map_app.html', feature_data=feature_data_list, stats=system.stats)
    else:
        return render_template('map_app.html', feature_data=[], stats=None)

from flask_cors import CORS

CORS(app) # Enable CORS for all routes

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Init System
        system = MapDeclutterSystem()
        
        # 2. Extract
        features = MapImageProcessor.process_image(filepath)
        system.features = features
        system._store_original_features()
        
        # 3. Detect & Resolve
        system.detect_overlaps()
        system.resolve_overlaps()
        
        # 4. Serialize Data for JSON response
        feature_data = serialize_features(system.features)
        
        return {
            "features": feature_data,
            "stats": system.stats
        }

@app.route('/process_map', methods=['POST'])
def process_map():
    # Keep existing route for backward compatibility or direct form POST
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 1. Init System
        system = MapDeclutterSystem()
        
        # 2. Extract
        features = MapImageProcessor.process_image(filepath)
        system.features = features
        system._store_original_features()
        
        # 3. Detect & Resolve
        system.detect_overlaps()
        system.resolve_overlaps()
        
        # 4. Serialize Data
        feature_data = serialize_features(system.features)
        
        return render_template('map_app.html', feature_data=feature_data, stats=system.stats)


if __name__ == '__main__':
    print("Starting Map Decluttering Web App...")
    print("Go to http://localhost:5000")
    app.run(debug=True, port=5000)