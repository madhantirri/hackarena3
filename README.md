# Map Decluttering & Feature Extraction System

A comprehensive system that transforms raw map images into clean, structured vector data. It uses Computer Vision to "see" map features and a Physics-based Displacement Engine to automatically resolve spatial conflicts (decluttering).

---

## 🔁 The Full Process: How It Works

The system operates in a linear pipeline consisting of three main phases: **Extraction**, **Modeling**, and **Resolution**.

### Phase 1: Image Processing & Feature Extraction
*Input: Raw Image (JPG/PNG)*
*Output: Raw Geometric Shapes*

The `MapImageProcessor` (in `map_server.py`) analyzes the image to identify distinct map elements.

1.  **Preprocessing**:
    -   **Resizing**: Large images are scaled down (max dimension 1500px) for performance.
    -   **Denoising**: A strong Median Blur (kernel size 7) is applied. This is critical to "melt" away thin text labels and small icons while preserving large colored areas like roads and parks.
    -   **Color Space Conversion**: The image is converted from BGR to HSV (Hue, Saturation, Value) to robustness against lighting changes.

2.  **Feature Detection**:
    -   **Rivers (Blue Masks)**: We threshold for blue hues. We then use **Morphological Closing** to fill gaps and `findContours` to trace the river banks. Only large shapes (>0.5% of image area) are kept.
    -   **Highways (Yellow/Orange Masks)**: We combine masks for yellow and orange. We use **Hough Line Transform** (`HoughLinesP`) to detect straight segments. A custom "Merge Lines" algorithm combines fragmented segments into continuous highways.
    -   **Local Roads (White)**: Detecting white roads is the hardest part. We mask for high brightness/low saturation. We then use **Canny Edge Detection** on this mask to find the boundaries of the roads, followed by Hough Transforms to vectorize them.
    -   **Parks (Green Masks)**: Similar to rivers, we mask for green values and convert the contours into polygons.

### Phase 2: Data Modeling
*Input: Raw Shapes*
*Output: `MapFeature` Objects*

The raw shapes are converted into `MapFeature` objects (in `map_declutter_system.py`).

1.  **Classification**: Each shape is assigned a `FeatureType` (e.g., `HIGHWAY`, `BUILDING`).
2.  **Styling & Priority Assignment**:
    -   **High Priority (Fixed)**: Highways, Rivers. These act as "anchors" and cannot be moved.
    -   **Low Priority (Movable)**: Buildings, Icons, Labels. These are allowed to shift position to avoid overlaps.
    -   **Clearance Rules**: Each type is assigned a `display_width` (how thick it looks) and a `min_clearance` (how much empty space it needs).

### Phase 3: The Decluttering Engine
*Input: Conflicting Map Features*
*Output: Clean, Non-Overlapping Map*

The `DisplacementEngine` solves the puzzle of fitting everything together.

1.  **Spatial Indexing**: 
    -   Instead of checking every feature against every other (O(N²) complexity), we build a **Spatial Grid**. We only check for overlaps between features in the same grid cell.
2.  **Overlap Detection**:
    -   The system calculates the "Buffer Geometry" (actual shape + display width) for every feature.
    -   It identifies "Conflicts" where buffers intersect.
3.  **Physics-Based Resolution (Iterative)**:
    -   The engine enters a loop (default 100 iterations).
    -   For every Conflict:
        -   It calculates a **Repulsion Vector** (direction and magnitude) pushing the lower-priority feature away from the higher-priority one.
    -   **Movement**:
        -   Movable features apply these forces to shift their position.
        -   If a pushed feature hits *another* object, it recalculates and moves again in the next iteration.
    -   **Spiral Search Fallback**:
        -   If a feature is "stuck" (blocked on all sides), the system attempts a spiral search pattern to find the nearest valid open space.

---

## 📌 Project Overview

This project provides a solution for two common mapping challenges:
1.  **Convert Raster to Vector**: extracting meaningful data from pixels.
2.  **Automated Cartography**: Ensuring map elements don't clump together.

The system includes a **Flask-based web interface** used to upload map images, visualize the extracted features, and see the decluttering process in action with real-time statistics.

## ✨ Key Features

-   **Intelligent Feature Extraction**:
    -   Reduces "waste lines" by blurring out text before processing.
    -   Differentiates between highways (warm colors) and local streets (white).
-   **Advanced Decluttering Engine**:
    -   **Priority Hierarchy**: Roads > Rivers > Buildings.
    -   **Conflict Metrics**: Updates stats on overlap area and success rate.
-   **Interactive Web Interface**:
    -   Drag-and-drop image upload.
    -   Side-by-side comparison.

## 🛠️ Tech Stack

-   **Backend**: Python, Flask
-   **Computer Vision**: OpenCV (`cv2`), NumPy
-   **Geometry**: Shapely (for polygon/line math)
-   **Frontend**: HTML5, CSS3, JavaScript

## 🚀 Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install flask flask-cors opencv-python numpy shapely
    ```

2.  **Run the Server**:
    ```bash
    python map_server.py
    ```

3.  **Use the App**:
    Go to `http://localhost:5000` and upload a map image.

## 📂 Project Structure

-   `map_server.py`: The Web Server and Image Processing Logic (Phase 1).
-   `map_declutter_system.py`: The Geometry Engine and Decluttering Logic (Phases 2 & 3).
-   `debug_extraction.py`: A tool to view intermediate computer vision steps (masks, edges).
-   `templates/map_app.html`: The frontend dashboard.

## 🚧 Known Limitations

-   **Text Labels**: The system explicitly tries *not* to read text. It blurs them out. If text is misinterpreted as a road, it results in short "waste lines".
-   **Complex Intersections**: Very dense urban areas may result in merged road blobs rather than distinct lines.
