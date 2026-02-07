<div align="center">

<!-- Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Map%20Decluttering%20System&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Transform%20Raw%20Maps%20into%20Clean%20Vector%20Data&descAlignY=52&descSize=18"/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Leaflet](https://img.shields.io/badge/Leaflet.js-1.9+-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)

<!-- Badges Row 2 -->
![GitHub stars](https://img.shields.io/github/stars/madhantirri/hackarena3?style=for-the-badge&logo=github&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/madhantirri/hackarena3?style=for-the-badge&logo=github&color=blue)
![GitHub issues](https://img.shields.io/github/issues/madhantirri/hackarena3?style=for-the-badge&logo=github&color=red)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<br/>

**🚀 A comprehensive Computer Vision system that transforms raw map images into clean, structured vector data.**

It uses **Computer Vision** to "see" map features and a **Physics-based Displacement Engine** to automatically resolve spatial conflicts (decluttering).

<br/>

<!-- Quick Links -->
[📖 Documentation](#-the-full-process-how-it-works) • 
[🚀 Quick Start](#-installation--usage) • 
[💡 Features](#-key-features) • 
[👥 Team](#-contributors) • 
[🐛 Report Bug](https://github.com/madhantirri/hackarena3/issues)

---

### 🎯 Why This Project?

Maps contain overlapping elements - roads crossing rivers, buildings near highways, labels cluttering important features. This system solves two critical challenges:

| Challenge | Solution |
|-----------|----------|
| **Raster to Vector** | Extract meaningful geometric data from pixel-based map images |
| **Automated Decluttering** | Resolve feature overlaps to ensure visual clarity |

</div>

---

<!-- Table of Contents -->
<details>
<summary><b>📑 Table of Contents</b></summary>

- [🔁 How It Works](#-the-full-process-how-it-works)
  - [Phase 1: Image Processing](#phase-1-image-processing--feature-extraction)
  - [Phase 2: Data Modeling](#phase-2-data-modeling)
  - [Phase 3: Decluttering Engine](#phase-3-the-decluttering-engine)
- [📌 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Installation & Usage](#-installation--usage)
- [📂 Project Structure](#-project-structure)
- [📊 Performance Metrics](#-performance-metrics)
- [👥 Contributors](#-contributors)

</details>

---

## 🔁 The Full Process: How It Works

The system operates in a linear pipeline consisting of three main phases: **Extraction**, **Modeling**, and **Resolution**.

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│     PHASE 1         │      │     PHASE 2         │      │     PHASE 3         │
│  Image Processing   │ ───▶ │   Data Modeling     │ ───▶ │  Decluttering       │
│  & Extraction       │      │                     │      │  Engine             │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
   Input: PNG/JPG              Input: Raw Shapes           Input: MapFeatures
   Output: Geometry            Output: MapFeature          Output: Clean Map
```

<details>
<summary><b>🖼️ See the Process in Action (Click to Expand)</b></summary>

<br/>

| Step | Description | Visual |
|:----:|-------------|:------:|
| **1** | Upload raw map image | 📷 → 🗺️ |
| **2** | Extract features (roads, rivers, parks) | 🔍 Color Detection |
| **3** | Build geometric models | 📐 Shapely Objects |
| **4** | Detect overlapping features | ⚠️ Conflict Detection |
| **5** | Apply physics-based displacement | 💥 Repulsion Vectors |
| **6** | Output clean, decluttered map | ✅ Clean Result |

</details>

---

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

### Priority System

Features are assigned priorities to determine which elements stay fixed and which can move:

| Feature Type | Priority | Movable? | Display Width | Min Clearance |
|-------------|----------|----------|---------------|---------------|
| Highway | 100 | ❌ No | 5.0 | 3.0 |
| Main Road | 80 | ❌ No | 3.5 | 2.5 |
| River | 70 | ❌ No | 3.0 | 2.0 |
| Local Road | 60 | ❌ No | 2.0 | 2.0 |
| Building | 50 | ✅ Yes | 0.0 | 1.0 |
| Icon | 20 | ✅ Yes | 0.0 | 1.0 |
| Label | 10 | ✅ Yes | 0.0 | 1.0 |

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

<div align="center">

| Layer | Technologies |
|:-----:|:-------------|
| **🔧 Backend** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) ![Flask-CORS](https://img.shields.io/badge/Flask--CORS-000000?style=flat&logo=flask&logoColor=white) |
| **👁️ Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |
| **📐 Geometry** | ![Shapely](https://img.shields.io/badge/Shapely-3DDC84?style=flat&logo=python&logoColor=white) |
| **🎨 Frontend** | ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) ![Leaflet](https://img.shields.io/badge/Leaflet.js-199900?style=flat&logo=leaflet&logoColor=white) |
| **📦 Data Formats** | ![JSON](https://img.shields.io/badge/JSON-000000?style=flat&logo=json&logoColor=white) ![WKT](https://img.shields.io/badge/WKT-4A90E2?style=flat) |

</div>

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/madhantirri/hackarena3.git
cd hackarena3

# Install dependencies
pip install flask flask-cors opencv-python numpy shapely

# Run the server
python map_server.py
```

### Use the App

1. Open browser and go to `http://localhost:5000`
2. Click **"Upload Map Photo"** to select a map image (PNG/JPG)
3. Click **"Predict Your Photo"** to process the image
4. View extracted features with interactive layer controls
5. Toggle labels and inspect individual features

### API Endpoint

```http
POST /api/predict
Content-Type: multipart/form-data
```

**Example Request:**
```bash
curl -X POST -F "file=@map_image.png" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "features": [{"id": "hwy_001", "type": "highway", "coords": [...]}],
  "stats": {"total_conflicts": 150, "resolved_conflicts": 142, "success_rate": 94.7}
}
```

## 📂 Project Structure

```
hackarena3/
├── map_server.py              # Flask server + Image Processing (Phase 1)
├── map_declutter_system.py    # Geometry Engine & Decluttering (Phase 2 & 3)
├── map_interactive.html       # Standalone visualization page
├── debug_extraction.py        # Debug tool for CV steps (masks, edges)
├── enhanced_training_data.json # Sample feature data for testing
├── streets_ugen.wkt           # WKT sample map data
├── homestead_simulation.wkt   # Demo simulation data
├── templates/
│   └── map_app.html           # Frontend dashboard template
└── README.md
```

### Key Files Explained

| File | Purpose |
|------|--------|
| `map_server.py` | Main Flask server, handles image upload, runs CV pipeline |
| `map_declutter_system.py` | Core algorithm - 1800+ lines of geometry & physics logic |
| `debug_extraction.py` | Visualize intermediate masks, edges for debugging |

## 🚧 Known Limitations

| Limitation | Description |
|------------|-------------|
| **Text Labels** | System blurs out text. If misinterpreted as road, creates short "waste lines" |
| **Complex Intersections** | Very dense urban areas may result in merged road blobs |
| **Color Dependency** | Optimized for OpenStreetMap-style color schemes |
| **Dense Maps** | May not achieve 100% decluttering on extremely crowded maps |

---

## 📊 Performance Metrics

| Metric | Typical Value |
|--------|---------------|
| Processing Time | 2-5 seconds per image |
| Feature Capacity | 500+ features |
| Conflict Resolution Rate | 60-95% (map dependent) |
| Memory Usage | ~100MB typical |

---

## 👥 Contributors

<div align="center">

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/sanojyadav17">
        <img src="https://github.com/sanojyadav17.png" width="120px;" style="border-radius:50%;" alt="sanojyadav17"/>
      </a><br/>
      <a href="https://github.com/sanojyadav17" style="text-decoration:none;"><b>🎯 Sanoj Yadav</b></a><br/>
      <sub>Team Lead</sub><br/><br/>
      <a href="https://github.com/sanojyadav17"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/madhantirri">
        <img src="https://ui-avatars.com/api/?name=Madhan+Tirri&size=120&background=FF6B6B&color=fff&rounded=true&bold=true" width="120px;" style="border-radius:50%;" alt="madhantirri"/>
      </a><br/>
      <a href="https://github.com/madhantirri" style="text-decoration:none;"><b>💻 Madhan Tirri</b></a><br/>
      <sub>Developer</sub><br/><br/>
      <a href="https://github.com/madhantirri"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/></a>
    </td>
    <td align="center">
      <a href="https://github.com/goyaltanisha447">
        <img src="https://ui-avatars.com/api/?name=Tanisha+Goyal&size=120&background=6C63FF&color=fff&rounded=true&bold=true" width="120px;" style="border-radius:50%;" alt="goyaltanisha447"/>
      </a><br/>
      <a href="https://github.com/goyaltanisha447" style="text-decoration:none;"><b>🎨 Tanisha Goyal</b></a><br/>
      <sub>Developer</sub><br/><br/>
      <a href="https://github.com/goyaltanisha447"><img src="https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white"/></a>
    </td>
  </tr>
</table>

</div>

---

## 🤝 Contributing

Contributions welcome! Feel free to submit issues and pull requests.

```bash
# Fork and clone
git checkout -b feature/improvement
git commit -am 'Add new feature'
git push origin feature/improvement
# Open Pull Request
```

---

<div align="center">

### 🏆 Built for HackArena 3.0

<table>
<tr>
<td align="center">

[![⭐ Star This Repo](https://img.shields.io/badge/⭐_Star_This_Repo-yellow?style=for-the-badge&logo=github)](https://github.com/madhantirri/hackarena3)

</td>
<td align="center">

[![🍴 Fork This Repo](https://img.shields.io/badge/🍴_Fork_This_Repo-blue?style=for-the-badge&logo=github)](https://github.com/madhantirri/hackarena3/fork)

</td>
<td align="center">

[![🐛 Report Bug](https://img.shields.io/badge/🐛_Report_Bug-red?style=for-the-badge&logo=github)](https://github.com/madhantirri/hackarena3/issues/new)

</td>
</tr>
<tr>
<td align="center">

[![💡 Request Feature](https://img.shields.io/badge/💡_Request_Feature-green?style=for-the-badge&logo=github)](https://github.com/madhantirri/hackarena3/issues/new?labels=enhancement)

</td>
<td align="center">

[![📧 Contact Team](https://img.shields.io/badge/📧_Contact_Team-purple?style=for-the-badge&logo=gmail)](mailto:sanojyadav2700@gmail.com,Madhan.tirri.1323@gmail.com,goyaltanisha447@gmail.com?subject=HackArena3%20Project%20Inquiry)

</td>
</tr>
</table>

<br/>

**Made with ❤️ by Team Trident**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

</div>
