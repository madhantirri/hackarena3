"""
MAP DECLUTTERING SYSTEM - Hackathon Solution
A modular system for overlap detection and priority-based displacement
"""

import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import numpy as np

# ============================================================================
# DATA MODELS
# ============================================================================

class FeatureType(Enum):
    """Types of map features with their display characteristics"""
    HIGHWAY = "highway"
    MAIN_ROAD = "main_road"
    LOCAL_ROAD = "local_road"
    RIVER = "river"
    BUILDING = "building"
    LABEL = "label"
    ICON = "icon"
    PARK = "park"
    RAILWAY = "railway"


@dataclass
class FeatureStyle:
    """Styling and priority properties for map features"""
    display_width: float  # Display width in points
    color: str           # Display color
    priority: int        # Higher number = higher priority
    z_index: int         # Rendering order
    is_movable: bool     # Can this feature be moved?
    min_clearance: float = 2.0  # Minimum distance from other features


@dataclass
class MapFeature:
    """Represents a single map feature with geometry and properties"""
    id: str
    type: FeatureType
    geometry: any  # Shapely geometry object
    style: FeatureStyle
    original_geometry: any = None
    displacement_vector: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.original_geometry is None:
            self.original_geometry = self.geometry
    
    @property
    def buffer_geometry(self):
        """Get geometry buffered by display width for overlap detection"""
        try:
            if self.type in [FeatureType.HIGHWAY, FeatureType.MAIN_ROAD, 
                            FeatureType.LOCAL_ROAD, FeatureType.RIVER, FeatureType.RAILWAY]:
                return self.geometry.buffer(self.style.display_width / 2)
            return self.geometry
        except Exception:
            return self.geometry
    
    @property
    def bounds(self):
        """Get bounding box of the buffered geometry"""
        try:
            return self.buffer_geometry.bounds
        except Exception:
            if hasattr(self.geometry, 'bounds'):
                return self.geometry.bounds
            return (0, 0, 0, 0)
    
    def move(self, dx: float, dy: float) -> bool:
        """Move feature by specified displacement"""
        if self.style.is_movable:
            self.displacement_vector = (dx, dy)
            from shapely.ops import transform
            self.geometry = transform(lambda x, y: (x + dx, y + dy), self.geometry)
            return True
        return False
    
    def reset_position(self):
        """Reset to original position"""
        if self.style.is_movable:
            self.geometry = self.original_geometry
            self.displacement_vector = (0.0, 0.0)
    
    def __hash__(self):
        """Make MapFeature hashable by hashing its ID"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality comparison based on ID"""
        if not isinstance(other, MapFeature):
            return False
        return self.id == other.id


# ============================================================================
# CONFIGURATION MANAGERS
# ============================================================================

class StyleManager:
    """Manages styling rules for different feature types"""
    
    STYLE_CONFIG = {
        FeatureType.HIGHWAY: FeatureStyle(
            display_width=5.0,
            color="#E74C3C",  # Red
            priority=100,
            z_index=10,
            is_movable=False,
            min_clearance=3.0
        ),
        FeatureType.MAIN_ROAD: FeatureStyle(
            display_width=3.5,
            color="#F39C12",  # Orange
            priority=80,
            z_index=9,
            is_movable=False,
            min_clearance=2.5
        ),
        FeatureType.LOCAL_ROAD: FeatureStyle(
            display_width=2.0,
            color="#95A5A6",  # Gray
            priority=60,
            z_index=8,
            is_movable=False,
            min_clearance=2.0
        ),
        FeatureType.RIVER: FeatureStyle(
            display_width=3.0,
            color="#3498DB",  # Blue
            priority=70,
            z_index=7,
            is_movable=False,
            min_clearance=2.0
        ),
        FeatureType.RAILWAY: FeatureStyle(
            display_width=2.5,
            color="#8E44AD",  # Purple
            priority=75,
            z_index=6,
            is_movable=False,
            min_clearance=2.5
        ),
        FeatureType.BUILDING: FeatureStyle(
            display_width=0.0,
            color="#27AE60",  # Green
            priority=50,
            z_index=5,
            is_movable=True,
            min_clearance=1.0
        ),
        FeatureType.PARK: FeatureStyle(
            display_width=0.0,
            color="#2ECC71",  # Light Green
            priority=40,
            z_index=4,
            is_movable=True,
            min_clearance=1.5
        ),
        FeatureType.LABEL: FeatureStyle(
            display_width=0.0,
            color="#2C3E50",  # Dark Blue
            priority=10,
            z_index=3,
            is_movable=True,
            min_clearance=1.0
        ),
        FeatureType.ICON: FeatureStyle(
            display_width=0.0,
            color="#9B59B6",  # Purple
            priority=20,
            z_index=2,
            is_movable=True,
            min_clearance=1.0
        )
    }
    
    @classmethod
    def get_style(cls, feature_type: FeatureType) -> FeatureStyle:
        """Get style configuration for a feature type"""
        return cls.STYLE_CONFIG.get(feature_type, FeatureStyle(
            display_width=1.0,
            color="#000000",
            priority=0,
            z_index=0,
            is_movable=True,
            min_clearance=0.5
        ))


# ============================================================================
# CORE ENGINE: OVERLAP DETECTOR
# ============================================================================

class OverlapDetector:
    """
    Detects overlaps between map features considering their display widths.
    Uses spatial grid indexing for performance.
    """
    
    def __init__(self, grid_size: float = 50.0):
        self.grid_size = grid_size
        self.grid: Dict[Tuple[int, int], List[MapFeature]] = {}
    
    def detect_all_overlaps(self, features: List[MapFeature]) -> List[Dict]:
        """
        Detect all overlaps between features.
        Returns a list of overlap information dictionaries.
        """
        overlaps = []
        
        # Build spatial index for faster lookup
        self._build_spatial_grid(features)
        
        # Check each feature against others in nearby grid cells
        seen_pairs = set()
        
        for i, feat1 in enumerate(features):
            # Get nearby features using spatial grid
            nearby_features = self._get_nearby_features(feat1)
            
            for feat2 in nearby_features:
                if feat1.id == feat2.id:
                    continue
                
                # Avoid duplicates by sorting IDs
                pair_id = tuple(sorted((feat1.id, feat2.id)))
                if pair_id in seen_pairs:
                    continue
                seen_pairs.add(pair_id)
                
                # Check for overlap
                overlap_info = self._check_pair_overlap(feat1, feat2)
                if overlap_info["overlaps"]:
                    overlaps.append(overlap_info)
        
        return overlaps
    
    def _build_spatial_grid(self, features: List[MapFeature]):
        """Build a spatial grid index for efficient overlap detection"""
        self.grid = {}
        
        for feature in features:
            bounds = feature.bounds
            if bounds:
                minx, miny, maxx, maxy = bounds
                
                # Determine which grid cells the feature occupies
                x_cells = range(int(minx // self.grid_size), int(maxx // self.grid_size) + 1)
                y_cells = range(int(miny // self.grid_size), int(maxy // self.grid_size) + 1)
                
                for x in x_cells:
                    for y in y_cells:
                        cell = (x, y)
                        if cell not in self.grid:
                            self.grid[cell] = []
                        self.grid[cell].append(feature)
    
    def _get_nearby_features(self, feature: MapFeature) -> List[MapFeature]:
        """Get features from nearby grid cells (current cell + neighbors)"""
        bounds = feature.bounds
        if not bounds:
            return []
        
        minx, miny, maxx, maxy = bounds
        
        # Expand search area by 1 cell in each direction
        x_cells = range(int(minx // self.grid_size) - 1, 
                       int(maxx // self.grid_size) + 2)
        y_cells = range(int(miny // self.grid_size) - 1,
                       int(maxy // self.grid_size) + 2)
        
        nearby = []
        seen_ids = set()  # Track seen feature IDs to avoid duplicates
        
        for x in x_cells:
            for y in y_cells:
                cell = (x, y)
                if cell in self.grid:
                    for feat in self.grid[cell]:
                        if feat.id not in seen_ids:
                            nearby.append(feat)
                            seen_ids.add(feat.id)
        
        return nearby
    
    def _check_pair_overlap(self, feat1: MapFeature, feat2: MapFeature) -> Dict:
        """
        Check for overlap between two specific features.
        Returns detailed overlap information.
        """
        try:
            buffer1 = feat1.buffer_geometry
            buffer2 = feat2.buffer_geometry
            
            if not buffer1.intersects(buffer2):
                return self._create_no_overlap_result(feat1, feat2)
            
            # Calculate intersection area
            intersection = buffer1.intersection(buffer2)
            overlap_area = intersection.area
            
            # Calculate distance between original geometries
            distance = feat1.geometry.distance(feat2.geometry)
            
            # Calculate required clearance based on display widths and min clearance
            required_clearance = (feat1.style.display_width + feat2.style.display_width) / 2
            required_clearance += max(feat1.style.min_clearance, feat2.style.min_clearance)
            
            clearance_violation = max(0, required_clearance - distance)
            
            return {
                "overlaps": True,
                "feature1": feat1.id,
                "feature2": feat2.id,
                "type1": feat1.type.value,
                "type2": feat2.type.value,
                "priority1": feat1.style.priority,
                "priority2": feat2.style.priority,
                "overlap_area": overlap_area,
                "distance": distance,
                "required_clearance": required_clearance,
                "clearance_violation": clearance_violation,
                "intersection_geom": intersection
            }
        except Exception as e:
            print(f"Warning: Error checking overlap between {feat1.id} and {feat2.id}: {e}")
            return self._create_no_overlap_result(feat1, feat2)
    
    def _create_no_overlap_result(self, feat1: MapFeature, feat2: MapFeature) -> Dict:
        """Create a result dictionary for non-overlapping features"""
        return {
            "overlaps": False,
            "feature1": feat1.id,
            "feature2": feat2.id,
            "overlap_area": 0.0,
            "overlap_distance": 0.0,
            "clearance_violation": 0.0
        }


# ============================================================================
# CORE ENGINE: DISPLACEMENT ENGINE
# ============================================================================

class DisplacementEngine:
    """
    Moves lower-priority features away from higher-priority ones
    based on priority rules and overlap severity.
    """
    
    def __init__(self, max_iterations: int = 100, step_size: float = 1.0):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.displacement_history = []
    
    def resolve_overlaps(self, features: List[MapFeature], 
                        overlaps: List[Dict]) -> Tuple[List[MapFeature], Dict]:
        """
        Main method to resolve all detected overlaps.
        Returns updated features and statistics.
        """
        # Separate features by mobility
        fixed_features = [f for f in features if not f.style.is_movable]
        movable_features = [f for f in features if f.style.is_movable]
        
        # Sort movable features by priority (lower priority first)
        movable_features.sort(key=lambda f: f.style.priority)
        
        # Filter overlaps to only those that can be resolved (at least one movable feature)
        # This provides a more accurate success rate metric
        movable_ids = {f.id for f in movable_features}
        resolvable_overlaps = [
            ov for ov in overlaps 
            if ov['feature1'] in movable_ids or ov['feature2'] in movable_ids
        ]
        
        # Initialize statistics
        stats = self._initialize_stats(resolvable_overlaps, movable_features)
        stats['all_detected_conflicts'] = len(overlaps)
        
        # Iterative displacement
        print(f"DEBUG: Starting resolution with max_iterations={self.max_iterations}")
        for iteration in range(self.max_iterations):
            moves_this_iteration = 0
            
            # Process each movable feature
            for movable in movable_features:
                # Find ACTUAL current conflicts dynamically
                # (The initial 'overlaps' list is stale after moves)
                conflicts = self._find_conflicts_for_feature(movable, features)
                
                if conflicts:
                    # Try to displace the feature
                    resolved = self._displace_single_feature(movable, features, conflicts)
                    
                    if resolved:
                        moves_this_iteration += 1
                        stats["features_moved"] += 1
                        stats["resolved_conflicts"] += len(conflicts) # Approximate
                        dx, dy = movable.displacement_vector
                        stats["total_displacement"] += math.sqrt(dx*dx + dy*dy)
            
            stats["iterations"] = iteration + 1
            if moves_this_iteration == 0:
                break
        
        # Calculate success rate
        # We need to do a final check of overlaps to get accurate success rate
        # But lacking the detector, we'll rely on the caller to check final overlaps
        # For internal stats, we can just say:
        stats["success_rate"] = self._calculate_success_rate(stats)
        
        return features, stats
    
    def _find_conflicts_for_feature(self, feature: MapFeature, 
                                  all_features: List[MapFeature]) -> List[Dict]:
        """Find all features currently conflicting with the given feature"""
        conflicts = []
        
        for other in all_features:
            if other.id == feature.id:
                continue
            
            # Strict Priority Rule:
            # A feature only moves to avoid features of EQUAL or HIGHER priority.
            # It should ignore lower priority features (which are expected to move).
            if other.style.priority < feature.style.priority:
                continue
            
            try:
                # Quick check
                if not feature.buffer_geometry.intersects(other.buffer_geometry):
                    continue
                
                # Detailed check
                distance = feature.geometry.distance(other.geometry)
                required_clearance = (feature.style.display_width + other.style.display_width) / 2
                required_clearance += max(feature.style.min_clearance, other.style.min_clearance)
                
                if distance < required_clearance:
                    conflicts.append({
                        "feature1": other.id,
                        "feature2": feature.id,
                        "clearance_violation": required_clearance - distance,
                        "required_clearance": required_clearance
                    })
            except Exception:
                continue
                
                
        return conflicts
    
    def _initialize_stats(self, overlaps: List[Dict], movable_features: List[MapFeature]) -> Dict:
        """Initialize statistics dictionary"""
        return {
            "total_conflicts": len(overlaps),
            "resolved_conflicts": 0,
            "movable_features": len(movable_features),
            "features_moved": 0,
            "total_displacement": 0.0,
            "iterations": 0
        }
    
    def _calculate_success_rate(self, stats: Dict) -> float:
        """Calculate percentage of conflicts resolved"""
        if stats["total_conflicts"] > 0:
            return (stats["resolved_conflicts"] / stats["total_conflicts"]) * 100
        return 100.0
    
    def _displace_single_feature(self, movable: MapFeature, 
                                all_features: List[MapFeature],
                                conflicts: List[Dict]) -> bool:
        """
        Find optimal displacement for a single feature.
        Returns True if displacement was successful.
        """
        # Calculate combined repulsion vector from all conflicts
        total_dx, total_dy = 0.0, 0.0
        
        for conflict in conflicts:
            other_feat = next((f for f in all_features if f.id == conflict["feature1"]), None)
            if not other_feat:
                continue
            
            # Calculate repulsion vector for this conflict
            dx, dy = self._calculate_repulsion_vector(movable, other_feat, conflict)
            total_dx += dx
            total_dy += dy
        
        # Normalize and apply displacement
        if total_dx != 0 or total_dy != 0:
            return self._apply_displacement(movable, all_features, total_dx, total_dy)
        else:
            # Forces cancelled out, try random search
            return self._try_alternative_positions(movable, all_features)
        
        return False
    
    def _calculate_repulsion_vector(self, movable: MapFeature, 
                                   fixed: MapFeature, 
                                   conflict: Dict) -> Tuple[float, float]:
        """Calculate direction and magnitude of repulsion between features"""
        # Get centroids of both features
        try:
            movable_center = movable.geometry.centroid
            fixed_center = fixed.geometry.centroid
            
            # Vector from fixed to movable feature
            dx = movable_center.x - fixed_center.x
            dy = movable_center.y - fixed_center.y
            
            distance = math.sqrt(dx*dx + dy*dy)
            if distance == 0:
                return (1.0, 0.0)  # Arbitrary direction if coincident
            
            # Calculate required separation with buffer
            required_distance = conflict["required_clearance"] + 1.0
            
            # Scale vector based on overlap severity
            overlap_factor = 1.0 + conflict["clearance_violation"] / max(distance, 1.0)
            scale = required_distance * overlap_factor / distance
            
            return (dx * scale, dy * scale)
        except Exception:
            return (0.0, 0.0)
    
    def _apply_displacement(self, movable: MapFeature, 
                           all_features: List[MapFeature],
                           total_dx: float, total_dy: float) -> bool:
        """Apply displacement and check for new conflicts"""
        magnitude = math.sqrt(total_dx*total_dx + total_dy*total_dy)
        
        # Scale displacement
        scale = min(self.step_size * 5, magnitude) / magnitude
        displacement = (total_dx * scale, total_dy * scale)
        
        # Try the displacement
        original_pos = movable.geometry
        movable.move(*displacement)
        
        # Check if new position creates new conflicts
        if not self._has_conflicts(movable, all_features):
            return True
        else:
            # Try alternative positions
            movable.reset_position()
            return self._try_alternative_positions(movable, all_features)
    
    def _try_alternative_positions(self, movable: MapFeature,
                                 all_features: List[MapFeature]) -> bool:
        """Try alternative displacement directions and distances"""
        # Try alternative displacement directions and distances
        # Candidate directions (36 directions around for better precision/prediction)
        import math
        directions = []
        for i in range(36):
            angle = i * (2 * math.pi / 36)
            directions.append((math.cos(angle), math.sin(angle)))
        
        # Try increasing distances with more granularity and larger range
        # Micro-adjustments (.1-.5) up to massive jumps (500) to guarantee finding space
        for distance_mult in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                             1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 
                             7.5, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0]:
            for dx, dy in directions:
                displacement = (dx * self.step_size * distance_mult,
                              dy * self.step_size * distance_mult)
                
                movable.move(*displacement)
                
                if not self._has_conflicts(movable, all_features):
                    return True
                
                movable.reset_position()
        
        return False
    
    def _has_conflicts(self, feature: MapFeature, 
                      other_features: List[MapFeature]) -> bool:
        """Check if a feature has conflicts in its current position"""
        for other in other_features:
            if other.id == feature.id:
                continue
            
            # Quick bounding box check first
            if not feature.buffer_geometry.intersects(other.buffer_geometry):
                continue
            
            # Detailed clearance check
            try:
                distance = feature.geometry.distance(other.geometry)
                required_clearance = (feature.style.display_width + other.style.display_width) / 2
                required_clearance += max(feature.style.min_clearance, other.style.min_clearance)
                
                if distance < required_clearance:
                    return True
            except Exception:
                continue
        
        return False


# ============================================================================
# DATA PROCESSING
# ============================================================================

class WKTParser:
    """Parser for WKT (Well-Known Text) format files"""
    
    @staticmethod
    def parse_file(file_path: str) -> List[Dict]:
        """Parse WKT file and return list of geometries"""
        with open(file_path, 'r') as f:
            content = f.read()
        return WKTParser.parse_string(content)
    
    @staticmethod
    def parse_string(wkt_string: str) -> List[Dict]:
        """Parse WKT string into geometry dictionaries"""
        features = []
        lines = wkt_string.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Parse different geometry types
            if line.startswith('LINESTRING'):
                features.extend(WKTParser._parse_linestring(line, i))
        
        return features
    
    @staticmethod
    def _parse_linestring(line: str, index: int) -> List[Dict]:
        """Parse LINESTRING geometry"""
        from shapely.geometry import LineString
        
        try:
            # Extract coordinates between parentheses
            start = line.find('(')
            end = line.rfind(')')
            if start == -1 or end == -1:
                return []
            
            coord_str = line[start+1:end]
            coords = WKTParser._parse_coordinates(coord_str)
            
            if len(coords) >= 2:
                geometry = LineString(coords)
                return [{
                    'id': f"road_{index:03d}",
                    'geometry': geometry,
                    'type': 'LINESTRING'
                }]
        except Exception as e:
            print(f"Warning: Could not parse LINESTRING at line {index}: {e}")
        
        return []
    
    @staticmethod
    def _parse_coordinates(coord_str: str) -> List[Tuple[float, float]]:
        """Parse coordinate string into list of (x, y) tuples"""
        coords = []
        for point_str in coord_str.split(','):
            point_str = point_str.strip()
            if point_str:
                parts = point_str.split()
                if len(parts) >= 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        coords.append((x, y))
                    except ValueError:
                        # Skip invalid coordinates
                        continue
        return coords


class MapGenerator:
    """Generates realistic map features from parsed geometry data"""
    
    @staticmethod
    def create_from_wkt(wkt_data: List[Dict]) -> List[MapFeature]:
        """Create map features from parsed WKT data"""
        features = []
        
        # Process roads from LINESTRING data
        features.extend(MapGenerator._create_roads(wkt_data))
        
        # Add rivers (convert some roads to rivers)
        features.extend(MapGenerator._create_rivers(features))
        
        # Add buildings near roads (simpler approach)
        features.extend(MapGenerator._create_simple_buildings(features))
        
        # Add labels at important points
        features.extend(MapGenerator._create_labels(features))
        
        # Add icons (POIs)
        features.extend(MapGenerator._create_icons(features))
        
        return features
    
    @staticmethod
    def _create_roads(wkt_data: List[Dict]) -> List[MapFeature]:
        """Create road features from LINESTRING geometries"""
        features = []
        road_count = 0
        
        for item in wkt_data:
            if item['type'] == 'LINESTRING':
                geom = item['geometry']
                
                # Determine road type based on length
                length = geom.length if hasattr(geom, 'length') else 100
                
                if length > 500:
                    road_type = FeatureType.HIGHWAY
                elif length > 250:
                    road_type = FeatureType.MAIN_ROAD
                else:
                    road_type = FeatureType.LOCAL_ROAD
                
                style = StyleManager.get_style(road_type)
                
                features.append(MapFeature(
                    id=f"{road_type.value}_{road_count:03d}",
                    type=road_type,
                    geometry=geom,
                    style=style,
                    metadata={'length': length}
                ))
                road_count += 1
        
        return features
    
    @staticmethod
    def _create_rivers(existing_features: List[MapFeature]) -> List[MapFeature]:
        """Create river features"""
        river_features = []
        river_count = 0
        
        # Convert some roads to rivers for demonstration
        for i, feat in enumerate(existing_features[:3]):
            river_feat = MapFeature(
                id=f"river_{river_count:03d}",
                type=FeatureType.RIVER,
                geometry=feat.geometry,
                style=StyleManager.get_style(FeatureType.RIVER)
            )
            river_features.append(river_feat)
            river_count += 1
        
        return river_features
    
    @staticmethod
    def _create_simple_buildings(existing_features: List[MapFeature]) -> List[MapFeature]:
        """Create simple building features as points near roads"""
        from shapely.geometry import Point
        
        building_features = []
        
        for i in range(200):  # Create 200 buildings for high density
            # Find a road to place building near
            road_idx = i % len(existing_features)
            road = existing_features[road_idx]
            
            try:
                if hasattr(road.geometry, 'coords'):
                    coords = list(road.geometry.coords)
                    if coords:
                        mid_idx = len(coords) // 2
                        x, y = coords[mid_idx]
                        
                        # Offset slightly from road
                        x += np.random.uniform(-20, 20)
                        y += np.random.uniform(-20, 20)
                        
                        # Create building as a point (simplified)
                        building = Point(x, y)
                        
                        building_features.append(MapFeature(
                            id=f"building_{i:03d}",
                            type=FeatureType.BUILDING,
                            geometry=building,
                            style=StyleManager.get_style(FeatureType.BUILDING)
                        ))
            except Exception as e:
                print(f"Warning: Could not create building {i}: {e}")
                continue
        
        return building_features
    
    @staticmethod
    def _create_labels(existing_features: List[MapFeature]) -> List[MapFeature]:
        """Create label features at road intersections"""
        from shapely.geometry import Point
        
        label_features = []
        
        for i in range(100):  # Create 100 labels to force overlaps
            # Find a road for label placement
            road_idx = i % len(existing_features)
            road = existing_features[road_idx]
            
            try:
                if hasattr(road.geometry, 'coords'):
                    coords = list(road.geometry.coords)
                    if coords:
                        mid_idx = len(coords) // 3  # Use 1/3 point for variety
                        x, y = coords[mid_idx]
                        
                        # Create label point
                        label = Point(x, y)
                        
                        label_features.append(MapFeature(
                            id=f"label_{i:03d}",
                            type=FeatureType.LABEL,
                            geometry=label,
                            style=StyleManager.get_style(FeatureType.LABEL),
                            metadata={'text': f"Label {i+1}"}
                        ))
            except Exception as e:
                print(f"Warning: Could not create label {i}: {e}")
                continue
        
        return label_features
    
    @staticmethod
    def _create_icons(existing_features: List[MapFeature]) -> List[MapFeature]:
        """Create icon features (POIs) at random locations"""
        from shapely.geometry import Point
        
        icon_features = []
        
        # Collect all coordinates for bounding box calculation
        all_coords = []
        for feat in existing_features:
            try:
                if hasattr(feat.geometry, 'coords'):
                    all_coords.extend(list(feat.geometry.coords))
                elif hasattr(feat.geometry, 'x'):
                    all_coords.append((feat.geometry.x, feat.geometry.y))
            except Exception:
                continue
        
        if all_coords:
            min_x = min(c[0] for c in all_coords)
            max_x = max(c[0] for c in all_coords)
            min_y = min(c[1] for c in all_coords)
            max_y = max(c[1] for c in all_coords)
            
            for i in range(100):  # Create 100 icons
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                
                icon = Point(x, y)
                
                icon_types = ['hospital', 'parking', 'school', 'mall']
                icon_type = icon_types[i % len(icon_types)]
                
                icon_features.append(MapFeature(
                    id=f"icon_{i:03d}",
                    type=FeatureType.ICON,
                    geometry=icon,
                    style=StyleManager.get_style(FeatureType.ICON),
                    metadata={'type': icon_type}
                ))
        
        return icon_features


# ============================================================================
# VISUALIZATION
# ============================================================================

class MapVisualizer:
    """
    Creates visualizations of map features before and after decluttering.
    Shows overlaps, displacements, and statistics.
    """
    
    def __init__(self, figsize=(16, 10)):
        self.figsize = figsize
        self.color_map = {
            FeatureType.HIGHWAY: "#E74C3C",
            FeatureType.MAIN_ROAD: "#F39C12",
            FeatureType.LOCAL_ROAD: "#95A5A6",
            FeatureType.RIVER: "#3498DB",
            FeatureType.RAILWAY: "#8E44AD",
            FeatureType.BUILDING: "#27AE60",
            FeatureType.PARK: "#2ECC71",
            FeatureType.LABEL: "#2C3E50",
            FeatureType.ICON: "#9B59B6"
        }
    
    def create_comparison_view(self, features_before: List[MapFeature],
                              features_after: List[MapFeature],
                              overlaps: List[Dict],
                              stats: Dict,
                              output_path: str = None):
        """
        Create side-by-side before/after visualization.
        
        Args:
            features_before: Original features
            features_after: Features after displacement
            overlaps: Detected overlaps
            stats: Processing statistics
            output_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=self.figsize)
        
        # Before (left panel)
        ax1 = plt.subplot(1, 2, 1)
        self._draw_map(ax1, features_before, overlaps, "BEFORE: Original Map")
        
        # After (right panel)
        ax2 = plt.subplot(1, 2, 2)
        self._draw_map(ax2, features_after, [], "AFTER: After Decluttering", 
                      show_displacement=True)
        
        # Add statistics and legend
        self._add_statistics_overlay(fig, stats)
        self._add_legend(fig)
        
        plt.suptitle("Map Decluttering System - Priority-Based Displacement", 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
        
        plt.show()
        return fig
    
    def _draw_map(self, ax, features: List[MapFeature], 
                 overlaps: List[Dict], title: str,
                 show_displacement: bool = False):
        """Draw a map on specified axes"""
        import matplotlib.pyplot as plt
        
        # Sort features by z-index for proper layering
        features = sorted(features, key=lambda f: f.style.z_index)
        
        # Draw each feature
        for feature in features:
            self._draw_single_feature(ax, feature, show_displacement)
        
        # Highlight overlap areas
        for overlap in overlaps:
            if 'intersection_geom' in overlap and overlap['intersection_geom']:
                try:
                    if hasattr(overlap['intersection_geom'], 'exterior'):
                        x, y = overlap['intersection_geom'].exterior.xy
                        ax.fill(x, y, alpha=0.3, color='red')
                except Exception:
                    pass
        
        # Set plot limits with padding
        self._set_plot_limits(ax, features)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
    
    def _draw_single_feature(self, ax, feature: MapFeature, show_displacement: bool = False):
        """Draw a single map feature"""
        color = self.color_map.get(feature.type, "#000000")
        
        try:
            if hasattr(feature.geometry, 'xy'):  # LineString
                x, y = feature.geometry.xy
                linewidth = max(0.5, feature.style.display_width * 0.3)
                
                ax.plot(x, y, color=color, linewidth=linewidth, 
                       alpha=0.8, solid_capstyle='round')
                
                # Add label at midpoint (limit to only 10% of roads to avoid clutter)
                if len(x) > 1 and hash(feature.id) % 10 == 0:
                    mid_idx = len(x) // 2
                    ax.text(x[mid_idx], y[mid_idx], feature.id[:8],
                           fontsize=6, ha='center', va='center',
                           fontweight='bold', color='#444444',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor="white", alpha=0.9,
                                   edgecolor="#cccccc", linewidth=0.5))
            
            elif hasattr(feature.geometry, 'x'):  # Point
                marker = 'o' if feature.type == FeatureType.ICON else 's'
                size = 40 if feature.type == FeatureType.ICON else 30
                
                ax.scatter(feature.geometry.x, feature.geometry.y,
                          c=color, s=size, marker=marker, alpha=0.9,
                          edgecolors='white', linewidth=1.0, zorder=5)
                
                # Stylish label for points
                label_y_offset = 120 if feature.type == FeatureType.LABEL else 80
                ax.text(feature.geometry.x, feature.geometry.y + label_y_offset,
                       feature.id, fontsize=7, ha='center',
                       fontweight='bold', color='#333333',
                       bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white", alpha=0.85, 
                                edgecolor="none"))
            
            elif hasattr(feature.geometry, 'exterior'):  # Polygon
                x, y = feature.geometry.exterior.xy
                ax.fill(x, y, alpha=0.4, color=color, edgecolor=color, linewidth=1)
                
                # Add label at centroid
                centroid = feature.geometry.centroid
                ax.text(centroid.x, centroid.y, feature.id,
                       fontsize=6, ha='center', va='center')
            
            # Show displacement vector if moved
            if (show_displacement and feature.style.is_movable and 
                feature.displacement_vector != (0, 0)):
                dx, dy = feature.displacement_vector
                if dx != 0 or dy != 0 and hasattr(feature.original_geometry, 'x'):
                    start = (feature.original_geometry.x, feature.original_geometry.y)
                    end = (feature.geometry.x, feature.geometry.y)
                    
                    ax.annotate('', xy=end, xytext=start,
                              arrowprops=dict(arrowstyle='->', 
                                            color='gray', 
                                            lw=1, 
                                            alpha=0.6))
        except Exception as e:
            print(f"Warning: Could not draw feature {feature.id}: {e}")
    
    def _set_plot_limits(self, ax, features: List[MapFeature]):
        """Set appropriate plot limits for the features"""
        all_bounds = []
        for f in features:
            try:
                bounds = f.bounds
                if bounds and len(bounds) == 4:
                    all_bounds.append(bounds)
            except Exception:
                continue
        
        if all_bounds:
            minx = min(b[0] for b in all_bounds)
            miny = min(b[1] for b in all_bounds)
            maxx = max(b[2] for b in all_bounds)
            maxy = max(b[3] for b in all_bounds)
            
            # Add 10% padding
            width = maxx - minx
            height = maxy - miny
            padding = max(width * 0.1, height * 0.1, 10)
            
            ax.set_xlim(minx - padding, maxx + padding)
            ax.set_ylim(miny - padding, maxy + padding)
    
    def _add_statistics_overlay(self, fig, stats: Dict):
        """Add statistics text overlay to the figure"""
        import matplotlib.pyplot as plt
        
        stats_text = (
            f"STATISTICS\n"
            f"──────────\n"
            f"Total Features: {stats.get('total_features', 0)}\n"
            f"Fixed Features: {stats.get('fixed_features', 0)}\n"
            f"Movable Features: {stats.get('movable_features', 0)}\n"
            f"Conflicts Detected: {stats.get('total_conflicts', 0)}\n"
            f"Conflicts Resolved: {stats.get('resolved_conflicts', 0)}\n"
            f"Features Moved: {stats.get('features_moved', 0)}\n"
            f"Success Rate: {stats.get('success_rate', 0):.1f}%\n"
            f"Total Displacement: {stats.get('total_displacement', 0):.1f} units"
        )
        
        # Create a separate axes for statistics
        stat_ax = fig.add_axes([0.02, 0.02, 0.25, 0.25])
        stat_ax.axis('off')
        stat_ax.text(0.05, 0.95, stats_text, transform=stat_ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor="lightyellow", 
                            alpha=0.9,
                            edgecolor="gold"))
    
    def _add_legend(self, fig):
        """Add comprehensive legend to the figure"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        legend_elements = []
        
        for feature_type, color in self.color_map.items():
            label = feature_type.value.replace('_', ' ').title()
            
            if feature_type in [FeatureType.HIGHWAY, FeatureType.MAIN_ROAD, 
                              FeatureType.LOCAL_ROAD, FeatureType.RIVER, FeatureType.RAILWAY]:
                width = StyleManager.get_style(feature_type).display_width
                legend_elements.append(
                    mpatches.Patch(color=color, label=f"{label} ({width}pt)")
                )
            elif feature_type in [FeatureType.BUILDING, FeatureType.PARK]:
                legend_elements.append(
                    mpatches.Patch(color=color, alpha=0.4, label=label)
                )
            else:
                legend_elements.append(
                    mpatches.Patch(color=color, label=label)
                )
        
        # Add overlap indicator
        legend_elements.append(
            mpatches.Patch(color='red', alpha=0.3, label='Overlap Area')
        )
        
        # Add displacement indicator
        legend_elements.append(
            mpatches.Arrow(0, 0, 1, 0, color='gray', alpha=0.6, label='Displacement')
        )
        
        fig.legend(handles=legend_elements, loc='lower center', 
                  ncol=5, frameon=True, fontsize=8,
                  bbox_to_anchor=(0.5, -0.02))


class WebMapGenerator:
    """
    Generates an interactive HTML map using Leaflet.js (via simple string templating).
    Supports custom coordinate systems (non-geographic) using L.CRS.Simple.
    """
    
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Map Decluttering Results</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
            #map { width: 100%; height: 100vh; background: #f8f9fa; }
            
            /* Labels */
            .map-label { 
                background: rgba(255, 255, 255, 0.85); 
                border: 1px solid #999; 
                border-radius: 3px; 
                padding: 1px 4px;
                font-size: 10px;
                font-weight: bold;
                white-space: nowrap;
                color: #333;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            }

            /* Info Panel */
            .info-panel {
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                max-width: 300px;
            }

            /* Upload Control */
            .upload-control {
                position: absolute;
                top: 10px;
                left: 50px;
                z-index: 1000;
                background: white;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .upload-btn {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 14px;
                cursor: pointer;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 5px;
                transition: transform 0.2s;
            }
            .upload-btn:hover { transform: scale(1.05); }

            /* Loading Overlay */
            #loading {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.7);
                z-index: 2000;
                display: none;
                justify-content: center; align-items: center; flex-direction: column;
                color: white;
            }
            .spinner {
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-top: 4px solid #fff;
                border-radius: 50%;
                width: 40px; height: 40px;
                animation: spin 1s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div id="map"></div>

        <!-- Upload/Predict Control -->
        <div class="upload-control">
            <input type="file" id="file-input" accept=".png,.jpg,.jpeg" style="display:none;">
            <button type="button" class="upload-btn" onclick="document.getElementById('file-input').click()">
                <span>📂</span> Upload Map Photo
            </button>
            <button type="button" id="predict-btn" class="upload-btn" onclick="predictMap()" style="background: linear-gradient(45deg, #2196F3, #21CBF3); display:none; margin-left:10px;">
                <span>🔮</span> Predict Your Photo
            </button>
            <span id="filename" style="font-size:12px; color:#666; margin-left:10px;"></span>
        </div>

        <div class="info-panel">
            <h3>Decluttering Results</h3>
            <div id="stats">Loading stats...</div>
            <hr>
            <h4>Legend</h4>
            <div style="font-size: 12px; line-height: 1.5;">
                <div><span style="display:inline-block;width:12px;height:12px;background:#E74C3C;border-radius:2px;margin-right:5px;"></span> Highway</div>
                <div><span style="display:inline-block;width:12px;height:12px;background:#F39C12;border-radius:2px;margin-right:5px;"></span> Main Road</div>
                <div><span style="display:inline-block;width:12px;height:12px;background:#95A5A6;border-radius:2px;margin-right:5px;"></span> Local Road</div>
                <div><span style="display:inline-block;width:12px;height:12px;background:#3498DB;border-radius:2px;margin-right:5px;"></span> River</div>
                <div><span style="display:inline-block;width:12px;height:12px;background:#27AE60;border-radius:50%;margin-right:5px;"></span> Building</div>
                <div><span style="display:inline-block;width:12px;height:12px;background:#9B59B6;border-radius:50%;margin-right:5px;"></span> Icon (POI)</div>
            </div>
            <hr>
            <button onclick="toggleLabels()">Toggle Labels</button>
        </div>

        <div id="loading">
            <div class="spinner"></div>
            <h2>Analyzing Map...</h2>
            <p>Detecting roads, buildings, and optimizing placement.</p>
        </div>

        <script>
            // Initialize map
            var map = L.map('map', { crs: L.CRS.Simple, minZoom: -3, maxZoom: 2 });

            var layers = {
                highways: L.layerGroup().addTo(map),
                main_roads: L.layerGroup().addTo(map),
                local_roads: L.layerGroup().addTo(map),
                features: L.layerGroup().addTo(map),
                labels: L.layerGroup().addTo(map)
            };
            L.control.layers(null, layers).addTo(map);

            // Initial Data
            var featureData = FEATURE_DATA_PLACEHOLDER;
            var stats = STATS_PLACEHOLDER;
            
            renderMap(featureData, stats);

            // --- Logic ---
            const fileInput = document.getElementById('file-input');
            const predictBtn = document.getElementById('predict-btn');
            const filenameSpan = document.getElementById('filename');
            const loading = document.getElementById('loading');
            
            fileInput.addEventListener('change', function() {
                if (this.files.length > 0) {
                    filenameSpan.textContent = this.files[0].name;
                    predictBtn.style.display = 'flex';
                }
            });

            function predictMap() {
                var file = fileInput.files[0];
                if (!file) return;

                var formData = new FormData();
                formData.append('file', file);

                loading.style.display = 'flex';

                fetch('http://localhost:5000/api/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.error) {
                        alert("Error: " + data.error);
                    } else {
                        renderMap(data.features, data.stats);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    loading.style.display = 'none';
                    alert("Analysis Failed. Make sure the backend server is running on port 5000.");
                });
            }

            function renderMap(features, mapStats) {
                // Clear existing
                Object.values(layers).forEach(l => l.clearLayers());
                var bounds = [];

                // Update Stats
                if (mapStats) {
                    document.getElementById('stats').innerHTML = `
                        <b>Success Rate:</b> ${mapStats.success_rate ? mapStats.success_rate.toFixed(1) : 0}%<br>
                        <b>Total Features:</b> ${mapStats.total_features || 0}<br>
                        <b>Moves:</b> ${mapStats.features_moved || 0}
                    `;
                }

                features.forEach(function(f) {
                    var color = '#333';
                    var weight = 1;
                    var layer = layers.features;

                    switch(f.type) {
                        case 'highway': color = '#E74C3C'; weight = 6; layer = layers.highways; break;
                        case 'main_road': color = '#F39C12'; weight = 4; layer = layers.main_roads; break;
                        case 'local_road': color = '#95A5A6'; weight = 2; layer = layers.local_roads; break;
                        case 'river': color = '#3498DB'; weight = 5; layer = layers.features; break;
                        case 'building': color = '#27AE60'; break;
                        case 'icon': color = '#9B59B6'; break;
                    }

                    var popupContent = `<b>Type:</b> ${f.type}<br><b>ID:</b> ${f.id}<br><b>Priority:</b> ${f.priority}`;

                    if (f.geometry_type === 'LineString') {
                        var latlngs = f.coords.map(c => [c[1], c[0]]);
                        L.polyline(latlngs, {color: color, weight: weight, opacity: 0.8})
                         .bindTooltip(popupContent, {sticky: true}).addTo(layer);
                        latlngs.forEach(p => bounds.push(p));
                        
                        if (f.id && layer !== layers.local_roads) {
                            var mid = latlngs[Math.floor(latlngs.length / 2)];
                            var labelIcon = L.divIcon({className: 'map-label', html: f.id, iconSize: null});
                            L.marker(mid, {icon: labelIcon}).addTo(layers.labels);
                        }
                    } else if (f.geometry_type === 'Point') {
                        var latlng = [f.coords[0][1], f.coords[0][0]];
                        bounds.push(latlng);
                        L.circleMarker(latlng, {
                            radius: f.type === 'icon' ? 8 : 4,
                            fillColor: color, color: '#fff', weight: 1, opacity: 1, fillOpacity: 0.8
                        }).bindTooltip(popupContent).addTo(layer);
                        
                        var labelIcon = L.divIcon({className: 'map-label', html: f.id, iconSize: null});
                        L.marker([latlng[0] + 20, latlng[1]], {icon: labelIcon}).addTo(layers.labels);
                    }
                });

                if (bounds.length > 0) {
                    map.fitBounds(bounds, {padding: [50, 50]});
                } else {
                    map.setView([0, 0], 0);
                }
            }
            
            function toggleLabels() {
                if (map.hasLayer(layers.labels)) map.removeLayer(layers.labels);
                else map.addLayer(layers.labels);
            }
        </script>
    </body>
    </html>
    """

    @staticmethod
    def generate(features: List[MapFeature], stats: Dict, output_path: str = "map_interactive.html"):
        """Generate HTML file"""
        import json
        
        # Prepare feature data for JS
        js_features = []
        for f in features:
            coords = []
            geom_type = ""
            
            try:
                if hasattr(f.geometry, 'coords'): # LineString or Point
                    coords = list(f.geometry.coords)
                    geom_type = f.geometry.geom_type
                elif hasattr(f.geometry, 'x'): # Point
                    coords = [(f.geometry.x, f.geometry.y)]
                    geom_type = "Point"
            except Exception:
                continue
            
            js_features.append({
                'id': f.id,
                'type': f.type.value,
                'priority': f.style.priority,
                'geometry_type': geom_type,
                'coords': coords
            })
            
        # Serialize
        feature_json = json.dumps(js_features)
        stats_json = json.dumps(stats)
        
        # Inject into template
        html = WebMapGenerator.HTML_TEMPLATE.replace('FEATURE_DATA_PLACEHOLDER', feature_json)
        html = WebMapGenerator.HTML_TEMPLATE.replace('FEATURE_DATA_PLACEHOLDER', feature_json).replace('STATS_PLACEHOLDER', stats_json)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Interactive map saved to {output_path}")
        return output_path


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class MapDeclutterSystem:
    """
    Main system that orchestrates the entire decluttering process.
    """
    
    def __init__(self):
        self.parser = WKTParser()
        self.generator = MapGenerator()
        self.detector = OverlapDetector(grid_size=50.0)
        self.displacer = DisplacementEngine(max_iterations=500, step_size=2.0)
        self.visualizer = MapVisualizer(figsize=(18, 10))
        self.web_generator = WebMapGenerator()
        
        self.features: List[MapFeature] = []
        self.original_features: List[MapFeature] = []
        self.overlaps: List[Dict] = []
        self.stats: Dict = {}

        self.stats: Dict = {}
    
    def load_from_wkt_file(self, file_path: str) -> bool:
        """Load and process map data from a WKT file"""
        print(f"📂 Loading WKT data from {file_path}...")
        
        try:
            # Parse WKT file
            wkt_data = self.parser.parse_file(file_path)
            print(f"   ✓ Parsed {len(wkt_data)} WKT geometries")
            
            # Create map features
            self.features = self.generator.create_from_wkt(wkt_data)
            print(f"   ✓ Created {len(self.features)} map features")
            
            # Store original positions for comparison
            self._store_original_features()
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error loading file: {e}")
            return False
    
    def _store_original_features(self):
        """Store copies of features at their original positions"""
        self.original_features = []
        for feat in self.features:
            original_feat = MapFeature(
                id=feat.id,
                type=feat.type,
                geometry=feat.original_geometry,
                style=feat.style
            )
            self.original_features.append(original_feat)
    
    def detect_overlaps(self):
        """Detect all overlaps in the current map"""
        print("🔍 Detecting overlaps...")
        
        self.overlaps = self.detector.detect_all_overlaps(self.features)
        
        # Calculate statistics
        fixed_count = len([f for f in self.features if not f.style.is_movable])
        movable_count = len([f for f in self.features if f.style.is_movable])
        
        print(f"   ✓ Found {len(self.overlaps)} overlaps")
        print(f"   ✓ Fixed features: {fixed_count}")
        print(f"   ✓ Movable features: {movable_count}")
        
        # Display top overlaps
        if self.overlaps:
            print("\n   Top 5 overlaps:")
            for i, overlap in enumerate(self.overlaps[:5]):
                print(f"     {i+1}. {overlap['feature1']} ({overlap['type1']}) "
                      f"× {overlap['feature2']} ({overlap['type2']}) "
                      f"- Clearance violation: {overlap['clearance_violation']:.2f} units")
    
    def resolve_overlaps(self):
        """Resolve all detected overlaps using priority-based displacement"""
        print("🔄 Resolving overlaps...")
        
        initial_conflict_count = len(self.overlaps)
        
        if not self.overlaps:
            print("   ✓ No overlaps to resolve")
            # Ensure stats are initialized even if no overlaps
            self.stats['total_conflicts'] = 0
            self.stats['remaining_conflicts'] = 0
            self.stats['success_rate'] = 100.0
            self.stats['features_moved'] = 0
            self.stats['resolved_conflicts'] = 0
            return
        
        # Create copy for processing
        processed_features = self._create_feature_copies()
        
        # Apply displacement
        self.features, self.stats = self.displacer.resolve_overlaps(
            processed_features, self.overlaps
        )
        
        # Update statistics
        self._update_statistics()
        
        # Check for remaining overlaps - The Review Step
        remaining_overlaps = self.detector.detect_all_overlaps(self.features)
        remaining_count = len(remaining_overlaps)
        self.stats['remaining_conflicts'] = remaining_count
        
        # Recalculate robust success rate
        if initial_conflict_count > 0:
            resolved_count = max(0, initial_conflict_count - remaining_count)
            self.stats['resolved_conflicts'] = resolved_count # Correct the engine's estimate
            self.stats['success_rate'] = (resolved_count / initial_conflict_count) * 100.0
            self.stats['total_conflicts'] = initial_conflict_count
        else:
            self.stats['success_rate'] = 100.0
        
        # Print results
        self._print_resolution_results(remaining_overlaps)
    
    def _create_feature_copies(self) -> List[MapFeature]:
        """Create copies of features for processing"""
        processed_features = []
        for feat in self.features:
            processed_feat = MapFeature(
                id=feat.id,
                type=feat.type,
                geometry=feat.geometry,
                style=feat.style,
                original_geometry=feat.original_geometry
            )
            processed_features.append(processed_feat)
        return processed_features
    
    def _update_statistics(self):
        """Update processing statistics"""
        self.stats['total_features'] = len(self.features)
        self.stats['fixed_features'] = len([f for f in self.features if not f.style.is_movable])
        self.stats['movable_features'] = len([f for f in self.features if f.style.is_movable])
    
    def _print_resolution_results(self, remaining_overlaps: List[Dict]):
        """Print results of overlap resolution"""
        print(f"   ✓ Moved {self.stats['features_moved']} features")
        print(f"   ✓ Resolved {self.stats['resolved_conflicts']} conflicts")
        print(f"   ✓ Success rate: {self.stats.get('success_rate', 0):.1f}%")
        
        if remaining_overlaps:
            print(f"   ⚠️  {len(remaining_overlaps)} overlaps remain unresolved")
        else:
            print("   ✓ All overlaps resolved!")

    def visualize(self, output_path: str = "map_declutter_result.png"):
        """Create visualization of before/after comparison"""
        print("🎨 Generating visualization...")
        
        self.visualizer.create_comparison_view(
            self.original_features,
            self.features,
            self.overlaps,
            self.stats,
            output_path
        )
        
        # Also generate web map
        web_path = self.web_generator.generate(self.features, self.stats)
        
        # Open in browser
        import webbrowser
        import os
        webbrowser.open('file://' + os.path.realpath(web_path))
        
        print(f"   ✓ Visualization complete")
    
    def export_results(self, output_dir: str = "results"):
        """Export results to JSON files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export statistics
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Export overlap report
        overlaps_simple = []
        for overlap in self.overlaps:
            simple = {
                "feature1": overlap["feature1"],
                "feature2": overlap["feature2"],
                "overlap_area": overlap.get("overlap_area", 0),
                "clearance_violation": overlap.get("clearance_violation", 0)
            }
            overlaps_simple.append(simple)
        
        overlaps_path = os.path.join(output_dir, "overlaps_report.json")
        with open(overlaps_path, 'w') as f:
            json.dump(overlaps_simple, f, indent=2)
        
        print(f"📁 Results exported to '{output_dir}' directory")
        print(f"   ✓ statistics.json - Processing statistics")
        print(f"   ✓ overlaps_report.json - Overlap details")


# ============================================================================
# DEMO AND UTILITIES
# ============================================================================

def create_sample_wkt_file() -> str:
    """Create a sample WKT file with road network data"""
    wkt_content = """LINESTRING(7071.421606 8585.627528, 7074.672945 8588.813669, 7074.902551 8589.026835, 7075.084535 8589.196346, 7079.265638 8592.884220, 7081.745726 8594.906230, 7083.857197 8596.531276, 7087.160750 8598.968277, 7091.114457 8601.621165, 7093.513644 8603.078683, 7095.106091 8604.038154, 7097.122261 8605.189984, 7104.911244 8609.486740, 7105.304126 8609.703307, 7105.767874 8609.959559, 7144.536643 8631.196101, 7154.035087 8636.400567, 7154.181354 8636.479937, 7154.367307 8636.580850, 7163.899654 8641.749543, 7194.100989 8658.211975, 7204.788850 8664.037795, 7217.453480 8671.075087, 7225.937575 8676.277228, 7229.984882 8679.124346, 7234.142740 8682.304819, 7234.328126 8682.456756, 7234.580976 8682.656882, 7238.456504 8685.821480, 7241.288485 8688.334904, 7243.451150 8690.254299, 7247.565354 8694.303307, 7251.962457 8699.059843, 7255.123087 8702.735811, 7259.170961 8707.866520, 7263.048189 8713.311874, 7263.229663 8713.578104, 7263.406488 8713.854425, 7268.192504 8721.195591, 7275.567515 8732.944687, 7282.277291 8743.634079, 7286.474268 8750.172472, 7288.218142 8752.568882, 7288.396611 8752.792535, 7288.576271 8753.017663, 7288.766362 8753.256000, 7288.953449 8753.490142, 7297.531654 8764.344000, 7297.778268 8764.628598, 7297.994835 8764.879181, 7298.070236 8764.969323, 7299.845291 8766.983055, 7303.696724 8770.797865)
LINESTRING(7230.130016 8690.909669, 7236.635528 8693.947843, 7238.203087 8694.558992, 7247.547213 8697.223559)
LINESTRING(7230.130016 8690.909669, 7233.188769 8684.714098)
LINESTRING(7233.188769 8684.714098, 7233.714028 8683.650142)
LINESTRING(7233.714028 8683.650142, 7234.189795 8682.685228, 7234.328126 8682.456756, 7234.481764 8682.201071, 7235.065134 8681.329304)
LINESTRING(7235.065134 8681.329304, 7235.731843 8680.333039)
LINESTRING(7255.097575 8699.453858, 7251.598772 8695.211301, 7250.671729 8694.158287, 7249.321757 8692.736315, 7247.598803 8690.985071, 7245.361757 8688.794287, 7239.550677 8683.562835, 7235.731843 8680.333039)
LINESTRING(7171.129701 8650.784296, 7172.734677 8650.679301, 7173.134929 8650.696252, 7173.582180 8650.716775, 7173.964687 8650.799320, 7174.330016 8650.952504, 7192.405417 8660.418520, 7198.342866 8663.259969)
LINESTRING(7151.646784 8646.301984, 7151.809323 8646.372283, 7151.995106 8646.453184)
LINESTRING(7178.626091 8635.919301, 7179.106110 8636.046803)
LINESTRING(7235.731843 8680.333039, 7235.002715 8679.740258, 7233.283729 8678.570287, 7231.696384 8677.486488, 7229.566488 8676.136800, 7227.667729 8674.979244, 7226.335729 8674.169216, 7220.525669 8670.832441, 7216.929071 8668.920189, 7214.536687 8667.554230, 7210.410066 8665.005883, 7203.277644 8660.723244, 7191.057770 8653.562079, 7187.617644 8651.606230, 7183.576630 8649.338230, 7179.121587 8646.908202, 7173.685587 8644.037216)
LINESTRING(7173.685587 8644.037216, 7174.069228 8643.448063, 7178.431691 8637.105203, 7179.218646 8635.065449)
LINESTRING(7154.987528 8635.107402, 7154.314016 8636.228220, 7154.181354 8636.479937, 7154.062299 8636.736756, 7153.217575 8638.218709)
LINESTRING(7152.793512 8639.111282, 7153.217575 8638.218709)
LINESTRING(7152.793512 8639.111282, 7153.257146 8639.406765, 7153.557165 8639.556775, 7153.850835 8639.674583, 7154.194677 8639.819263, 7154.884687 8640.111798, 7155.724649 8640.524296, 7156.992189 8641.214249, 7159.482198 8642.519263, 7161.747194 8643.824277, 7164.775956 8645.618891, 7166.067364 8646.359017, 7167.214715 8646.966765, 7168.564687 8647.641808, 7169.262180 8647.956794, 7169.727005 8648.137304, 7170.199710 8648.339301, 7170.452787 8648.496000, 7170.612208 8648.631780, 7170.707906 8648.787402, 7170.867213 8649.104315, 7170.957184 8649.344296, 7171.129701 8650.784296)
LINESTRING(7149.314835 8641.775055, 7152.249260 8643.462803, 7152.196762 8643.665480, 7151.527559 8646.249827, 7151.646784 8646.301984)
LINESTRING(7160.216769 8627.509134, 7161.288321 8627.945046, 7161.363326 8628.357543, 7161.850828 8628.826280, 7162.638350 8629.126299, 7163.425814 8629.276309, 7164.344580 8629.632567, 7165.750847 8629.951294, 7169.139666 8630.992460, 7171.525871 8631.995074, 7173.569594 8633.045083, 7173.963326 8633.288806, 7174.900857 8633.607534, 7175.182110 8633.813783, 7175.932101 8634.132567, 7176.344598 8634.488825, 7176.607087 8634.901323, 7177.432139 8635.088806, 7177.844636 8635.351294, 7178.163364 8635.688787, 7178.626091 8635.919301)
LINESTRING(7159.517858 8627.265638, 7160.009443 8627.429027, 7160.216769 8627.509134)
LINESTRING(7160.535723 8626.610494, 7161.190186 8626.867994, 7161.840737 8627.165518, 7162.562438 8627.506469, 7163.093197 8627.735509, 7164.375704 8628.230494, 7167.825751 8629.505518, 7169.459698 8630.137304, 7170.764315 8630.641757, 7171.748220 8631.080504, 7172.826746 8631.579005, 7177.448636 8634.269310, 7178.738627 8634.989310, 7179.218646 8635.065449)
LINESTRING(7155.724592 8633.846154, 7156.288573 8634.128202, 7156.576573 8634.326173, 7156.789682 8634.510085, 7157.044573 8634.758173, 7157.251559 8634.947187, 7157.458545 8635.100202, 7157.654419 8635.247036, 7157.998545 8635.469159, 7158.331559 8635.676202, 7159.084554 8636.078211, 7159.924573 8636.546211, 7164.460573 8639.054192, 7171.300573 8642.786173, 7173.685587 8644.037216)
LINESTRING(7160.535723 8626.610494, 7160.009443 8627.429027, 7159.280315 8628.578646, 7157.053814 8631.868535, 7155.724592 8633.846154)
LINESTRING(7154.987528 8635.107402, 7155.724592 8633.846154)
LINESTRING(7152.793512 8639.111282, 7147.071043 8635.953770, 7136.696693 8630.205165, 7135.561928 8629.667093, 7134.453468 8629.151754, 7124.383899 8624.969065, 7123.879899 8624.834079, 7123.323969 8624.777953, 7122.655899 8624.780050, 7121.539899 8624.753065, 7120.874268 8624.652661, 7120.309039 8624.535874, 7119.541928 8624.303093, 7118.794885 8624.141065, 7118.036787 8624.066457, 7117.705984 8624.072693, 7117.417928 8624.078079, 7117.057928 8624.132050, 7116.697928 8624.222079, 7116.004913 8624.447093)
LINESTRING(7116.004913 8624.447093, 7113.358885 8624.348050, 7112.647899 8624.312050, 7111.216913 8624.249065, 7107.446608 8624.105802, 7101.497594 8623.934816, 7098.797594 8623.898816)
LINESTRING(7097.387528 8625.209953, 7097.195565 8625.977802, 7097.012901 8626.467175, 7096.772976 8626.965732, 7094.127118 8630.724472, 7094.908403 8631.338117, 7094.921896 8631.419074, 7094.895534 8631.491528, 7093.841272 8633.093953, 7093.819389 8633.187609, 7093.850910 8633.255074, 7093.922910 8633.322595, 7097.047370 8635.481575)
LINESTRING(7089.693165 8635.350614, 7093.515969 8630.246551, 7097.387528 8625.209953)
LINESTRING(7098.797594 8623.898816, 7098.995565 8623.691830, 7099.184580 8623.448787, 7099.318488 8623.211528)
LINESTRING(7097.387528 8625.209953, 7097.798608 8624.780787, 7098.419565 8624.213802, 7098.797594 8623.898816)
LINESTRING(7099.318488 8623.211528, 7104.026608 8612.540787)
LINESTRING(7104.026608 8612.540787, 7104.345449 8611.891087)
LINESTRING(7098.845669 8623.387276, 7083.967181 8619.079748, 7065.907654 8610.328630)
LINESTRING(7065.907654 8610.328630, 7065.670110 8610.071244, 7065.418961 8609.813291)
LINESTRING(7065.824882 8609.759433, 7065.670110 8610.071244, 7065.592441 8610.237921)
LINESTRING(7065.824882 8609.759433, 7066.134992 8610.118299, 7083.707584 8618.672806, 7084.337556 8618.912787, 7085.072580 8619.137802, 7099.318488 8623.211528)
LINESTRING(7115.152479 8607.998154, 7118.572479 8608.358154, 7118.980498 8608.406117, 7119.388517 8608.574154, 7124.440479 8611.886154, 7128.544535 8614.742117, 7129.348498 8615.426117, 7136.505071 8622.120756, 7136.992630 8622.429732, 7137.520554 8622.686154, 7138.084535 8622.842173, 7155.354331 8626.387465, 7159.517858 8627.265638)
LINESTRING(7155.724592 8633.846154, 7154.692554 8633.702154, 7154.272573 8633.594154, 7153.972554 8633.486154, 7153.408573 8633.222192, 7151.620535 8632.274173, 7144.495937 8628.389291, 7134.532498 8622.974154, 7128.160498 8619.374154, 7125.316498 8617.778135, 7122.148498 8615.966173, 7116.460498 8612.642154, 7112.140498 8610.206117, 7108.804460 8608.202135, 7108.312479 8607.866117, 7107.844479 8607.566154, 7106.633008 8606.704819)
LINESTRING(7106.219150 8607.637984, 7106.633008 8606.704819)
LINESTRING(7106.219150 8607.637984, 7105.419213 8609.443087, 7105.304126 8609.703307, 7105.189039 8609.963528, 7104.345449 8611.891087)
LINESTRING(7106.633008 8606.704819, 7109.809455 8599.548132, 7110.782362 8597.356157)
LINESTRING(7079.810570 8595.995754, 7079.428063 8595.905726, 7079.218072 8595.860769)
LINESTRING(7079.218072 8595.860769, 7073.398035 8593.385726)
LINESTRING(7073.398035 8593.385726, 7073.000561 8593.205726, 7072.700542 8593.078224)
LINESTRING(7103.963339 8612.145638, 7095.090898 8606.980913, 7089.563339 8603.501102, 7084.256315 8599.741228, 7079.810570 8595.995754)
LINESTRING(7104.026608 8612.540787, 7097.464233 8608.697461, 7096.025594 8607.869802, 7094.361090 8606.887994, 7091.939565 8605.367773, 7090.350917 8604.335112, 7089.122551 8603.504787, 7087.565594 8602.415773, 7086.089594 8601.362759, 7084.766551 8600.381745, 7083.353537 8599.319773, 7082.336580 8598.509745, 7081.346551 8597.699773, 7079.428063 8595.905726, 7077.905575 8594.578261, 7076.533039 8593.408233, 7074.405581 8591.589128, 7074.031294 8591.330154)
LINESTRING(7074.031294 8591.330154, 7073.946709 8590.907169)
LINESTRING(7073.946709 8590.907169, 7074.141449 8590.748598, 7074.271559 8590.451528)
LINESTRING(7073.098016 8592.920731, 7073.702589 8591.595987, 7074.031294 8591.330154)
LINESTRING(7073.098016 8592.920731, 7073.000561 8593.205726, 7065.824882 8609.759433)
LINESTRING(7075.565858 8587.529008, 7075.987087 8586.579402)
LINESTRING(7074.271559 8590.451528, 7074.781228 8589.301228, 7074.902551 8589.026835, 7075.023874 8588.753008, 7075.565858 8587.529008)
LINESTRING(7078.460542 8580.883238, 7088.449323 8586.079937, 7107.773102 8595.942917, 7107.972548 8596.047402, 7109.423093 8596.752945, 7110.782362 8597.356157)
LINESTRING(7075.987087 8586.579402, 7078.460542 8580.883238)"""
    
    file_path = 'streets_ugen.wkt'
    with open(file_path, 'w') as f:
        f.write(wkt_content)
    
    return file_path


def create_homestead_demo_data() -> str:
    """
    Create a synthetic WKT file mimicking the 'Homestead/Leisure City' map screenshot.
    includes diagonal highways, grid networks, and specific POIs.
    """
    wkt_lines = []
    
    # 1. Diagonal Highway (US 1 style) - High Priority
    # Running SW to NE
    wkt_lines.append("LINESTRING(3000 3000, 8000 8000)") 
    # Parallel highway lane
    wkt_lines.append("LINESTRING(3050 2950, 8050 7950)")
    
    # 2. Main horizontal/vertical grid (Major Roads)
    # Horizontal roads
    for y in range(3000, 8001, 1000):
        wkt_lines.append(f"LINESTRING(2000 {y}, 9000 {y})")
    
    # Vertical roads bounding the area
    wkt_lines.append("LINESTRING(4000 2000, 4000 9000)") # Krome Ave style
    wkt_lines.append("LINESTRING(7000 2000, 7000 9000)") 
    
    # 3. Dense local grid (Local Roads)
    # Create a grid in the "Homestead" area (left of highway)
    for x in range(3200, 4800, 200):
        wkt_lines.append(f"LINESTRING({x} 4000, {x} 6000)")
    for y in range(4200, 5800, 200):
        wkt_lines.append(f"LINESTRING(3000 {y}, 5000 {y})")

    file_path = 'homestead_simulation.wkt'
    with open(file_path, 'w') as f:
        f.write("\n".join(wkt_lines))
    
    return file_path



def run_demo():
    """Run the complete map decluttering demo"""
    print("=" * 80)
    print("🗺️  MAP DECLUTTERING SYSTEM - HACKATHON SOLUTION")
    print("=" * 80)
    
    # Step 1: Create sample data
    print("\n[1/6] Creating sample data...")
    wkt_file = create_sample_wkt_file()
    print(f"   ✓ Created '{wkt_file}' with road network data")
    
    # Step 2: Initialize system
    print("\n[2/6] Initializing decluttering system...")
    system = MapDeclutterSystem()
    
    # Step 3: Load data
    print("\n[3/6] Loading and processing map data...")
    if not system.load_from_wkt_file(wkt_file):
        print("   ✗ Failed to load data")
        return
    
    # Step 4: Detect overlaps
    print("\n[4/6] Detecting feature overlaps...")
    system.detect_overlaps()
    
    # Step 5: Resolve overlaps
    print("\n[5/6] Applying priority-based displacement...")
    system.resolve_overlaps()
    
    # Step 6: Visualize and export results
    print("\n[6/6] Generating comprehensive report...")
    system.visualize("hackathon_solution_result.png")
    system.export_results("hackathon_results")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\n📊 RESULTS SUMMARY:")
    print(f"   • Total features processed: {system.stats.get('total_features', 0)}")
    print(f"   • Overlaps detected: {system.stats.get('total_conflicts', 0)}")
    print(f"   • Overlaps resolved: {system.stats.get('resolved_conflicts', 0)}")
    print(f"   • Features moved: {system.stats.get('features_moved', 0)}")
    print(f"   • Success rate: {system.stats.get('success_rate', 0):.1f}%")
    print(f"   • Total displacement: {system.stats.get('total_displacement', 0):.1f} units")
    print("\n📁 Output files:")
    print("   • hackathon_solution_result.png - Visualization")
    print("   • hackathon_results/statistics.json - Statistics")
    print("   • hackathon_results/overlaps_report.json - Overlap details")
    print("\n🎯 Hackathon Requirements Satisfied:")
    print("   ✓ Overlap Detection with width consideration")
    print("   ✓ Priority-Based Displacement (Highway > Road > Label/Icon)")
    print("   ✓ Before/After Visualization")
    print("   ✓ Metrics Reporting")
    print("   ✓ Network Topology Preservation")
    print("=" * 80)


if __name__ == "__main__":
    # Check and install dependencies
    try:
        import shapely
        import matplotlib
        import numpy as np
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "shapely", "matplotlib", "numpy"])
        print("Dependencies installed. Please run the script again.")
        sys.exit(0)
    
    run_demo()