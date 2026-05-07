"""
Evaluation utilities for coordinate validation against polygons and masks.
"""

from typing import List, Tuple

# Type aliases for clarity
Point = Tuple[float, float]
Polygon = List[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Check if a point is inside or on a polygon using the ray casting algorithm.
    
    Args:
        point: A tuple (x, y) representing the point to test
        polygon: A list of (x, y) tuples representing the polygon vertices
        
    Returns:
        True if the point is inside or on the boundary of the polygon, False otherwise
    """
    if len(polygon) < 3:
        return False
    
    x, y = point
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, len(polygon) + 1):
        p2x, p2y = polygon[i % len(polygon)]
        
        # Check if point is on the edge
        if _point_on_segment(point, (p1x, p1y), (p2x, p2y)):
            return True
        
        # Ray casting algorithm
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def _point_on_segment(point: Point, seg_start: Point, seg_end: Point) -> bool:
    """
    Check if a point lies on a line segment.
    """
    x, y = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Check if point is collinear with segment
    cross_product = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if abs(cross_product) > 1e-9:
        return False
    
    # Check if point is within segment bounds
    if x < min(x1, x2) - 1e-9 or x > max(x1, x2) + 1e-9:
        return False
    if y < min(y1, y2) - 1e-9 or y > max(y1, y2) + 1e-9:
        return False
    
    return True


def evaluate_coordinates(
    coordinates: List[Point],
    json_data: dict
) -> Tuple[bool, str]:
    """
    Evaluate if a list of coordinates is correct based on the JSON data rules.
    
    Rules:
    1. If there are "Banned" type polygons, no coordinate should be inside any of them
    2. If all Target polygons have rank == "":
       - Every Target polygon must contain at least one coordinate
    3. If Target polygons have numeric ranks:
       - Must have at least K coordinates (where K = max rank)
       - The i-th coordinate must be inside at least one polygon with rank == str(i)
    
    Args:
        coordinates: List of (x, y) coordinate tuples from user clicks
        json_data: The JSON data dict containing polygons and other info
        
    Returns:
        (is_correct: bool, error_message: str)
        Returns (True, "") if coordinates are correct
        Returns (False, error_message) if coordinates are incorrect
    """
    polygons = json_data.get("polygons", [])
    image_id = json_data.get("image_id", "unknown")
    
    if not polygons:
        return False, f"No polygons found in data for {image_id}"
    
    # Convert polygon dicts to list of Point tuples
    polygon_objects = []
    for poly_dict in polygons:
        points = [tuple(p) for p in poly_dict.get("points", [])]
        if points:
            polygon_objects.append({
                "id": poly_dict.get("id", ""),
                "type": poly_dict.get("type", ""),
                "rank": poly_dict.get("rank", ""),
                "points": points
            })
    
    # Step 1: Check Banned polygons
    banned_polygons = [p for p in polygon_objects if p["type"] == "Banned"]
    for coord in coordinates:
        for banned_poly in banned_polygons:
            if point_in_polygon(coord, banned_poly["points"]):
                return False, f"Coordinate {coord} is inside a Banned polygon for {image_id}"
    
    # Get Target polygons
    target_polygons = [p for p in polygon_objects if p["type"] == "Target"]
    
    if not target_polygons:
        return True, ""  # No targets to validate
    
    # Determine rank type (all empty strings or all numeric)
    ranks = [p["rank"] for p in target_polygons]
    all_empty_rank = all(r == "" for r in ranks)
    all_numeric_rank = all(r != "" for r in ranks)
    
    if not all_empty_rank and not all_numeric_rank:
        return False, f"Inconsistent rank types in {image_id}"
    
    if all_empty_rank:
        # Rule: Every Target polygon must contain at least one coordinate
        for target_poly in target_polygons:
            found = False
            for coord in coordinates:
                if point_in_polygon(coord, target_poly["points"]):
                    found = True
                    break
            if not found:
                return False, f"Target polygon {target_poly['id']} contains no coordinates for {image_id}"
        return True, ""
    
    else:  # all_numeric_rank
        # Rule: Each coordinate i should be in a polygon with rank == str(i)
        # First, find the max rank to determine required coordinate count
        max_rank = 0
        for r in ranks:
            if r and r.isdigit():
                max_rank = max(max_rank, int(r))
        
        if len(coordinates) < max_rank:
            return False, f"Need at least {max_rank} coordinates but got {len(coordinates)} for {image_id}"
        
        # Check each required coordinate
        for rank_num in range(1, max_rank + 1):
            rank_str = str(rank_num)
            rank_polygons = [p for p in target_polygons if p["rank"] == rank_str]
            
            if not rank_polygons:
                return False, f"No polygon with rank {rank_str} found for {image_id}"
            
            coord = coordinates[rank_num - 1]
            found = False
            for rank_poly in rank_polygons:
                if point_in_polygon(coord, rank_poly["points"]):
                    found = True
                    break
            
            if not found:
                return False, f"Coordinate {rank_num} at {coord} is not in any polygon with rank {rank_str} for {image_id}"
        
        return True, ""


def validate_against_json_file(
    coordinates: List[Point],
    json_data_dict: dict
) -> Tuple[bool, str]:
    """
    Public API for validation. Wrapper around evaluate_coordinates.
    
    Args:
        coordinates: List of (x, y) coordinate tuples
        json_data_dict: Loaded JSON data
        
    Returns:
        (is_correct: bool, error_message: str)
    """
    return evaluate_coordinates(coordinates, json_data_dict)
