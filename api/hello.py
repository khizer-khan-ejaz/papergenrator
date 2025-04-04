from flask import Flask, request, jsonify
import folium
from geographiclib.geodesic import Geodesic
import scipy.optimize as optimize
import io
from folium.map import Figure
import base64
from flask_cors import CORS

import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Function to convert DMS to Decimal Degrees
def dms_to_dd(degrees, minutes, seconds, direction):
    try:
        dd = float(degrees) + float(minutes)/60 + float(seconds)/3600
        if direction in ['S', 'W']:
            dd *= -1
        return dd
    except:
        return 0.0
def calculate_geodesic(P1, P2, P3, P4, TAS, wind_speed, degree):
    # Initialize geodesic calculator
    geod = Geodesic.WGS84

    def geodesic_intersection(line1_start, line1_end, line2_start, line2_end):
        """
        Find the intersection of two geodesic lines using optimization with improved convergence
        """
        def distance_between_lines(params):
            s1, s2 = params
            # Constrain parameters to be within bounds
            s1 = max(0.0, min(1.0, s1))
            s2 = max(0.0, min(1.0, s2))
            
            line1 = geod.InverseLine(line1_start[0], line1_start[1], 
                                     line1_end[0], line1_end[1])
            line2 = geod.InverseLine(line2_start[0], line2_start[1], 
                                     line2_end[0], line2_end[1])
            point1 = line1.Position(s1 * line1.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            point2 = line2.Position(s2 * line2.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            g = geod.Inverse(point1['lat2'], point1['lon2'], 
                             point2['lat2'], point2['lon2'])
            return g['s12']
        
        # Try multiple initial guesses to avoid local minima
        best_result = None
        best_distance = float('inf')
        
        for s1_guess in [0.3, 0.5, 0.7]:
            for s2_guess in [0.3, 0.5, 0.7]:
                initial_guess = [s1_guess, s2_guess]
                try:
                    result = optimize.minimize(
                        distance_between_lines, 
                        initial_guess, 
                        method='L-BFGS-B',  # Changed to L-BFGS-B for better constrained optimization
                        bounds=[(0, 1), (0, 1)],
                        options={'gtol': 1e-10, 'ftol': 1e-10}  # Tighter tolerances
                    )
                    
                    if result.fun < best_distance:
                        best_distance = result.fun
                        best_result = result
                except:
                    continue
                    
        if best_result is None:
            # Fallback to Nelder-Mead if L-BFGS-B fails for all initial guesses
            initial_guess = [0.5, 0.5]
            best_result = optimize.minimize(
                distance_between_lines, 
                initial_guess, 
                method='Nelder-Mead',
                bounds=[(0, 1), (0, 1)]
            )
        
        line1 = geod.InverseLine(line1_start[0], line1_start[1], 
                                 line1_end[0], line1_end[1])
        line2 = geod.InverseLine(line2_start[0], line2_start[1], 
                                 line2_end[0], line2_end[1])
        
        # Constrain parameters to be within bounds
        s1 = max(0.0, min(1.0, best_result.x[0]))
        s2 = max(0.0, min(1.0, best_result.x[1]))
        
        point1 = line1.Position(s1 * line1.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        point2 = line2.Position(s2 * line2.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        
        # Check if the distance between points is very small
        g = geod.Inverse(point1['lat2'], point1['lon2'], 
                         point2['lat2'], point2['lon2'])
        
        # If points are very close (less than 1 meter), use exact point
        if g['s12'] < 1:
            intersection = (point1['lat2'], point1['lon2'])
        else:
            # Otherwise use weighted average based on confidence in each line
            # This is a more accurate way to handle near-intersections
            intersection = (
                (point1['lat2'] + point2['lat2']) / 2, 
                (point1['lon2'] + point2['lon2']) / 2
            )
        
        return intersection, g['s12']

    def generate_geodesic_points(start, end, num_points=200):  # Increased point count for smoothness
        """Generate points along a geodesic line with higher resolution"""
        line = geod.InverseLine(start[0], start[1], end[0], end[1])
        ds = line.s13 / (num_points - 1)
        points = []
        for i in range(num_points):
            s = ds * i
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            points.append((g['lat2'], g['lon2']))
        return points, line

    # Calculate midpoint of P3-P4 (previously labeled as C-D)
    # Using direct geodesic midpoint calculation rather than intersection
    line_P3_P4 = geod.InverseLine(P3[0], P3[1], P4[0], P4[1])
    mid_point = line_P3_P4.Position(line_P3_P4.s13/2, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
    mid_C_D = (mid_point['lat2'], mid_point['lon2'])

    # Calculate perpendicular line with more accurate bearing calculation
    g_CD = geod.Inverse(P3[0], P3[1], P4[0], P4[1])
    bearing_CD = g_CD['azi1']
    
    # Use the direct forward azimuth for the perpendicular calculation
    perp_bearing = (bearing_CD + 90) % 360
    
    # Generate geodesic lines with higher point density
    p1_p2_geodesic, p1_p2_line = generate_geodesic_points(P1, P2)
    p3_p4_geodesic, p3_p4_line = generate_geodesic_points(P3, P4)

    # Create longer perpendicular line for more reliable intersection
    perp_distance = 1000000  # 1000 km in meters - increased for better intersection
    perp_point1 = geod.Direct(mid_C_D[0], mid_C_D[1], perp_bearing, perp_distance)
    perp_point1 = (perp_point1['lat2'], perp_point1['lon2'])
    perp_point2 = geod.Direct(mid_C_D[0], mid_C_D[1], (perp_bearing + 180) % 360, perp_distance)
    perp_point2 = (perp_point2['lat2'], perp_point2['lon2'])
    perp_geodesic, perp_line = generate_geodesic_points(perp_point1, perp_point2)

    # Find precise intersections with improved algorithm
    p1p2_perp_intersection, p1p2_dist = geodesic_intersection(P1, P2, perp_point1, perp_point2)
    
    # Check if the intersection distance is reasonable
    if p1p2_dist > 1000:  # If greater than 1 km, the lines might not be truly intersecting
        # Try alternate intersection method
        # Find closest points on both lines
        def distance_to_point(line_start, line_end, point):
            line = geod.InverseLine(line_start[0], line_start[1], line_end[0], line_end[1])
            
            def calc_distance(s):
                pos = line.Position(s * line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                return geod.Inverse(pos['lat2'], pos['lon2'], point[0], point[1])['s12']
            
            result = optimize.minimize_scalar(calc_distance, bounds=(0, 1), method='bounded')
            pos = line.Position(result.x * line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            return (pos['lat2'], pos['lon2']), result.fun
        
        p1p2_closest, _ = distance_to_point(P1, P2, mid_C_D)
        p1p2_perp_intersection = p1p2_closest
    
    # Calculate distance to P3 more accurately
    distance_to_P3_m = geod.Inverse(p1p2_perp_intersection[0], p1p2_perp_intersection[1], 
                                   P3[0], P3[1])['s12']
    distance_to_P3_nm = distance_to_P3_m / 1852.0  # Convert to nautical miles directly

    # Apply the correct wind correction formula
    distance_to_degree = (distance_to_P3_nm / TAS) * wind_speed
    
    # Generate the degree line with corrected bearing
    line_distance = distance_to_degree * 1852  # Convert to meters
    
    # Make sure we use great circle navigation for the degree line
    nm_line_point = geod.Direct(p1p2_perp_intersection[0], p1p2_perp_intersection[1], degree, line_distance)
    nm_line_end_point = (nm_line_point['lat2'], nm_line_point['lon2'])
    nm_geodesic, nm_line = generate_geodesic_points(p1p2_perp_intersection, nm_line_end_point)

    # Calculate the correct perpendicular bearing to the P1-P2 line
    # First get the azimuth of P1-P2 at the point closest to nm_line_end_point
    def get_azimuth_at_nearest_point(line_start, line_end, point):
        line = geod.InverseLine(line_start[0], line_start[1], line_end[0], line_end[1])
        
        def calc_distance(s):
            pos = line.Position(s * line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            return geod.Inverse(pos['lat2'], pos['lon2'], point[0], point[1])['s12']
        
        result = optimize.minimize_scalar(calc_distance, bounds=(0, 1), method='bounded')
        s = result.x
        
        # Get azimuth at this point
        if s < 0.01:
            # Near the start, use the initial azimuth
            return geod.Inverse(line_start[0], line_start[1], line_end[0], line_end[1])['azi1']
        elif s > 0.99:
            # Near the end, use the final azimuth
            return geod.Inverse(line_start[0], line_start[1], line_end[0], line_end[1])['azi2']
        else:
            # In the middle, calculate the forward azimuth at this point
            pos1 = line.Position((s - 0.01) * line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            pos2 = line.Position((s + 0.01) * line.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            return geod.Inverse(pos1['lat2'], pos1['lon2'], pos2['lat2'], pos2['lon2'])['azi1']
    
    p1p2_azimuth_at_nearest = get_azimuth_at_nearest_point(P1, P2, nm_line_end_point)
    perp_to_p1p2_bearing = (p1p2_azimuth_at_nearest + 90) % 360

    # Create perpendicular line from degree line end point
    perp_nm_distance = 1000000  # 1000 km in meters
    perp_nm_point1 = geod.Direct(nm_line_end_point[0], nm_line_end_point[1], perp_to_p1p2_bearing, perp_nm_distance)
    perp_nm_point1 = (perp_nm_point1['lat2'], perp_nm_point1['lon2'])
    perp_nm_point2 = geod.Direct(nm_line_end_point[0], nm_line_end_point[1], (perp_to_p1p2_bearing + 180) % 360, perp_nm_distance)
    perp_nm_point2 = (perp_nm_point2['lat2'], perp_nm_point2['lon2'])
    perp_nm_geodesic, perp_nm_line = generate_geodesic_points(perp_nm_point1, perp_nm_point2)

    # Find intersection with improved algorithm
    perp_nm_p1p2_intersection, p1p2_nm_dist = geodesic_intersection(P1, P2, perp_nm_point1, perp_nm_point2)
    
    # If the intersection is not precise, use the nearest point method
    if p1p2_nm_dist > 100:  # If greater than 100m, try alternate method
        perp_nm_p1p2_closest, _ = distance_to_point(P1, P2, nm_line_end_point)
        perp_nm_p1p2_intersection = perp_nm_p1p2_closest

    # Calculate distance from perpendicular intersection to P1 with direct geodesic calculation
    distance_to_P1_m = geod.Inverse(perp_nm_p1p2_intersection[0], perp_nm_p1p2_intersection[1], 
                                   P1[0], P1[1])['s12']
    distance_to_P1_nm = distance_to_P1_m / 1852.0  # Direct conversion to nautical miles

    # Create map for optional HTML return
    my_map = folium.Map(location=p1p2_perp_intersection, zoom_start=6, tiles='OpenStreetMap')

    # Add lines with clearer styling
    folium.PolyLine(p1_p2_geodesic, color='purple', weight=4, tooltip='P1 to P2').add_to(my_map)
    folium.PolyLine(p3_p4_geodesic, color='orange', weight=4, tooltip='P3 to P4').add_to(my_map)
    folium.PolyLine(perp_geodesic, color='red', weight=3, tooltip='Perpendicular Line').add_to(my_map)
    folium.PolyLine(nm_geodesic, color='blue', weight=3, tooltip=f'{degree}-Degree Line').add_to(my_map)
    folium.PolyLine(perp_nm_geodesic, color='green', weight=3, tooltip='Perpendicular to P1-P2').add_to(my_map)

    # Add more informative markers
    folium.Marker(
        location=p1p2_perp_intersection,
        popup=f'Initial Intersection<br>Lat: {p1p2_perp_intersection[0]:.6f}<br>Lon: {p1p2_perp_intersection[1]:.6f}<br>Intersection accuracy: {p1p2_dist:.2f}m',
        icon=folium.Icon(color='black', icon='info-sign')
    ).add_to(my_map)

    folium.Marker(
        location=nm_line_end_point,
        popup=f'{degree}-Degree Line End<br>Lat: {nm_line_end_point[0]:.6f}<br>Lon: {nm_line_end_point[1]:.6f}<br>Distance: {distance_to_degree:.2f}nm',
        icon=folium.Icon(color='blue', icon='arrow-up')
    ).add_to(my_map)

    folium.Marker(
        location=perp_nm_p1p2_intersection,
        popup=f'Final Intersection<br>Lat: {perp_nm_p1p2_intersection[0]:.6f}<br>Lon: {perp_nm_p1p2_intersection[1]:.6f}<br>Intersection accuracy: {p1p2_nm_dist:.2f}m',
        icon=folium.Icon(color='green', icon='crosshair')
    ).add_to(my_map)

    # Add point markers
    points = {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4}
    colors = {'P1': 'blue', 'P2': 'blue', 'P3': 'green', 'P4': 'green'}
    for label, coords in points.items():
        folium.Marker(
            location=coords,
            popup=f"{label}<br>Lat: {coords[0]:.6f}<br>Lon: {coords[1]:.6f}",
            icon=folium.Icon(color=colors[label])
        ).add_to(my_map)
    
    # Add midpoint marker
    folium.Marker(
        location=mid_C_D,
        popup=f"Midpoint P3-P4<br>Lat: {mid_C_D[0]:.6f}<br>Lon: {mid_C_D[1]:.6f}",
        icon=folium.Icon(color='purple', icon='info-sign')
    ).add_to(my_map)

    # Save the map to HTML string
    map_html = my_map._repr_html_()

    # Generate GeoJSON for API responses
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "P1 to P2", "color": "purple", "weight": 4},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in p1_p2_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "P3 to P4", "color": "orange", "weight": 4},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in p3_p4_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Perpendicular Line", "color": "red", "weight": 3},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in perp_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": f"{degree}-Degree Line", "color": "blue", "weight": 3},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in nm_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Perpendicular to P1-P2", "color": "green", "weight": 3},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in perp_nm_geodesic]
                }
            }
        ]
    }
    
    # Add point features
    for label, coords in points.items():
        point_feature = {
            "type": "Feature",
            "properties": {"name": label, "color": colors[label]},
            "geometry": {
                "type": "Point",
                "coordinates": [coords[1], coords[0]]
            }
        }
        geojson_data["features"].append(point_feature)

    # Add important intersection points with accuracy metrics
    key_points = {
        "Initial Intersection": {
            "coords": p1p2_perp_intersection,
            "accuracy": p1p2_dist
        },
        f"{degree}-Degree Line End": {
            "coords": nm_line_end_point,
            "distance": distance_to_degree
        },
        "Final Intersection": {
            "coords": perp_nm_p1p2_intersection,
            "accuracy": p1p2_nm_dist
        },
        "Midpoint P3-P4": {
            "coords": mid_C_D
        }
    }
    
    for label, data in key_points.items():
        coords = data["coords"]
        properties = {"name": label}
        
        if "accuracy" in data:
            properties["accuracy"] = f"{data['accuracy']:.2f}m"
        if "distance" in data:
            properties["distance"] = f"{data['distance']:.2f}nm"
            
        point_feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "Point",
                "coordinates": [coords[1], coords[0]]
            }
        }
        geojson_data["features"].append(point_feature)

    results = {
        'p1p2_perp_intersection': {
            'lat': p1p2_perp_intersection[0],
            'lon': p1p2_perp_intersection[1],
            'accuracy_m': p1p2_dist
        },
        'nm_line_end_point': {
            'lat': nm_line_end_point[0],
            'lon': nm_line_end_point[1],
            'distance_nm': distance_to_degree
        },
        'perp_nm_p1p2_intersection': {
            'lat': perp_nm_p1p2_intersection[0],
            'lon': perp_nm_p1p2_intersection[1],
            'accuracy_m': p1p2_nm_dist
        },
        'midpoint_p3_p4': {
            'lat': mid_C_D[0],
            'lon': mid_C_D[1]
        },
        'p1p2_nm_dist_km': p1p2_nm_dist/1000,
        'distance_to_P1_nm': distance_to_P1_nm,
        'distance_to_P3_nm': distance_to_P3_nm,
        'distance_to_degree': distance_to_degree,
        'geojson': geojson_data,
        'map_html': map_html
    }
    
    return results
@app.route('/api/calculate', methods=['POST'])
@app.route('/api/calculate', methods=['POST'])
def api_calculate():
    try:
        data = request.get_json()
        
        # Get points from request
        P1 = (data.get('P1_lat', 0.0), data.get('P1_lon', 0.0))
        P2 = (data.get('P2_lat', 0.0), data.get('P2_lon', 0.0))
        P3 = (data.get('P3_lat', 0.0), data.get('P3_lon', 0.0))
        P4 = (data.get('P4_lat', 0.0), data.get('P4_lon', 0.0))
        
        # Get other parameters
        TAS = float(data.get('TAS', 220))
        wind_speed = float(data.get('wind_speed', 50))
        wind_direction = float(data.get('wind_direction', 0))  # Add this line
        degree = float(data.get('degree', 330))
        
        # Check if we should include map HTML
        include_map = data.get('include_map', False)
        
        # Perform calculations
        results = calculate_geodesic(P1, P2, P3, P4, TAS, wind_speed, wind_direction, degree)
        
        # Remove map HTML if not requested
        if not include_map and 'map_html' in results:  # Add check if map_html exists
            del results['map_html']
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/calculate_dms', methods=['POST'])
def api_calculate_dms():
    try:
        data = request.get_json()
        
        # Extract DMS coordinates from request
        P1_lat = dms_to_dd(
            data.get('P1_lat_deg', '0'),
            data.get('P1_lat_min', '0'),
            data.get('P1_lat_sec', '0'),
            data.get('P1_lat_dir', 'N')
        )
        
        P1_lon = dms_to_dd(
            data.get('P1_lon_deg', '0'),
            data.get('P1_lon_min', '0'),
            data.get('P1_lon_sec', '0'),
            data.get('P1_lon_dir', 'E')
        )
        
        P2_lat = dms_to_dd(
            data.get('P2_lat_deg', '0'),
            data.get('P2_lat_min', '0'),
            data.get('P2_lat_sec', '0'),
            data.get('P2_lat_dir', 'N')
        )
        
        P2_lon = dms_to_dd(
            data.get('P2_lon_deg', '0'),
            data.get('P2_lon_min', '0'),
            data.get('P2_lon_sec', '0'),
            data.get('P2_lon_dir', 'E')
        )
        
        P3_lat = dms_to_dd(
            data.get('P3_lat_deg', '0'),
            data.get('P3_lat_min', '0'),
            data.get('P3_lat_sec', '0'),
            data.get('P3_lat_dir', 'N')
        )
        
        P3_lon = dms_to_dd(
            data.get('P3_lon_deg', '0'),
            data.get('P3_lon_min', '0'),
            data.get('P3_lon_sec', '0'),
            data.get('P3_lon_dir', 'E')
        )
        
        P4_lat = dms_to_dd(
            data.get('P4_lat_deg', '0'),
            data.get('P4_lat_min', '0'),
            data.get('P4_lat_sec', '0'),
            data.get('P4_lat_dir', 'N')
        )
        
        P4_lon = dms_to_dd(
            data.get('P4_lon_deg', '0'),
            data.get('P4_lon_min', '0'),
            data.get('P4_lon_sec', '0'),
            data.get('P4_lon_dir', 'E')
        )
        
        # Create point tuples
        P1 = (P1_lat, P1_lon)
        P2 = (P2_lat, P2_lon)
        P3 = (P3_lat, P3_lon)
        P4 = (P4_lat, P4_lon)
        
        # Get other parameters
        TAS = float(data.get('TAS', 220))
        wind_speed = float(data.get('wind_speed', 50))
        degree = float(data.get('degree', 330))
        
        # Check if we should include map HTML
        include_map = data.get('include_map', False)
        
        # Perform calculations
        results = calculate_geodesic(P1, P2, P3, P4, TAS, wind_speed, degree)
        
        # Remove map HTML if not requested
        if not include_map:
            del results['map_html']
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/docs', methods=['GET'])
def api_docs():
    docs = {
        "api_version": "1.0",
        "description": "Geodesic Calculation API",
        "endpoints": [
            {
                "path": "/api/calculate",
                "method": "POST",
                "description": "Calculate geodesic intersections using decimal degrees",
                "request_format": {
                    "P1_lat": "float (decimal degrees)",
                    "P1_lon": "float (decimal degrees)",
                    "P2_lat": "float (decimal degrees)",
                    "P2_lon": "float (decimal degrees)",
                    "P3_lat": "float (decimal degrees)",
                    "P3_lon": "float (decimal degrees)",
                    "P4_lat": "float (decimal degrees)",
                    "P4_lon": "float (decimal degrees)",
                    "TAS": "float (default: 220)",
                    "wind_speed": "float (default: 50)",
                    "degree": "float (default: 330)",
                    "include_map": "boolean (default: false)"
                }
            },
            {
                "path": "/api/calculate_dms",
                "method": "POST",
                "description": "Calculate geodesic intersections using DMS format",
                "request_format": {
                    "P1_lat_deg": "string",
                    "P1_lat_min": "string",
                    "P1_lat_sec": "string",
                    "P1_lat_dir": "string (N/S)",
                    "P1_lon_deg": "string",
                    "P1_lon_min": "string",
                    "P1_lon_sec": "string",
                    "P1_lon_dir": "string (E/W)",
                    "(same pattern for P2, P3, P4)": "",
                    "TAS": "float (default: 220)",
                    "wind_speed": "float (default: 50)",
                    "degree": "float (default: 330)",
                    "include_map": "boolean (default: false)"
                }
            }
        ],
        "response_format": {
            "status": "string (success/error)",
            "results": {
                "p1p2_perp_intersection": {"lat": "float", "lon": "float"},
                "nm_line_end_point": {"lat": "float", "lon": "float"},
                "perp_nm_p1p2_intersection": {"lat": "float", "lon": "float"},
                "p1p2_nm_dist_km": "float",
                "distance_to_P1_nm": "float",
                "distance_to_P3_nm": "float",
                "distance_to_degree": "float",
                "geojson": "GeoJSON object for map rendering",
                "map_html": "HTML string (only if include_map=true)"
            }
        },
        "examples": {
            "decimal_degrees": {
                "request": {
                    "P1_lat": -12.42327778,
                    "P1_lon": 130.7398056,
                    "P2_lat": -23.79288889,
                    "P2_lon": 133.8782222,
                    "P3_lat": -14.51919444,
                    "P3_lon": 132.3710000,
                    "P4_lat": -19.63475000,
                    "P4_lon": 134.1819444,
                    "TAS": 220,
                    "wind_speed": 50,
                    "degree": 330,
                   "include_map": False

                }
            },
            "dms_format": {
                "request": {
                    "P1_lat_deg": "12",
                    "P1_lat_min": "25",
                    "P1_lat_sec": "23.8",
                    "P1_lat_dir": "S",
                    "P1_lon_deg": "130",
                    "P1_lon_min": "44",
                    "P1_lon_sec": "23.3",
                    "P1_lon_dir": "E",
                    "(similar pattern for P2, P3, P4)": "...",
                    "TAS": 220,
                    "wind_speed": 50,
                    "degree": 330,
                    "include_map": False

                }
            }
        }
    }
    
    return jsonify(docs)

if __name__ == '__main__':
    app.run(debug=True)