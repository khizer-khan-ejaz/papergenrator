from flask import Flask, request, jsonify,render_template
import folium
from geographiclib.geodesic import Geodesic
import scipy.optimize as optimize
from folium.map import Figure
from flask_cors import CORS

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
        Find the intersection of two geodesic lines using optimization
        """
        def distance_between_lines(params):
            s1, s2 = params
            line1 = geod.InverseLine(line1_start[0], line1_start[1], 
                                     line1_end[0], line1_end[1])
            line2 = geod.InverseLine(line2_start[0], line2_start[1], 
                                     line2_end[0], line2_end[1])
            point1 = line1.Position(s1 * line1.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            point2 = line2.Position(s2 * line2.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            g = geod.Inverse(point1['lat2'], point1['lon2'], 
                             point2['lat2'], point2['lon2'])
            return g['s12']
        
        initial_guess = [0.5, 0.5]
        result = optimize.minimize(
            distance_between_lines, 
            initial_guess, 
            method='Nelder-Mead',
            bounds=[(0, 1), (0, 1)]
        )
        line1 = geod.InverseLine(line1_start[0], line1_start[1], 
                                 line1_end[0], line1_end[1])
        line2 = geod.InverseLine(line2_start[0], line2_start[1], 
                                 line2_end[0], line2_end[1])
        point1 = line1.Position(result.x[0] * line1.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        point2 = line2.Position(result.x[1] * line2.s13, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
        intersection = (
            (point1['lat2'] + point2['lat2']) / 2, 
            (point1['lon2'] + point2['lon2']) / 2
        )
        g = geod.Inverse(point1['lat2'], point1['lon2'], 
                         point2['lat2'], point2['lon2'])
        return intersection, g['s12']

    def generate_geodesic_points(start, end, num_points=100):
        """Generate points along a geodesic line"""
        line = geod.InverseLine(start[0], start[1], end[0], end[1])
        ds = line.s13 / (num_points - 1)
        points = []
        for i in range(num_points):
            s = ds * i
            g = line.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            points.append((g['lat2'], g['lon2']))
        return points, line

    # Calculate midpoint of C-D
    mid_C_D = geodesic_intersection(P3, P4, P3, P4)[0]

    # Calculate perpendicular line
    g_CD = geod.Inverse(P3[0], P3[1], P4[0], P4[1])
    bearing_CD = g_CD['azi1']
    perp_bearing = (bearing_CD + 90) % 360

    # Generate lines
    p1_p2_geodesic, p1_p2_line = generate_geodesic_points(P1, P2)
    p3_p4_geodesic, p3_p4_line = generate_geodesic_points(P3, P4)

    # Perpendicular line generation
    perp_distance = 500000  # 500 km in meters
    perp_point1 = geod.Direct(mid_C_D[0], mid_C_D[1], perp_bearing, perp_distance)
    perp_point1 = (perp_point1['lat2'], perp_point1['lon2'])
    perp_point2 = geod.Direct(mid_C_D[0], mid_C_D[1], (perp_bearing + 180) % 360, perp_distance)
    perp_point2 = (perp_point2['lat2'], perp_point2['lon2'])
    perp_geodesic, perp_line = generate_geodesic_points(perp_point1, perp_point2)

    # Find precise intersections
    p1p2_perp_intersection, p1p2_dist = geodesic_intersection(P1, P2, perp_point1, perp_point2)
    distance_to_P3_nm = (geod.Inverse(p1p2_perp_intersection[0], p1p2_perp_intersection[1], 
                                      P3[0], P3[1])['s12'] / 1000) * 0.539957

    distance_to_degree = (distance_to_P3_nm/TAS)*wind_speed
    
    # Generate 330-degree line from intersection
    line_distance = distance_to_degree * 1852  # Convert to meters
    nm_line_point = geod.Direct(p1p2_perp_intersection[0], p1p2_perp_intersection[1], degree, line_distance)
    nm_line_end_point = (nm_line_point['lat2'], nm_line_point['lon2'])
    nm_geodesic, nm_line = generate_geodesic_points(p1p2_perp_intersection, nm_line_end_point)

    # Generate perpendicular line from 330-degree line end point towards P1-P2 line
    g_p1p2 = geod.Inverse(P1[0], P1[1], P2[0], P2[1])
    p1p2_bearing = g_p1p2['azi1']
    perp_to_p1p2_bearing = (p1p2_bearing + 90) % 360

    perp_nm_distance = 1000000  # 1000 km in meters
    perp_nm_point1 = geod.Direct(nm_line_end_point[0], nm_line_end_point[1], perp_to_p1p2_bearing, perp_nm_distance)
    perp_nm_point1 = (perp_nm_point1['lat2'], perp_nm_point1['lon2'])
    perp_nm_point2 = geod.Direct(nm_line_end_point[0], nm_line_end_point[1], (perp_to_p1p2_bearing + 180) % 360, perp_nm_distance)
    perp_nm_point2 = (perp_nm_point2['lat2'], perp_nm_point2['lon2'])
    perp_nm_geodesic, perp_nm_line = generate_geodesic_points(perp_nm_point1, perp_nm_point2)

    # Find intersection of perpendicular line with P1-P2 line
    perp_nm_p1p2_intersection, p1p2_nm_dist = geodesic_intersection(P1, P2, perp_nm_point1, perp_nm_point2)

    # Calculate distance from perpendicular intersection to P1
    distance_to_P1 = geod.Inverse(perp_nm_p1p2_intersection[0], perp_nm_p1p2_intersection[1], 
                                  P1[0], P1[1])['s12'] / 1000  # Convert to kilometers

    # Create map for optional HTML return
    my_map = folium.Map(location=p1p2_perp_intersection, zoom_start=6, tiles='OpenStreetMap')

    # Add lines
    folium.PolyLine(p1_p2_geodesic, color='purple', weight=3, tooltip='P1 to P2').add_to(my_map)
    folium.PolyLine(p3_p4_geodesic, color='orange', weight=3, tooltip='P3 to P4').add_to(my_map)
    folium.PolyLine(perp_geodesic, color='red', weight=3, tooltip='Perpendicular Line').add_to(my_map)
    folium.PolyLine(nm_geodesic, color='blue', weight=3, tooltip=f'{degree}-Degree Line').add_to(my_map)
    folium.PolyLine(perp_nm_geodesic, color='green', weight=3, tooltip='Perpendicular to P1-P2').add_to(my_map)

    # Add markers
    folium.Marker(
        location=p1p2_perp_intersection,
        popup=f'Initial Intersection\nLat: {p1p2_perp_intersection[0]:.6f}\nLon: {p1p2_perp_intersection[1]:.6f}',
        icon=folium.Icon(color='black', icon='info-sign')
    ).add_to(my_map)

    folium.Marker(
        location=nm_line_end_point,
        popup=f'{degree}-Degree Line End\nLat: {nm_line_end_point[0]:.6f}\nLon: {nm_line_end_point[1]:.6f}',
        icon=folium.Icon(color='blue', icon='arrow-up')
    ).add_to(my_map)

    folium.Marker(
        location=perp_nm_p1p2_intersection,
        popup=f'Perpendicular Intersection\nLat: {perp_nm_p1p2_intersection[0]:.6f}\nLon: {perp_nm_p1p2_intersection[1]:.6f}',
        icon=folium.Icon(color='green', icon='crosshair')
    ).add_to(my_map)

    # Add point markers
    points = {'P1': P1, 'P2': P2, 'C (P3)': P3, 'D (P4)': P4}
    colors = {'P1': 'blue', 'P2': 'blue', 'C (P3)': 'green', 'D (P4)': 'green'}
    for label, coords in points.items():
        folium.Marker(
            location=coords,
            popup=f"{label}\nLat: {coords[0]:.6f}\nLon: {coords[1]:.6f}",
            icon=folium.Icon(color=colors[label])
        ).add_to(my_map)

    # Save the map to HTML string
    map_html = my_map._repr_html_()

    # Generate GeoJSON for API responses
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "P1 to P2", "color": "purple"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in p1_p2_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "P3 to P4", "color": "orange"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in p3_p4_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Perpendicular Line", "color": "red"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in perp_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": f"{degree}-Degree Line", "color": "blue"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p[1], p[0]] for p in nm_geodesic]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Perpendicular to P1-P2", "color": "green"},
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

    # Add important intersection points
    key_points = {
        "Initial Intersection": p1p2_perp_intersection,
        f"{degree}-Degree Line End": nm_line_end_point,
        "Perpendicular Intersection": perp_nm_p1p2_intersection
    }
    
    for label, coords in key_points.items():
        point_feature = {
            "type": "Feature",
            "properties": {"name": label},
            "geometry": {
                "type": "Point",
                "coordinates": [coords[1], coords[0]]
            }
        }
        geojson_data["features"].append(point_feature)

    results = {
        'p1p2_perp_intersection': {
            'lat': p1p2_perp_intersection[0],
            'lon': p1p2_perp_intersection[1]
        },
        'nm_line_end_point': {
            'lat': nm_line_end_point[0],
            'lon': nm_line_end_point[1]
        },
        'perp_nm_p1p2_intersection': {
            'lat': perp_nm_p1p2_intersection[0],
            'lon': perp_nm_p1p2_intersection[1]
        },
        'p1p2_nm_dist_km': p1p2_nm_dist/1000,
        'distance_to_P1_nm': distance_to_P1*0.539957,
        'distance_to_P3_nm': distance_to_P3_nm,
        'distance_to_degree': distance_to_degree,
        'geojson': geojson_data,
        'map_html': map_html,
    }
    
    return results

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
        }
    }
    
    return jsonify(docs)

# Base route for Vercel
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')  # This will look for an HTML file in a templates folder
