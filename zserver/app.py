import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import colorsys

app = Flask(__name__)
CORS(app)
# CORS(app, resources={r"/analyze": {"origins": "*"}})

# MediaPipe face mesh for detailed facial landmarks
mp_face_mesh = mp.solutions.face_mesh

def rgb_to_color_name(r, g, b):
    """Convert RGB to color name"""
    # Normalize RGB values
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Skin tone classification
    if v > 0.5 and s < 0.3:
        return "Fair"
    elif v > 0.4 and s < 0.4:
        return "Light Medium"
    elif v > 0.3 and s < 0.5:
        return "Medium"
    elif v > 0.2 and s < 0.6:
        return "Olive"
    elif v > 0.1 and s < 0.7:
        return "Tan"
    else:
        return "Deep"

def rgb_to_hair_color(r, g, b):
    """Classify hair color based on RGB values"""
    # Convert RGB to HSV
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Hair color classification based on hue and brightness
    if v < 0.3:
        return "Black"
    elif 0.05 < h < 0.15 and v > 0.4:
        return "Brown"
    elif 0.15 < h < 0.2 and v > 0.5:
        return "Red"
    elif 0.2 < h < 0.4 and v > 0.4:
        return "Blonde"
    else:
        return "Gray"

def rgb_to_eyebrow_color(r, g, b):
    """Classify eyebrow color based on RGB values"""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    if v < 0.3:
        return "Black"
    elif 0.05 < h < 0.15 and v > 0.4:
        return "Brown"
    elif 0.15 < h < 0.2 and v > 0.5:
        return "Red"
    elif 0.2 < h < 0.4 and v > 0.4:
        return "Blonde"
    else:
        return "Gray"

def determine_face_shape(landmarks, image_shape):
    """
    Determine face shape based on facial measurements and proportions
    """
    h, w = image_shape[:2]
    
    # Extract key measurements
    # Face width at different points
    forehead_width = abs(landmarks.landmark[251].x - landmarks.landmark[21].x) * w
    cheekbone_width = abs(landmarks.landmark[234].x - landmarks.landmark[454].x) * w
    jawline_width = abs(landmarks.landmark[172].x - landmarks.landmark[397].x) * w
    
    # Face length measurements
    face_length = abs(landmarks.landmark[10].y - landmarks.landmark[152].y) * h
    forehead_height = abs(landmarks.landmark[10].y - landmarks.landmark[109].y) * h
    jaw_height = abs(landmarks.landmark[152].y - landmarks.landmark[172].y) * h
    
    # Calculate ratios
    width_to_length_ratio = cheekbone_width / face_length
    forehead_to_jaw_ratio = forehead_width / jawline_width
    cheekbone_to_jaw_ratio = cheekbone_width / jawline_width
    forehead_height_ratio = forehead_height / face_length
    jaw_height_ratio = jaw_height / face_length
    
    # Face shape classification logic
    if width_to_length_ratio <= 0.75:
        if forehead_to_jaw_ratio > 1.1:
            return "Heart"
        else:
            return "Oblong"
            
    elif width_to_length_ratio >= 0.85:
        if cheekbone_to_jaw_ratio > 1.1:
            return "Round"
        else:
            return "Square"
            
    else:  # Medium width-to-length ratio
        if forehead_to_jaw_ratio > 1.1 and cheekbone_to_jaw_ratio > 1.1:
            return "Diamond"
        elif forehead_to_jaw_ratio < 0.9 and cheekbone_to_jaw_ratio > 1.1:
            return "Triangle"
        elif abs(forehead_to_jaw_ratio - 1.0) < 0.1 and cheekbone_to_jaw_ratio > 1.05:
            return "Oval"
        elif forehead_height_ratio > 0.35:
            if jaw_height_ratio < 0.3:
                return "Heart"
            else:
                return "Rectangle"
        else:
            return "Oval"  # Default to oval if no other clear match

def analyze_face(image_path):
    """Comprehensive face analysis function"""
    # Read image
    image = cv2.imread(image_path)
    
    # Face mesh detection
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # Convert image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        # If no face detected
        if not results.multi_face_landmarks:
            return None
        
        # Extract facial landmarks
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        
        # Skin Tone Analysis
        forehead_x_min = int(landmarks.landmark[10].x * w - 0.1 * w)
        forehead_x_max = int(landmarks.landmark[10].x * w + 0.1 * w)
        forehead_y_min = int(landmarks.landmark[10].y * h - 0.05 * h)
        forehead_y_max = int(landmarks.landmark[10].y * h + 0.02 * h)
        
        forehead_x_min = max(0, forehead_x_min)
        forehead_x_max = min(w, forehead_x_max)
        forehead_y_min = max(0, forehead_y_min)
        forehead_y_max = min(h, forehead_y_max)
        
        skin_roi = image[forehead_y_min:forehead_y_max, forehead_x_min:forehead_x_max]
        avg_skin_color = np.mean(skin_roi, axis=(0, 1)) if skin_roi.size > 0 else np.array([0, 0, 0])
        
        # Hair Color Analysis
        hair_x_min = int(landmarks.landmark[10].x * w - 0.15 * w)
        hair_x_max = int(landmarks.landmark[10].x * w + 0.15 * w)
        hair_y_min = int(landmarks.landmark[10].y * h - 0.2 * h)
        hair_y_max = int(landmarks.landmark[10].y * h - 0.05 * h)
        
        hair_x_min = max(0, hair_x_min)
        hair_x_max = min(w, hair_x_max)
        hair_y_min = max(0, hair_y_min)
        hair_y_max = min(h, hair_y_max)
        
        hair_roi = image[hair_y_min:hair_y_max, hair_x_min:hair_x_max]
        avg_hair_color = np.mean(hair_roi, axis=(0, 1)) if hair_roi.size > 0 else np.array([0, 0, 0])
        
        # Eyebrow Color Analysis
        eyebrow_x_min = int(landmarks.landmark[19].x * w - 0.1 * w)
        eyebrow_x_max = int(landmarks.landmark[24].x * w + 0.1 * w)
        eyebrow_y_min = int(min(landmarks.landmark[19].y, landmarks.landmark[24].y) * h - 0.1 * h)
        eyebrow_y_max = int(min(landmarks.landmark[19].y, landmarks.landmark[24].y) * h + 0.05 * h)
        
        eyebrow_x_min = max(0, eyebrow_x_min)
        eyebrow_x_max = min(w, eyebrow_x_max)
        eyebrow_y_min = max(0, eyebrow_y_min)
        eyebrow_y_max = min(h, eyebrow_y_max)
        
        eyebrow_roi = image[eyebrow_y_min:eyebrow_y_max, eyebrow_x_min:eyebrow_x_max]
        avg_eyebrow_color = np.mean(eyebrow_roi, axis=(0, 1)) if eyebrow_roi.size > 0 else np.array([0, 0, 0])
        
        # Lip analysis
        lip_x_min = int(min(landmarks.landmark[0].x, landmarks.landmark[17].x) * w) - 5
        lip_x_max = int(max(landmarks.landmark[0].x, landmarks.landmark[17].x) * w) + 5
        lip_y_min = int(min(landmarks.landmark[13].y, landmarks.landmark[14].y) * h) - 5
        lip_y_max = int(max(landmarks.landmark[13].y, landmarks.landmark[14].y) * h) + 5
        
        lip_x_min = max(0, lip_x_min)
        lip_x_max = min(w, lip_x_max)
        lip_y_min = max(0, lip_y_min)
        lip_y_max = min(h, lip_y_max)
        
        lip_roi = image[lip_y_min:lip_y_max, lip_x_min:lip_x_max]
        avg_lip_color = np.mean(lip_roi, axis=(0, 1)) if lip_roi.size > 0 else np.array([0, 0, 0])
        
        # Measurements
        lip_width = abs(landmarks.landmark[13].x - landmarks.landmark[14].x) * w
        eye_distance = abs(landmarks.landmark[33].x - landmarks.landmark[263].x) * w
        
        # Determine face shape using the new function
        face_shape = determine_face_shape(landmarks, image.shape)
        
        # Face shape description and characteristics
        face_shape_characteristics = {
            "Oval": "Balanced proportions with a gently rounded hairline and jawline",
            "Round": "Similar length and width with soft angles and full cheeks",
            "Square": "Strong jawline with similar face width and length",
            "Heart": "Wider forehead tapering to a narrower jawline",
            "Diamond": "Narrow forehead, wide cheekbones, and narrow jawline",
            "Triangle": "Narrow forehead with a wider jawline",
            "Oblong": "Face length notably longer than width with straight sides",
            "Rectangle": "Similar to oblong but with more angular features"
        }
        
        return {
            "skin_tone": {
                "color": f"rgb({avg_skin_color[2]:.0f}, {avg_skin_color[1]:.0f}, {avg_skin_color[0]:.0f})",
                "description": rgb_to_color_name(avg_skin_color[2], avg_skin_color[1], avg_skin_color[0])
            },
            "hair_color": {
                "color": f"rgb({avg_hair_color[2]:.0f}, {avg_hair_color[1]:.0f}, {avg_hair_color[0]:.0f})",
                "description": rgb_to_hair_color(avg_hair_color[2], avg_hair_color[1], avg_hair_color[0])
            },
            "eyebrow_color": {
                "color": f"rgb({avg_eyebrow_color[2]:.0f}, {avg_eyebrow_color[1]:.0f}, {avg_eyebrow_color[0]:.0f})",
                "description": rgb_to_eyebrow_color(avg_eyebrow_color[2], avg_eyebrow_color[1], avg_eyebrow_color[0])
            },
            "lip_details": {
                "fullness": lip_width,
                "color": f"rgb({avg_lip_color[2]:.0f}, {avg_lip_color[1]:.0f}, {avg_lip_color[0]:.0f})"
            },
            "face_shape": {
                "shape": face_shape,
                "description": face_shape_characteristics[face_shape]
            },
            "eye_details": {
                "distance": eye_distance
            }
        }

@app.route('/analyze', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Save temporarily
    filename = "temp_upload.jpg"
    file.save(filename)
    
    # Analyze
    results = analyze_face(filename)
    
    # Clean up
    os.remove(filename)
    
    if results is None:
        return jsonify({"error": "No face detected"}), 400
    
    return jsonify(results)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)