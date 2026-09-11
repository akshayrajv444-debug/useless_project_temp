import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_vada(image_path, pixels_per_cm=100.0):
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Could not read image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours and their hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return None, "No contours found"

    largest_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if hierarchy[0][i][3] == -1 and area > max_area:
            max_area = area
            largest_idx = i
            
    if largest_idx == -1:
        return None, "No vada found"

    hole_idx = -1
    max_hole_area = 0
    for i, cnt in enumerate(contours):
        parent_idx = hierarchy[0][i][3]
        if parent_idx == largest_idx:
            area = cv2.contourArea(cnt)
            if area > max_hole_area:
                max_hole_area = area
                hole_idx = i

    if hole_idx == -1:
        return None, "No inner hole found in the detected vada."

    hole_contour = contours[hole_idx]

    (x, y), radius = cv2.minEnclosingCircle(hole_contour)
    center = (int(x), int(y))
    radius = int(radius)
    area_contour = cv2.contourArea(hole_contour)

    # Convert to standard units (cm)
    radius_cm = radius / pixels_per_cm
    area_cm2 = area_contour / (pixels_per_cm ** 2)

    # Draw visualization
    cv2.circle(img, center, radius, (0, 0, 255), 2)
    cv2.circle(img, center, 2, (0, 255, 0), 3)
    cv2.drawContours(img, [hole_contour], -1, (255, 0, 0), 2)

    return img, {
        "center": center,
        "radius_px": radius,
        "area_px": round(area_contour, 2),
        "radius_cm": round(radius_cm, 2),
        "area_cm2": round(area_cm2, 2)
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get scale from request, default to 100 if not provided
        try:
            pixels_per_cm = float(request.form.get('pixels_per_cm', 100.0))
        except ValueError:
            pixels_per_cm = 100.0

        # Process image
        processed_img, data = process_vada(filepath, pixels_per_cm)
        if processed_img is None:
            return jsonify({'error': data})
            
        # Encode output image to base64 to send it back easily without saving multiple files
        _, buffer = cv2.imencode('.jpg', processed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'data': data
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
