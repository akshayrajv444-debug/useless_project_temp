import cv2
import numpy as np
import argparse

def analyze_vada(image_path):
    """
    Detects a Uzhunnuvada (Medu Vada) in an image and calculates its radius and area.
    Assumes a reasonably contrasting background.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to separate the vada from the background
    # Using Otsu's thresholding; assuming inverted binary (vada is white, background is black)
    # You might need to change cv2.THRESH_BINARY_INV to cv2.THRESH_BINARY depending on your background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours and their hierarchy to detect holes (children)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("No contours found.")
        return

    # Find the largest outer contour (the vada)
    largest_idx = -1
    max_area = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Parent == -1 means it's an outer contour
        if hierarchy[0][i][3] == -1 and area > max_area:
            max_area = area
            largest_idx = i
            
    if largest_idx == -1:
        print("No vada found.")
        return

    # Find the largest inner contour (the hole) that is a child of the largest outer contour
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
        print("No inner hole found in the detected vada.")
        return

    hole_contour = contours[hole_idx]

    # Fit a minimum enclosing circle to the hole contour to get the radius
    (x, y), radius = cv2.minEnclosingCircle(hole_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # Area based on the actual contour shape
    area_contour = cv2.contourArea(hole_contour)

    print(f"--- Results for {image_path} ---")
    print(f"Inner Hole Center (x, y): {center}")
    print(f"Inner Hole Radius (pixels): {radius}")
    print(f"Inner Hole Area (pixels^2): {area_contour:.2f}")
    print("-----------------------------------")

    # Draw the circle and center on the original image for visualization
    cv2.circle(img, center, radius, (0, 0, 255), 2)
    cv2.circle(img, center, 2, (0, 255, 0), 3)

    # Draw the contour itself
    cv2.drawContours(img, [hole_contour], -1, (255, 0, 0), 2)

    # Optionally resize for display if the image is too large
    # scale_percent = 50 
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

    # Save the visualization instead of showing it (since we are in a headless environment)
    output_path = "output_" + image_path.split('/')[-1] if '/' in image_path else "output_" + image_path
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect radius and area of a Uzhunnuvada.")
    parser.add_argument("-i", "--image", type=str, default="vada.jpg", help="Path to the Uzhunnuvada image")
    args = parser.parse_args()
    
    analyze_vada(args.image)
