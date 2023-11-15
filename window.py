import cv2
import numpy as np

def detect_windows(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(blurred, 150, 450)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has four corners (potentially a rectangle)
        if len(approx) == 4:
            # Draw a rectangle around the identified window
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Window Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "projects\FireFlyer\w576.jpg"
detect_windows(image_path)