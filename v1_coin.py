import numpy as np
import cv2
import matplotlib.pyplot as plt

# Diaplay detection result
# def display_coin_detection(image, coin_detected, wound_area=None):
#     if coin_detected is not None:
#         x, y, radius = coin_detected
#         cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

#         if wound_area is not None:  # Only draw the wound area if it is provided
#             for point in wound_area:
#                 x, y = point
#                 cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

#     plt.imshow(image)
#     plt.show()
def display_coin_detection(image, coin_detected, wound_area=None):
    if best_circle is not None:
        x, y, radius = best_circle
        diameter = 2 * radius
        # print(f"Circle diameter: {diameter}")
        cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

    if wound_area is not None:  # Only draw the wound area if it is provided
        for point in wound_area:
            x, y = point
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    plt.imshow(image)
    plt.show()


# Function to detect the 2-dollar coin using the Hough Circle Transform
def detect_coin(image, min_radius, max_radius):
    if image is None:
        print("Error loading the image.")
        return None

    # Convert image to 8-bit format
    image = np.uint8(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return circles[0]  # Return the first detected circle
    else:
        return None
    

# Function to calculate the wound area in pixels
def calculate_wound_area(mask):
    return np.sum(mask == 255)

def estimate_actual_area(image, wound_area_pixels, coin_detected):
    if coin_detected is None:
        print("No coin detected. Unable to estimate actual area.")
        return None

    _, _, coin_radius = coin_detected

    coin_area_pixels = np.pi * coin_radius ** 2
    coin_area_cm2 = 4.07  # Area of the 2-dollar coin in square centimeters

    pixel_to_cm2_ratio = coin_area_cm2 / coin_area_pixels
    wound_area_cm2 = wound_area_pixels * pixel_to_cm2_ratio

    return wound_area_cm2