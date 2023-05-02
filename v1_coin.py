import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from v1_border import extract_blue_contour, process_image

# Diaplay detection result
def display_coin_detection(image, coin_detected, wound_area):
    if coin_detected is not None:
        x, y, radius = coin_detected
        cv2.circle(image, (x, y), radius, (0, 255, 0), 2)

        if wound_area is not None:  # Only draw the wound area if it is provided
            for point in wound_area:
                x, y = point
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    plt.imshow(image)
    plt.show()

def calculate_actual_wound_area(ratio_coin, coin_actual_area, wound_ratio):
    # ratio = circle_area / image_area
    # print(f"Ratio of circle area to image area: {ratio:.6f}")
    # print(f"Circle diameter: {diameter}")
    # coin_actual_size = 2  # Diameter of the 2-dollar coin in centimeters
    wound_actual_size = coin_actual_area / ratio_coin * wound_ratio
    return wound_actual_size

def calculate_ratio_wound_image(wound_pixel,image,image_path):
    wound_pixel, wound_ratio = process_image(image,image_path)
    image_area = image.shape[0] * image.shape[1]
    # ratio_wound = wound_pixel / image_area
    return wound_ratio

def calculate_actual_coin_area(diameter):
    coin_actual_area = math.pi * (diameter / 2) ** 2  # Area of a 2-dollar coin in millimeters squared
    return coin_actual_area

# Function to detect the 2-dollar coin using the Hough Circle Transform
def detect_coin(image, min_radius, max_radius):
    if image is None:
        print("Error loading the image.")
        return None

    # Convert image to 8-bit format
    image = np.uint8(image)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9),1.5)
    canny = cv2.Canny(blur, 100, 255)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=40, minRadius=0, maxRadius=0)

    # if circles is not None:
    #     circles = np.round(circles[0, :]).astype("int")
    #     return circles[0]  # Return the first detected circle
    # else:
    #     return None
    if circles is not None: 
        circles = np.uint16(np.around(circles))
        
        # Find the circle with the highest confidence (strongest edge)
        best_circle = None
        best_param2 = 0
        
        for i in circles[0, :]:
            x, y, radius = i
            circle_canny = canny[y - radius:y + radius, x - radius:x + radius]
            
            # Calculate the confidence score for the circle based on edge intensity
            confidence = np.sum(circle_canny) / (circle_canny.shape[0] * circle_canny.shape[1])
            
            if confidence > best_param2:
                best_circle = i
                best_param2 = confidence
        
        # Draw the best circle
        if best_circle is not None:
            # #ratio of circle area to img area
            circle_area = math.pi * radius**2
            image_area = image.shape[0] * image.shape[1]
            ratio = circle_area / image_area
            # print(f"Ratio of circle area to image area: {ratio:.6f}")
            return best_circle, ratio
            # x, y, radius = best_circle
            # diameter = 2 * radius
            # print(f"Circle diameter: {diameter}")
            # cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
            # cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        # # Size of the image
        # height, width, channels = img.shape
        # print(f"Image size: {width}x{height}")
    

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

def estimate_actual_size(image, binary_mask, coin_detected):
    if coin_detected is None:
        print("No coin detected. Unable to estimate actual area.")
        return None
    
    _, _, coin_radius = coin_detected
    
    # Get the contour from the mask. This is probably not necessary here. Ideally it would be done elsewhere
    contour = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Generate a Rect around the contour
    # cv2.boundingRect() will draw a rect around the contour, but is bound to images X and Y dimensions.
    # cv2.minAreaRect() will draw a rect that will rotate to make the smallest possible area around the contour
    rect = cv2.minAreaRect(contour)
    
    # We could display the original image here, with a box drawn over it. But I would prefer to do that in a separate function. Discuss with team
    # Here is the code, just in case:
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    
    # Calculate the real-world measurements
    # Temorary! Do elsewhere!  
    coin_radius_mm = 14.25 # Radius of an Australian $2 coin in millimeters
    pixels_to_metric_ratio = coin_radius / coin_radius_mm
    rect_size_mm = rect[1] / pixels_to_metric_ratio
    
    print(f'The wound is {rect_size_mm[1]}mm wide, and {rect_size_mm[1]}mm long')   # Length and width are arbitrary.
                                                                                    # Could be replaced so that the longest measurement is length
                                                                                    # But I don't think it's necessary
    
    
    return rect



input_directory = 'fake_evaluation/'

# Coin size
COIN_DIAMETER_MM = 28.0  # Diameter of a 2-dollar coin in millimeters

# Process images in the input directory
for image_file in os.listdir(input_directory):
    if image_file.endswith('.jpg'):
        input_path = os.path.join(input_directory, image_file)

        # Read image
        image_test = cv2.imread(input_path)

        # Pixel count of the wound
        wound_pixel = process_image(image_test, input_path)

        # Detect the 2-dollar coin
        best_circle, ratio_coin = detect_coin(image_test, min_radius=30, max_radius=100)
        print(best_circle)

        # Get the actual pixel of wound area
        coin_actual_area = calculate_actual_coin_area(COIN_DIAMETER_MM)
        contour_image, pixel_count, ratio_wound = extract_blue_contour(input_path)
        wound_area = calculate_actual_wound_area(ratio_coin, ratio_wound, coin_actual_area)
        print(f"coin area is :{coin_actual_area}")
        print(f"coin/image ratio is :{ratio_coin}")
        print(f"wound/image ratio is :{ratio_wound}")
        print(f"wound area is :{wound_area}")

        # Display the coin detection result
        display_coin_detection(image_test, best_circle, None)
