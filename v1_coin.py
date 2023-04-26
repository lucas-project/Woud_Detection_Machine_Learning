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
def display_coin_detection(image, best_circle, wound_area=None):
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

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7),1.2)
    canny = cv2.Canny(blur, 50, 255)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=30, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

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
            return best_circle
            # x, y, radius = best_circle
            # diameter = 2 * radius
            # print(f"Circle diameter: {diameter}")
            # cv2.circle(image, (x, y), radius, (0, 255, 0), 2)
            # cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

        # # Size of the image
        # height, width, channels = img.shape
        # print(f"Image size: {width}x{height}")

        # #ratio of circle area to img area
        # circle_area = math.pi * radius**2
        # image_area = img.shape[0] * img.shape[1]
        # ratio = circle_area / image_area
        # print(f"Ratio of circle area to image area: {ratio:.6f}")
    

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

### Just testing some alternative functions. Can ignore for now ###

def pixels_to_metric(pixel_size, real_size):
	return pixel_size / real_size
	
def calculate_area(image, wound_area_pixels, coin_detected):
	if coin_detected is None:
		print("No coin detected. Unable to estimate actual area.")
		return None
	
	_, _, coin_radius = coin_detected
	
	coin_area_pixels = np.pi * (coin_radius ** 2)
	coin_area_cm2 = 6.375 # Area of the 2-dollar coin in square centimeters

	return wound_area_pixels * pixels_to_metric(coin_area_cm2, coin_area_pixels)
	
def calculate_length(image, wound_length_pixels, coin_detected):
	if coin_detected is None:
		print("No coin detected. Unable to estimate actual length.")
		return None
		
	_, _, coin_radius = coin_detected
	
	coin_radius_cm = 1.025 # Radius of the $2 coin in centimeters
	
	return wound_length_pixels * pixels_to_metric(coin_radius_cm * coin_radius)
