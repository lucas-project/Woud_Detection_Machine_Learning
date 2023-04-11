import numpy as np
import cv2


# Get all colours from a image
def extract_color_information(original_image, mask):
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    return masked_image

# Colour quantization using K-means (Reducing number of colour in an image)
def quantize_image(image, num_clusters=8):
    pixels = image.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return quantized_image, centers

# Calculate colour percentage
def calculate_color_percentage(image, centers):
    unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    color_percentages = []

    for color, count in zip(unique_colors, counts):
        color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        percentage = (count / np.sum(counts)) * 100
        color_percentages.append((color_hex, percentage))

    return color_percentages



