import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Processes the wound and mask image
def process_image(image_path, mask_path, resize_shape=(400, 300), kernel_size=(10, 10), num_clusters=4):
    # load images and masks
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    # convert images and masks to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize images and masks
    image_resized = cv2.resize(image, resize_shape)
    mask_resized = cv2.resize(mask, resize_shape)

    # replace black values in mask
    mask_replaced = replace_black_value(mask_resized)

    # clean the mask of artifacts and noise
    kernel = np.ones(kernel_size, np.uint8)
    cleaned_mask = cv2.morphologyEx(mask_replaced, cv2.MORPH_CLOSE, kernel)

    # extract color information from masks within white areas
    white_pixels = np.where(cleaned_mask == 255)
    masked_image = image_resized[white_pixels]

    # colour quantization to reduce colors
    quantized_image, centers = quantize_image(masked_image, num_clusters)

    return image_resized, cleaned_mask, quantized_image, centers

# replaces the black values in the mask with #0513d4 blue
def replace_black_value(mask):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    black_pixels = np.all(mask_rgb == [0, 0, 0], axis=2)
    mask_rgb[black_pixels] = [5, 19, 212]  # RGB values for #0513d4
    return mask_rgb

# quantize image for color reduction
def quantize_image(image, num_clusters=4):
    pixels = image.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized_image = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    return quantized_image, centers

# calculates the percentage of colors in image
def calculate_color_percentage(image, exclude_color= np.array([24, 23, 172]), distance_threshold=50):
    unique_colors, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    color_percentages = []

    for color, count in zip(unique_colors, counts):
        color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        if exclude_color is not None:
            color_distance = np.linalg.norm(color - exclude_color)
            if color_distance <= distance_threshold:
                continue
        percentage = (count / np.sum(counts)) * 100
        color_percentages.append((color_hex, percentage))

    return color_percentages

# calculates distance for grouping colors
def calculate_distance(colors1, colors2):
    return np.linalg.norm(colors1 - colors2)

# NOTE: Global variable, as this dictionary is referenced everywhere
wound_dictionary = {
    'Slough': np.array([255, 255, 0]),
    'Granulation': np.array([255, 0, 0]),
    'Epithelial': np.array([255, 192, 203]),
    'Necrotic (Late)': np.array([0, 0, 0]),
    'Necrotic (Early)': np.array([165, 42, 42])
    
}
# groups colors into categories
def group_colors(color_percentages, color_similarity_threshold=200):
    color_groups = {}

    for color, percentage in color_percentages:
        color_rgb = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])

        for group_name, group_color in wound_dictionary.items():
            dist = calculate_distance(color_rgb, group_color)
            if dist < color_similarity_threshold:
                if group_name in color_groups:
                    color_groups[group_name] += percentage
                else:
                    color_groups[group_name] = percentage
                break

    for group_name in wound_dictionary.keys():
        if group_name not in color_groups:
            color_groups[group_name] = 0

    grouped_color_groups = [(group_name, percentage) for group_name, percentage in color_groups.items()]
    return grouped_color_groups

# plots the pie chart with wound dictionary
def pie_chart(colors, percentages, grouped_color_groups, title):
    # convert colors to RGB values for handle/colorbar
    rgb_colors = [wound_dictionary.get(group, [255, 255, 255]) for group, _ in grouped_color_groups]
    rgb_colors = np.array(rgb_colors) / 255.0

    plt.pie(percentages, colors=colors, autopct='%1.1f%%')
    plt.title(title)

    # labels with group color percentages
    label_text = [f'{label} ({group_percentage:.1f}%)' for (label, group_percentage) in grouped_color_groups]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in rgb_colors]
    plt.legend(legend_handles, label_text, loc='lower center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)

    plt.show()


# main code that analyses wounds
def wound_analysis(image_paths, mask_paths):
    num_wounds = len(image_paths)
    fig, axs = plt.subplots(1, num_wounds, figsize=(6*num_wounds, 6))

    for i in range(num_wounds):
        image_path = image_paths[i]
        mask_path = mask_paths[i]

        image, mask, quantized_image, centers = process_image(image_path, mask_path)
        color_percentages = calculate_color_percentage(quantized_image)
        grouped_color_groups = group_colors(color_percentages)

        colors = [color for color, _ in color_percentages]
        percentages = [percentage for _, percentage in color_percentages]

        group_names = [group_name for group_name, _ in grouped_color_groups]
        group_percentages = [percentage for _, percentage in grouped_color_groups]
        rgb_colors = [wound_dictionary.get(group, [255, 255, 255]) for group, _ in grouped_color_groups]
        rgb_colors = np.array(rgb_colors) / 255.0

        ax = axs[i] if num_wounds > 1 else axs
        ax.pie(percentages, colors=colors, autopct='%1.1f%%')
        ax.set_title('Wound {}'.format(i + 1))

        label_text = [f'{label} ({group_percentage:.1f}%)' for (label, group_percentage) in grouped_color_groups]
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in rgb_colors]
        ax.legend(legend_handles, label_text, loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True)

    plt.show()


# call here to analyze multiple wounds and masks
image_paths = ['results/heal_test/wound_image_heal_test_2023-05-14_21-52-33.jpg', 'results/heal_test/wound_image_heal_test_2023-05-14_22-03-54.jpg']
mask_paths = ['results/heal_test/wound_mask_heal_test_2023-05-14_21-52-33.jpg', 'results/heal_test/wound_mask_heal_test_2023-05-14_22-03-54.jpg']
wound_analysis(image_paths, mask_paths)