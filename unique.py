import cv2
from skimage.metrics import structural_similarity as compare_ssim
import os
from PIL import Image
import shutil
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


def load_images_from_folder(folder, size=(300, 300)):
    images = []
    sorted_filenames = sorted(os.listdir(folder), key=lambda x: int(x.split("_")[1]))
    for filename in tqdm(sorted_filenames):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            images.append((filename, img))
    return images


def crop(filepath, mouse_position):
    img = cv2.imread(filepath, 0)
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    # Combine the horizontal and vertical edges
    scharr_combined = cv2.magnitude(scharrx, scharry)
    scharr_combined = cv2.convertScaleAbs(scharr_combined)

    thresh = cv2.adaptiveThreshold(
        scharr_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_size = 500000

    # alternative
    contours = [c for c in contours if cv2.contourArea(c) > min_size]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assume the foreground window is the largest contour near the center
    center_x, center_y = mouse_position.x, mouse_position.y
    min_distance_to_center = float("inf")
    foreground_contour = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_center_x, contour_center_y = x + w // 2, y + h // 2
        distance_to_center = np.sqrt(
            (center_x - contour_center_x) ** 2 + (center_y - contour_center_y) ** 2
        )

        if distance_to_center < min_distance_to_center:
            min_distance_to_center = distance_to_center
            foreground_contour = contour

    # Draw the foreground contour
    if foreground_contour is not None:
        x, y, w, h = cv2.boundingRect(foreground_contour)
    else:
        x, y, w, h = 0, 0, 0, 0

    cropped_img = img[y : y + h, x : x + w]

    return cropped_img


def crop_then_filter(folder):
    images, filenames = [], []
    sorted_filenames = sorted(os.listdir(folder), key=lambda x: int(x.split("_")[1]))

    for filename in tqdm(sorted_filenames):
        x, y = map(int, filename.replace(".png", "").split("_")[-2:])
        mouse_position = OmegaConf.create({"x": x, "y": y})
        img = crop(os.path.join(folder, filename), mouse_position)
        if img is not None:
            filenames.append(filename)
            images.append(img)

    return images, filenames


def filter_similar_images(images, similarity_threshold=0.9):
    unique_images = []
    for i in tqdm(range(len(images))):
        is_unique = True
        for j in range(i + 1, min(i + 30, len(images))):
            score, _ = compare_ssim(images[i][1], images[j][1], full=True)
            if score > similarity_threshold:
                is_unique = False
                break
        if is_unique:
            unique_images.append(images[i][0])
    return unique_images


def copy_images_to_target(source_folder, target_folder, image_filenames):
    os.makedirs(target_folder, exist_ok=True)
    for filename in image_filenames:
        shutil.copy(
            os.path.join(source_folder, filename), os.path.join(target_folder, filename)
        )


def save_to_dir(imgs, folder, filenames):
    for img, filename in tqdm(zip(imgs, filenames)):
        filepath = folder / filename
        cv2.imwrite(str(filepath), img)


screenshot_dir = "/Users/sachalevy/IMPLEMENT/datamakr/data/screenshots"
all_images = load_images_from_folder(screenshot_dir)
unique_images = filter_similar_images(all_images)
print("found", len(unique_images), "unique images")
unique_screenshot_dir = "/Users/sachalevy/IMPLEMENT/datamakr/data/unique_screenshots"
copy_images_to_target(screenshot_dir, unique_screenshot_dir, unique_images)

# screenshot_dir = "/Users/sachalevy/IMPLEMENT/datamakr/data/unique_screenshots"
# imgs, filenames = crop_then_filter(screenshot_dir)
# cropped_screenshot_dir = Path(
#    "/Users/sachalevy/IMPLEMENT/datamakr/data/cropped_screenshots"
# )
# cropped_screenshot_dir.mkdir(parents=True, exist_ok=True)
# save_to_dir(imgs, cropped_screenshot_dir, filenames)
