import AppKit
from pathlib import Path
import time

import cv2

from PIL import Image
import pytesseract
import subprocess
import os

import pyautogui
import numpy as np


def get_active_app_name():
    # Get the shared workspace
    workspace = AppKit.NSWorkspace.sharedWorkspace()

    # Get the frontmost (active) application
    active_app = workspace.frontmostApplication()
    metadata = {
        "name": active_app.localizedName(),
        "bundle_identifier": active_app.bundleIdentifier(),
        "process_identifier": active_app.processIdentifier(),
        "executable_url": str(active_app.executableURL()),
        "launch_date": str(active_app.launchDate()),
        "is_hidden": active_app.isHidden(),
        "is_terminated": active_app.isTerminated(),
    }

    return metadata


print("Current active application:", get_active_app_name())


def get_mouse_position():
    return pyautogui.position()


def take_screenshot(file_path):
    """
    Takes a screenshot and saves it to the specified file path.

    Args:
    file_path (str): The path where the screenshot will be saved.
    """
    time.sleep(3)
    mouse_position = get_mouse_position()
    try:
        subprocess.run(["screencapture", file_path], check=True)
        print(f"Screenshot saved to {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while taking the screenshot: {e}")

    return mouse_position


screenshot_dir = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/")
screenshot_dir.mkdir(parents=True, exist_ok=True)
c = 0
filepath = screenshot_dir / f"screen_{c}.png"

mouse_position = None
mouse_position = take_screenshot(filepath)
from omegaconf import OmegaConf

if not mouse_position:
    mouse_position = OmegaConf.create({"x": 493, "y": 576})
else:
    mouse_position = OmegaConf.create({"x": mouse_position.x, "y": mouse_position.y})
print(mouse_position)
mouse_position.x = mouse_position.x * 2
mouse_position.y = mouse_position.y * 2


def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from the image: {e}")
        return None


def scharr_sorted(filepath):
    img = cv2.imread(filepath, 0)
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    # Combine the horizontal and vertical edges
    scharr_combined = cv2.magnitude(scharrx, scharry)
    scharr_combined = cv2.convertScaleAbs(scharr_combined)

    thresh = cv2.adaptiveThreshold(
        scharr_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )
    cv2.imwrite("data/thresd.png", thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape
    min_size = 500000
    print(f"{min_size/(height*width)*100:.3f}% of image size")
    black_image = np.zeros((height, width, 3), np.uint8)

    # Draw bounding boxes in white
    for contour in contours:
        if cv2.contourArea(contour) > min_size:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imwrite("data/all_bounding_boxes.png", black_image)

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

    # Create a black image of the same size for drawing the contour
    black_image = np.zeros_like(img)

    # Draw the foreground contour
    if foreground_contour is not None:
        x, y, w, h = cv2.boundingRect(foreground_contour)
        cv2.rectangle(black_image, (x, y), (x + w, y + h), 255, 2)

    cross_color = (255, 255, 255)  # White color for the cross
    cross_size = 20  # Size of the cross arms
    cv2.line(
        black_image,
        (mouse_position.x - cross_size, mouse_position.y),
        (mouse_position.x + cross_size, mouse_position.y),
        cross_color,
        2,
    )
    cv2.line(
        black_image,
        (mouse_position.x, mouse_position.y - cross_size),
        (mouse_position.x, mouse_position.y + cross_size),
        cross_color,
        2,
    )

    return scharr_combined, black_image, foreground_contour


def extract_text_from_region(image_path, contour):
    # Load the image
    img = cv2.imread(image_path)

    # Crop the image using the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    cropped_img = img[y : y + h, x : x + w]

    # Convert the cropped image to a PIL Image for Tesseract
    pil_img = Image.fromarray(cropped_img)

    # Extract text using Tesseract
    text = pytesseract.image_to_string(pil_img)
    return text


edges, img, contour = scharr_sorted(str(filepath))
# Save or display the edge detection result
cv2.imwrite(os.path.expanduser("data/edges_schaar.png"), edges)
cv2.imwrite("data/bounding_boxes.png", img)

text = extract_text_from_region(str(filepath), contour)
# print(text)
import re


def remove_non_printable_chars(text):
    return re.sub(r"[^\x20-\x7E]", "", text)


def correct_common_ocr_errors(text):
    corrections = {
        "0": "O",  # Zero to letter O
        "1": "I",  # One to letter I
        "|": "I"
        # Add more based on common errors observed in your OCR output
    }
    return "".join(corrections.get(c, c) for c in text)


def remove_extra_whitespaces(text):
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def format_paragraphs(text):
    # Replace breaks with a single newline, etc.
    return text.replace("\n\n", "\n")


def clean_ocr_text(text):
    text = remove_non_printable_chars(text)
    text = correct_common_ocr_errors(text)
    text = remove_extra_whitespaces(text)
    text = format_paragraphs(text)
    return text


# Example usage
print(text)
cleaned_text = clean_ocr_text(text)
print(cleaned_text)


print("Edge detection result saved.")
