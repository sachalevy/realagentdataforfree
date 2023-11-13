import AppKit
import Quartz.CoreGraphics as CG
from Cocoa import NSURL, NSDictionary
import LaunchServices
from pathlib import Path
import time


def get_active_app_name():
    # Get the shared workspace
    workspace = AppKit.NSWorkspace.sharedWorkspace()

    # Get the frontmost (active) application
    active_app = workspace.frontmostApplication()

    # Get the application name
    active_app_name = active_app.localizedName()

    return active_app_name


print("Current active application:", get_active_app_name())


def capture_screen(file_path):
    # Create a region of interest (in this case, the entire screen)
    region = CG.CGRectInfinite

    # Capture the screen
    image = CG.CGWindowListCreateImage(
        region,
        CG.kCGWindowListOptionOnScreenOnly,
        CG.kCGNullWindowID,
        CG.kCGWindowImageDefault,
    )

    # Create a destination to write the image
    url = NSURL.fileURLWithPath_(file_path)
    dest = CG.CGImageDestinationCreateWithURL(url, LaunchServices.kUTTypePNG, 1, None)

    # Add the image to the destination and finalize it
    CG.CGImageDestinationAddImage(dest, image, None)
    CG.CGImageDestinationFinalize(dest)


import subprocess
import os

import pyautogui


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

# mouse_position = take_screenshot(filepath)
from omegaconf import OmegaConf

mouse_position = OmegaConf.create({"x": 1436, "y": 772})

from PIL import Image
import pytesseract


def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"An error occurred while extracting text from the image: {e}")
        return None


# text = extract_text_from_image(filepath)
# print(text)

import io
import requests
import torch
import numpy

from transformers import DetrImageProcessor, DetrForObjectDetection


def try_segmenting_image(filepath):
    image = Image.open(filepath)
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-101", revision="no_timm"
    )

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )


import cv2

import numpy as np


def scharr(filepath):
    img = cv2.imread(filepath, 0)
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    # Combine the horizontal and vertical edges
    scharr_combined = cv2.magnitude(scharrx, scharry)
    scharr_combined = cv2.convertScaleAbs(scharr_combined)

    # find bounding boxes
    _, thresh = cv2.threshold(scharr_combined, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape
    black_image = np.zeros((height, width, 3), np.uint8)

    # Draw bounding boxes in white
    for contour in contours:
        if cv2.contourArea(contour) > 36000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return scharr_combined, black_image


print(mouse_position)


def scharr_sorted(filepath):
    img = cv2.imread(filepath, 0)
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    # Combine the horizontal and vertical edges
    scharr_combined = cv2.magnitude(scharrx, scharry)
    scharr_combined = cv2.convertScaleAbs(scharr_combined)

    # Apply dilation to close gaps
    # _, thresh = cv2.threshold(scharr_combined, 50, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    thresh = cv2.adaptiveThreshold(
        scharr_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )
    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    cv2.imwrite("data/thresd.png", thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape
    black_image = np.zeros((height, width, 3), np.uint8)

    # Draw bounding boxes in white
    for contour in contours:
        if cv2.contourArea(contour) > 72000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.imwrite("data/all_bounding_boxes.png", black_image)

    # alternative
    contours = [c for c in contours if cv2.contourArea(c) > 72000]
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

    return scharr_combined, black_image


def sobel(filepath):
    # Read the image
    img = cv2.imread(filepath, 0)  # 0 means load in grayscale
    # 1. equalizer + canny
    # equalized_img = cv2.equalizeHist(img)
    # edges = cv2.Canny(equalized_img, 100, 200)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # edges = clahe.apply(img)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Combine both directions
    edges = cv2.magnitude(sobelx, sobely)

    # grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    # grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # Compute magnitude
    # edges = np.sqrt(grad_x**2 + grad_y**2)

    # kernel = np.ones((5, 5), np.uint8)

    # Apply dilation and erosion
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # erosion = cv2.erode(img, kernel, iterations=1)

    # Combine dilation and erosion
    # edges = cv2.bitwise_or(dilation, erosion)

    return edges


edges, img = scharr_sorted(str(filepath))
# Save or display the edge detection result
cv2.imwrite(os.path.expanduser("data/edges_schaar.png"), edges)
cv2.imwrite("data/bounding_boxes.png", img)
# edges = sobel(str(filepath))
# cv2.imwrite(os.path.expanduser("data/edges_sobel.png"), edges)

print("Edge detection result saved.")
