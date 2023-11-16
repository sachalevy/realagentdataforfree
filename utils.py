from pathlib import Path
from typing import List, Any, Tuple
import subprocess
import base64
import re

import cv2
from tqdm import tqdm
import numpy as np
import AppKit
import omegaconf
from PIL import Image
import pyautogui
import pytesseract
from skimage.metrics import structural_similarity as compare_ssim


def get_mouse_position():
    mouse_position = pyautogui.position()
    return omegaconf.OmegaConf.create({"x": mouse_position.x, "y": mouse_position.y})


def take_screenshot(filepath: Path):
    """Takes a screenshot and saves it to the specified file path."""
    try:
        assert filepath.parent.exists()
        subprocess.run(["screencapture", str(filepath)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while taking the screenshot: {e}")


def clean_ocr_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("\n\n", "\n")
    text = text.encode("ascii", "ignore").decode()

    return text


def extract_text_from_screenshot(image_path):
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_string(255 - gray)
    return clean_ocr_text(data)


def extract_text_from_region(filepath: Path, contour: Any):
    img = cv2.imread(filepath)

    # turn contour into bounding-box to crop img
    x, y, w, h = cv2.boundingRect(contour)
    cropped_img = img[y : y + h, x : x + w]

    pil_img = Image.fromarray(cropped_img)
    return pytesseract.image_to_string(pil_img)


def get_active_app_name():
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    return workspace.frontmostApplication().localizedName()


def get_active_app_metadata():
    workspace = AppKit.NSWorkspace.sharedWorkspace()

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


def load_screenshots_from_folder(
    folder: Path, downsample: bool = False, size: Tuple[int, int] = (300, 300)
):
    """Loads screenshots from a folder."""
    sorted_img_filepaths = sorted(
        folder.glob("*.png"), key=lambda x: int(x.stem.split("_")[1])
    )
    imgs, filepaths = [], []
    for img_filepath in tqdm(sorted_img_filepaths):
        img = cv2.imread(img_filepath, cv2.IMREAD_GRAYSCALE)
        if img and downsample:
            img = cv2.resize(img, size)
        imgs.append(img)
        filepaths.append(img_filepath)
    return imgs, filepaths


def filter_unique_screenshots(
    images: List[Image.Image],
    filepaths: List[Path],
    sim_threshold: float = 0.9,
    window_size: int = 30,
):
    """Naïve pairwise similarity filter for images."""
    unique_images, unique_image_filenames = [], []
    for i in tqdm(range(len(images))):
        is_unique = True
        for j in range(i + 1, min(i + window_size, len(images))):
            score, _ = compare_ssim(images[i], images[j], full=True)
            if score > sim_threshold:
                is_unique = False
                break
        if is_unique:
            unique_image_filenames.append(filepaths[i])
            unique_images.append(images[i])
    return unique_images, unique_image_filenames


def crop_screenshot(
    filepath: Path, mouse_position: omegaconf.OmegaConf, verbose: bool = False
):
    """
    Segments windows using edge detection and crops out the active one
    using mouse position at the time of capture.
    """

    img = cv2.imread(filepath, 0)
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)

    scharr_combined = cv2.magnitude(scharrx, scharry)
    scharr_combined = cv2.convertScaleAbs(scharr_combined)

    thresh = cv2.adaptiveThreshold(
        scharr_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10
    )
    get_verbose_filepath = (
        lambda x: filepath.parent / f"{filepath.stem}_{x}{filepath.suffix}"
    )
    if verbose:
        thresholded_img_filepath = get_verbose_filepath("thresholded")
        cv2.imwrite(thresholded_img_filepath, thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # set minsize to be around 10% of total screen area
    height, width = img.shape
    min_size = int(height * width * 0.1)
    if verbose:
        black_image = np.zeros((height, width, 3), np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > min_size:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        bbox_img_filepath = get_verbose_filepath("bounding_boxes")
        cv2.imwrite(bbox_img_filepath, black_image)

    # filter out contours that are too small
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

    if verbose:
        # display img with selected bbox and cursor position
        black_image = np.zeros_like(img)
        if foreground_contour is not None:
            x, y, w, h = cv2.boundingRect(foreground_contour)
            cv2.rectangle(black_image, (x, y), (x + w, y + h), 255, 2)

        cross_color = (255, 255, 255)
        cross_size = 20
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
        foreground_bbox_img_filepath = get_verbose_filepath("foreground_bbox")
        cv2.imwrite(foreground_bbox_img_filepath, black_image)

    # crop image
    if foreground_contour is not None:
        x, y, w, h = cv2.boundingRect(foreground_contour)
    else:
        x, y, w, h = 0, 0, 0, 0

    return img[y : y + h, x : x + w]


def extract_sentences_from_keystrokes(keystrokes: List):
    sentences = []
    current_sentence = ""
    shift_pressed = False

    for line in keystrokes:
        parts = line.strip().split(",")
        key, action = parts[0], parts[1]

        if key == "Key.shift":
            if action == "press":
                shift_pressed = True
            elif action == "release":
                shift_pressed = False
        elif action == "press":
            if key.startswith("'") and key.endswith("'"):
                char = key[1:-1]
                if shift_pressed:
                    char = char.upper()
                current_sentence += char
            elif key == "Key.space":
                current_sentence += " "
            elif key == "Key.enter":
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            elif key == "Key.backspace":
                current_sentence = current_sentence[:-1]

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences


def retrieve_current_event(filepath: Path, start_ts: int, end_ts: int):
    events = []
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            ts = float(parts[-1]) if filepath.stem != "clicks" else float(parts[-2])
            if ts >= start_ts and ts <= end_ts:
                events.append(line.strip())

    if filepath.stem == "clicks":
        return events, events[0].strip().split(",")[-1]

    return events


def encode_image(filepath: Path):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
