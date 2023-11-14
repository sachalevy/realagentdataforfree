import time
import subprocess
from pathlib import Path

import AppKit
import pyautogui
from pynput import keyboard, mouse


data_dir = Path("/Users/sachalevy/IMPLEMENT/datamakr/data/")
data_dir.mkdir(parents=True, exist_ok=True)
screenshot_dir = data_dir / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)

press_fp = open(data_dir / "presses.txt", "a")


def on_press(key):
    press_fp.write(f"{key},press,{time.time()}\n")
    press_fp.flush()


def on_release(key):
    press_fp.write(f"{key},release,{time.time()}\n")
    press_fp.flush()


last = time.time()


def take_screenshot(file_path):
    """
    Takes a screenshot and saves it to the specified file path.

    Args:
    file_path (str): The path where the screenshot will be saved.
    """
    try:
        subprocess.run(["screencapture", file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while taking the screenshot: {e}")


click_fp = open(data_dir / "clicks.txt", "a")
DELTA = 3


def get_active_app_name():
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    return workspace.frontmostApplication().localizedName()


def get_mouse_position():
    return pyautogui.position()


def on_click(x, y, button, pressed):
    global last

    click_fp.write(
        f"{x},{y},{button},{pressed},{time.time()},{get_active_app_name()}\n"
    )
    click_fp.flush()

    curr = time.time()
    if not pressed and curr - last > DELTA:
        c = int(time.time())
        mouse_position = get_mouse_position()
        filepath = (
            screenshot_dir / f"screen_{c}_{mouse_position.x}_{mouse_position.y}.png"
        )
        take_screenshot(filepath)
        last = curr


scroll_fp = open(data_dir / "scrolls.txt", "a")


def on_scroll(x, y, dx, dy):
    scroll_fp.write(f"{x},{y},{dx},{dy},{time.time()}\n")


move_fp = open(data_dir / "moves.txt", "a")


def on_move(x, y):
    move_fp.write(f"{x},{y},{time.time()}\n")


keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)

keyboard_listener.start()
time.sleep(1)
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()
