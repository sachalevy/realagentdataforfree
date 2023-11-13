import time
import subprocess
import threading

from pathlib import Path
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


def on_click(x, y, button, pressed):
    global last
    click_fp.write(f"{x},{y},{button},{pressed},{time.time()}\n")
    click_fp.flush()

    curr = time.time()
    if not pressed and curr - last > DELTA:
        c = int(time.time())  # Using the current timestamp as a unique identifier
        filepath = screenshot_dir / f"screen_{c}.png"
        take_screenshot(filepath)
        last = curr


scroll_fp = open(data_dir / "scrolls.txt", "a")


def on_scroll(x, y, dx, dy):
    scroll_fp.write(f"{x},{y},{dx},{dy},{time.time()}\n")


move_fp = open(data_dir / "moves.txt", "a")


def on_move(x, y):
    move_fp.write(f"{x},{y},{time.time()}\n")


# Set up as a listener that calls the appropriate function
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll, on_move=on_move)

# Start the listener threads
keyboard_listener.start()
time.sleep(1)  # wait for the keyboard listener to start
mouse_listener.start()

keyboard_listener.join()  # remove if you want non-blocking
mouse_listener.join()  # remove if you want non-blocking
