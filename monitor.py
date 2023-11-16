import time
from pathlib import Path

from pynput import keyboard, mouse

import utils

DATA_DIR = Path("data/")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = DATA_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# record movements & clicks to text files
PRESS_FP = DATA_DIR / "presses.txt"
PRESS_FD = open(PRESS_FP, "a")
MOVE_FP = DATA_DIR / "moves.txt"
MOVE_FD = open(MOVE_FP, "a")
CLICK_FP = DATA_DIR / "clicks.txt"
CLICK_FD = open(CLICK_FP, "a")
SCROLL_FP = DATA_DIR / "scrolls.txt"
SCROLL_FD = open(SCROLL_FP, "a")

DELTA = 30
LAST = time.time()


def on_press(key):
    PRESS_FD.write(f"{key},press,{time.time()}\n")
    PRESS_FD.flush()


def on_release(key):
    PRESS_FD.write(f"{key},release,{time.time()}\n")
    PRESS_FD.flush()


def on_click(x, y, button, pressed):
    global LAST

    CLICK_FD.write(
        f"{x},{y},{button},{pressed},{time.time()},{utils.get_active_app_name()}\n"
    )
    CLICK_FD.flush()

    curr = time.time()
    if not pressed and curr - last > DELTA:
        mouse_position = utils.get_mouse_position()
        filepath = (
            SCREENSHOT_DIR
            / f"screen_{int(time.time())}_{mouse_position.x}_{mouse_position.y}.png"
        )
        utils.take_screenshot(filepath)
        last = curr


def on_scroll(x, y, dx, dy):
    SCROLL_FD.write(f"{x},{y},{dx},{dy},{time.time()}\n")


def on_move(x, y):
    MOVE_FD.write(f"{x},{y},{time.time()}\n")


keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)

keyboard_listener.start()
time.sleep(1)
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()
