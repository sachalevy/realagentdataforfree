import time

import AppKit
from pynput import keyboard, mouse


def on_press(key):
    try:
        print(f"Alphanumeric key pressed: {key.char} at {time.time()}")
    except AttributeError:
        print(f"Special key pressed: {key} at {time.time()}")


def on_release(key):
    print(f"Key released: {key} at {time.time()}")
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y}) with {button} at {time.time()}")
    else:
        print(f"Mouse released at ({x}, {y}) with {button} at {time.time()}")


def on_scroll(x, y, dx, dy):
    print(f"Mouse scrolled at ({x},{y})({dx},{dy}) at {time.time()}")


# Set up as a listener that calls the appropriate function
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_click=on_click, on_scroll=on_scroll)

# Start the listener threads
keyboard_listener.start()
mouse_listener.start()

keyboard_listener.join()  # remove if you want non-blocking
mouse_listener.join()  # remove if you want non-blocking
