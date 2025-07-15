import cv2
import numpy as np
import os

# Load templates
TEMPLATE_DIR = "digits"
templates = {}
for filename in os.listdir(TEMPLATE_DIR):
    if filename.endswith(".png"):
        digit = filename.split(".")[0]
        path = os.path.join(TEMPLATE_DIR, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        templates[digit] = img

print(f"Loaded templates: {list(templates.keys())}")

def detect_digit(frame, region):
    """
    frame: full BGR or grayscale frame (as numpy array)
    region: (x1, y1, x2, y2) coordinates to crop digit area
    """
    x1, y1, x2, y2 = region
    roi = frame[y1:y2, x1:x2]
    
    # Convert to grayscale
    if img is None or img.size == 0:
        print("Error: empty image, cannot convert color")
        return None  # or handle error
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    best_digit = None
    best_score = -1

    for digit, template in templates.items():
        res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        # print(f"Digit {digit} match score: {max_val}")
        if max_val > best_score:
            best_score = max_val
            best_digit = digit
    if(best_score < 0.5 or best_score > 1):
        best_digit = 0
        best_score = 1
    return best_digit, best_score


def get_speed_reward():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        frame = sct.grab(monitor)
        img = np.array(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Define your digit regions (adjust as needed)
    digit1_region = (x1, y1, x2, y2)
    digit2_region = (x3, y3, x4, y4)  # if you have two digits for speed

    d1, conf1 = detect_digit(img, digit1_region)
    d2, conf2 = detect_digit(img, digit2_region)

    if d1 is None or d2 is None:
        return 0  # or some fallback

    speed = int(d1 + d2)
    return speed  # reward = current speed