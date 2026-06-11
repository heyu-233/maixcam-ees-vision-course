from maix import app, camera, display, image, time
import cv2
import numpy as np


CAM_W = 512
CAM_H = 320

MIN_AREA = 1200
MIN_ASPECT = 0.45
MAX_ASPECT = 2.8
APPROX_EPSILON_RATIO = 0.02
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 8


def find_best_rectangle(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue

        aspect = w / h
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        candidate = {
            "area": area,
            "rect": (x, y, w, h),
            "approx": approx,
        }

        if best is None or area > best["area"]:
            best = candidate

    return best


def order_points(points):
    pts = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(4)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    return ordered


def draw_ordered_points(img, ordered):
    labels = ["TL", "TR", "BR", "BL"]
    colors = [
        image.COLOR_RED,
        image.COLOR_GREEN,
        image.COLOR_BLUE,
        image.COLOR_YELLOW,
    ]

    for i, point in enumerate(ordered):
        x, y = int(point[0]), int(point[1])
        img.draw_circle(x, y, 5, colors[i], thickness=-1)
        img.draw_string(x + 4, y + 4, labels[i], colors[i], scale=1)


def draw_polygon(img, ordered):
    pts = ordered.astype("int32")
    for i in range(4):
        x1, y1 = int(pts[i][0]), int(pts[i][1])
        x2, y2 = int(pts[(i + 1) % 4][0]), int(pts[(i + 1) % 4][1])
        img.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, thickness=2)


def get_polygon_center(ordered):
    tl, tr, br, bl = ordered
    x1, y1 = tl
    x2, y2 = br
    x3, y3 = tr
    x4, y4 = bl

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        center = ordered.mean(axis=0)
        return int(center[0]), int(center[1])

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return int(px), int(py)


def run_find_rects():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    while not app.need_exit():
        img = cam.read()
        img_cv = image.image2cv(img, False, False)

        best = find_best_rectangle(img_cv)
        show_img = img

        if best:
            ordered = order_points(best["approx"])
            cx, cy = get_polygon_center(ordered)
            draw_polygon(show_img, ordered)
            show_img.draw_circle(cx, cy, 4, image.COLOR_RED, thickness=-1)
            draw_ordered_points(show_img, ordered)
            show_img.draw_string(8, 8, f"center=({cx},{cy})", image.COLOR_GREEN, scale=1)
        else:
            show_img.draw_string(8, 8, "no rectangle", image.COLOR_RED, scale=1)

        fps = time.fps()
        if fps > 0:
            show_img.draw_string(8, 28, f"fps: {fps:.1f}", image.COLOR_WHITE, scale=1)

        disp.show(show_img)

