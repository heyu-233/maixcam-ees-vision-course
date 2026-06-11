from maix import app, camera, display, image, time
import cv2


CAM_W = 512
CAM_H = 320
MIN_AREA = 1200
APPROX_EPSILON_RATIO = 0.02
MIN_ASPECT = 0.45
MAX_ASPECT = 2.8


def find_best_rectangle(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
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
            "center": (x + w // 2, y + h // 2),
            "approx": approx,
        }

        if best is None or area > best["area"]:
            best = candidate

    return best


def run_find_rects():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    while not app.need_exit():
        img = cam.read()
        img_cv = image.image2cv(img, False, False)
        best = find_best_rectangle(img_cv)

        show_img = img
        if best:
            x, y, w, h = best["rect"]
            cx, cy = best["center"]
            show_img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=2)
            show_img.draw_circle(cx, cy, 4, image.COLOR_RED, thickness=-1)
            show_img.draw_string(
                8,
                8,
                f"x={x}, y={y}, w={w}, h={h}, c=({cx},{cy})",
                image.COLOR_GREEN,
                scale=1,
            )
        else:
            show_img.draw_string(8, 8, "no rectangle", image.COLOR_RED, scale=1)

        fps = time.fps()
        if fps > 0:
            show_img.draw_string(8, 28, f"fps: {fps:.1f}", image.COLOR_WHITE, scale=1)

        disp.show(show_img)

