from maix import app, camera, display, image, time
import cv2


CAM_W = 512
CAM_H = 320


def run():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    while not app.need_exit():
        img = cam.read()
        img_cv = image.image2cv(img, False, False)

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

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        print("contours:", len(contours))

        binary_img = image.cv2image(binary, False, False)
        disp.show(binary_img)

        fps = time.fps()
        if fps > 0:
            print(f"time: {1000 / fps:.02f}ms, fps: {fps:.02f}")

