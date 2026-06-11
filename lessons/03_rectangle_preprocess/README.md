# 03 矩形识别前处理

本节目标：

- 打开 MaixCAM 摄像头。
- 将 Maix 图像转成 OpenCV 图像。
- 完成灰度化、GaussianBlur 滤波和自适应二值化。
- 使用 `cv2.findContours()` 找出画面中的轮廓。

运行方法：

1. 用 MaixVision 打开本目录。
2. 运行 `main.py`。
3. 屏幕会显示二值化后的图像。
4. 终端会打印轮廓数量和 FPS。

这一节对应“矩形识别逐步开发讲解”的前 6 步。

