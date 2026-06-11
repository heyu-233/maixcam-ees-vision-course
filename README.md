# MaixCAM 电赛视觉教程代码

这是《MaixCAM 电赛视觉教程》的配套代码仓库。

每个 `lessons/` 子目录对应一期视频，目录内保留最小可运行文件：

- `main.py`
- `camera_display.py`
- `README.md`

课程主线是让 MaixCAM 作为电赛视觉前端，完成摄像头采集、图像处理、目标识别和后续串口输出。

## 课时目录

| 目录 | 内容 |
| --- | --- |
| `lessons/03_rectangle_preprocess` | 摄像头图像转 OpenCV，灰度、滤波、二值化、找轮廓 |
| `lessons/04_rectangle_detection` | 轮廓筛选、四边形近似、长宽比过滤、最佳矩形与中心点 |
| `lessons/05_corner_sorting` | 角点排序、真实四边形绘制、真实中心计算 |

## 使用方式

打开某一节课目录，将其中的 `main.py` 和 `camera_display.py` 放到 MaixCAM 项目中，用 MaixVision 运行 `main.py`。

建议每次只运行一个 lesson 目录中的代码，避免不同课时文件混在一起。

## 说明

代码面向教学和电赛备赛，优先保证结构清晰、方便修改和调参。

