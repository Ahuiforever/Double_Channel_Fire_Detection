#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 17:13
# @Author  : Ahuiforever
# @File    : debug.py
# @Software: PyCharm

import cv2
import numpy as np

# 构造一个 800x600 的输入图像和 1200x900 的输出图像
input_img = np.ones((600, 800))
output_img = np.zeros((900, 1200))

# 构造一个 mapx 和 mapy 数组的样例
mapx, mapy = np.meshgrid(np.arange(output_img.shape[1]),
                         np.arange(output_img.shape[0]))

# 将 mapx 和 mapy 转换为 float32 类型
mapx = mapx.astype(np.float32)
mapy = mapy.astype(np.float32)

# 乘上一个系数，让映射结果不完全与原图相同
mapx = mapx * 0.5
mapy = mapy * 0.5

# 进行映射
result = cv2.remap(input_img, mapx, mapy, cv2.INTER_LINEAR)

# 打印结果
print(result.shape)
