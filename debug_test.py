import unittest

import cv2
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

    # 构造一个 800x600 的输入图像和 1200x900 的输出图像
    input_img = np.ones((600, 800))
    output_img = np.zeros((900, 1200))

    # 构造一个 mapx 和 mapy 数组的样例
    mapx, mapy = np.meshgrid(np.arange(output_img.shape[1]),
                             np.arange(output_img.shape[0]))

    # 乘上一个系数，让映射结果不完全与原图相同
    mapx = mapx * 0.5
    mapy = mapy * 0.5

    # 进行映射
    result = cv2.remap(input_img, mapx, mapy, cv2.INTER_LINEAR)

    # 打印结果
    print(result.shape)
