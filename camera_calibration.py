#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 17:15
# @Author  : Ahuiforever
# @File    : camera_calibration.py
# @Software: PyCharm
"""
    Try to realize the function of camera calibration with some functions or method built in opencv .
    Input the chessboard images from different perspectives, afterward, output the Intrinsic Matrix and the Extrinsic
Matrix.
    At last, calibrate the given image in real-time according to the above matrix and functions of opencv.
    Realize the whole program with PYTHON CLASS instead of defining functions.
"""

import glob
import os

import cv2
import numpy as np


class Calibrator:
    """
    The CLASS Calibrator was created for camera calibration computation, which is based on the input calibration images
    with chessboard. Output the Intrinsic Matrix and the Extrinsic Matrix with an order of _mtx, _dist, _rvecs, _tvecs.
    For details, see # compute.
    """

    def __init__(self, _chessboard_width: int, _chessboard_height: int, _chessboard_length: float):
        self._cw = _chessboard_width  # * 10 - 1 棋盘格上宽度方向的角点个数
        self._ch = _chessboard_height  # * 7 - 1 棋盘格上高度方向的角点个数
        self._cl = _chessboard_length  # * 棋盘上打印的棋盘格边长，单位毫米

        self._chessboard_image_folder = None
        self._suffix = None

        self._files = []
        self._mtx, self._dist, self._rvecs, self._tvecs = [0.] * 4

        # ? 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # ? 世界坐标系中的棋盘格点 [[u, v, 0], ···]
        self._chessboard_points = np.zeros((self._cw * self._ch, 3), np.float32)
        # ! 下面这行代码是为了生成索引矩阵的。并将生成的（-1， 2）矩阵赋值给_object_point的前两列
        self._chessboard_points[:, :2] = np.mgrid[:self._cw, :self._ch].T.reshape(-1, 2)
        # * 乘以单个棋盘格的边长，因此得到棋盘格点的坐标
        self._chessboard_points = np.multiply(self._chessboard_points, self._cl)

        # * 世界坐标系中的三维点，图像平面的二维点
        self._object_points, self._image_points = ([], [])

    def go_through_images(self) -> None:
        _search_path = os.path.join(self._chessboard_image_folder, f'*.{self._suffix}')
        # * _search_path = 'D:/Desktop/Any/*.jpg'
        self._files = glob.glob(_search_path)
        # * _files[0] = 'D:/Desktop/Any/0001.jpg'

    def compute(self, _chessboard_image_folder: str, _suffix: str = 'jpg', _show_img: bool = False) -> None:
        self._chessboard_image_folder = _chessboard_image_folder
        self._suffix = _suffix
        self.go_through_images()
        _gray = None

        for _file in self._files:
            _img = cv2.imread(_file)
            _gray = cv2.cvtColor(_img, cv2.COLOR_BGRA2GRAY)
            # * 找到棋盘的角点，精度为像素级
            _ret, _corners = cv2.findChessboardCorners(_gray, (self._cw, self._ch), None)
            if _ret is True:
                # * cv2.cornerSubPix()可实现精度为亚像素级的角点检测
                # ? (11, 11)是搜索窗口的尺寸，(-1, -1)表示没有死区
                cv2.cornerSubPix(_gray, _corners, (11, 11), (-1, -1), self._criteria)
                self._object_points.append(self._chessboard_points)
                self._image_points.append(_corners)
                # * 将角点绘制在图片上
                cv2.drawChessboardCorners(_img, (self._cw, self._ch), _corners, _ret)
                cv2.imwrite(
                    # * './results/Cap1_1_corners.jpg'
                    "./results/" + f"{_file.split(os.sep)[-1].replace(f'.{self._suffix}', f'_corners.{self._suffix}')}",
                    _img)
                if _show_img is True:
                    cv2.namedWindow('corners', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('corners', 640, 640)
                    cv2.imshow('corners', _img)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
        # * Compute.
        if _gray is not None:
            # ! a[::-1]相当于 a[-1:-len(a)-1:-1], 从最后一个元素到第一个元素复制一遍，最后一个-1表示反向读取，而非索引值
            # ! 因为这里的size是（宽， 高），而np.ndarray.shape是（高， 宽）
            _ret, self._mtx, self._dist, self._rvecs, self._tvecs = \
                cv2.calibrateCamera(self._object_points, self._image_points, _gray.shape[::-1], None, None)
            '''
            _ret 是重投影误差；
            _mtx 是内参矩阵；>>>{ndarray:(3, 3)}
            _dist 是畸变系数；distortion coefficients = (k1, k2, p1, p2, k3)；>>>{ndarray:(1, 5)}
            _rvecs 是旋转外参矩阵；rotation vectors；>>>{tuple:39}
            _tvecs 是平移外参矩阵；transformation vectors；>>>{tuple:39}
            '''

    def __call__(self, **kwargs) -> None:
        self.compute(kwargs['chessboard_image_folder'], kwargs['suffix'], kwargs['show_img'])

    @property
    def files(self):
        return self._files

    @property
    def mtx(self):
        return self._mtx

    @property
    def rvecs(self):
        return self._rvecs

    @property
    def tvecs(self):
        return self._tvecs


# ! ======================================================================================================OLD FUNCTION S
#     def normalization(self, _image_a: np.ndarray, _image_b: np.ndarray):
#         _k1 = self._mtx
#         _k1_inv = np.linalg.inv(_k1)
#         _k2 = self._new_mtx
#         _t1 = np.eye(4)
#         _num_images = self._files[]
#         _t1[:3, :3] = self._rvecs
#         _t1[:3, 3] = self._tvecs
#         _t1_inv = np.linalg.inv(_t1)
#         _t2 = np.copy(_t1)
#         _t2[:3, :3] = self._new_rvecs
#         _t2[:3, 3] = self._new_tvecs
#         _height, _width = _image_a.shape[:2]
#         _puv1 = np.ones((3, _width * _height), np.float32)
#         _puv1[:2, :] = np.mgrid[:_width, :_height].T.reshape(-1, 2).T
#         _puv2 = np.dot(_k2,
#                        np.dot(_t2,
#                               np.dot(_t1_inv,
#                                      np.dot(_k1_inv, _puv1))))
#         # _dst = np.zeros(np.max(_puv2, axis=0)[::-1][1:] + 1)
#         _new_image = cv2.remap(src=_image_a, map1=_puv2[:, :2], map2=None, interpolation=cv2.INTER_LINEAR)
#         cv2.imwrite('result.jpg', _new_image)
# ! ======================================================================================================OLD FUNCTION E

# / todo: 1. modify self.compute() to calculate two groups of parameters, got k and k_new and so on.
# / todo: 2. modify self.normalization() to realize image calibration according to parameter pairs like (k -> k_new).
# / todo: 3. BEFORE the above, read the document of opencv-camera-calibration carefully to see whether there are
# / todo:    existed functions could be used directly.
# ! Now I've found that function :cv2.remap(). But NOTICE: Before applying it, we are supposed to compute the map1 and
# ! map2 in the right way as well as in the expected type.


class Normalizer:
    """
    The CLASS Normalizer was created for normalizing the intrinsic & extrinsic of images. It means to synthesize the
    image C with the array of image A and the in-&-ex-trinsic of image B.
    """

    def __init__(self, *args, **kwargs):
        self._k1 = kwargs['original_k']
        # self._k1 = cv2.convertPointsToHomogeneous(self._k1).squeeze()
        self._k1_inv = np.linalg.inv(self._k1)
        self._k2 = kwargs['given_k']
        # self._k2 = cv2.convertPointsToHomogeneous(self._k2).squeeze()
        # ! cv::calibrateCamera output vector of rotation vectors (3, 1) estimated for each pattern view.
        # * Here _rvecs doesn't represent the Rotation Vectors, instead it is Rotation Matrix with a shape of (3, 3).
        self._rvecs1 = cv2.Rodrigues(kwargs['original_rvecs'], )[0]
        self._rvecs2 = cv2.Rodrigues(kwargs['given_rvecs'])[0]
        self._tvecs1 = kwargs['original_tvecs']
        self._tvecs2 = kwargs['given_tvecs']
        # self._t1 = np.vstack((np.hstack((self._rvecs1, self._tvecs1)), [0, 0, 0, 1]))
        self._t1 = np.hstack((self._rvecs1[:, :2], self._tvecs1))
        self._t1_inv = np.linalg.inv(self._t1)
        # self._t2 = np.vstack((np.hstack((self._rvecs2, self._tvecs2)), [0, 0, 0, 1]))
        self._t2 = np.hstack((self._rvecs2[:, :2], self._tvecs2))
        self._image_a_name = args[0]
        self._image_b_name = args[1]
        self._image_a = cv2.imread(self._image_a_name)
        self._image_b = cv2.imread(self._image_b_name)
        self._height, self._width = self._image_a.shape[:2]
        self._puv1 = np.ones((3, self._width * self._height), np.float32)
        self._puv1[:2, :] = np.mgrid[:self._width, :self._height].T.reshape(-1, 2).T
        self._puv2 = None

    def print_info(self) -> None:
        print(
            f'The Intrinsic Matrix of the original image A, \nK_1 = {self._k1}\n',
            f'The Extrinsic Matrix of the original image A, \nT_1 = {self._t1}\n',
            f'The Intrinsic Matrix of the original image B, \nK_2 = {self._k2}\n',
            f'The Extrinsic Matrix of the original image B, \nT_2 = {self._t2}\n',
        )

    def get_map(self) -> None:
        self._puv2 = np.dot(self._k2, np.dot(self._t2, np.dot(self._t1_inv, np.dot(self._k1_inv, self._puv1))))
        r'''
        - Equation as follows:
            $$
            \mathbf{P}_{uv2} = 
            \frac{Z_1}{Z_2} \mathbf{K}_{2} \mathbf{T}_2 \mathbf{T}_1^{-1} \mathbf{K}_1^{-1} \mathbf{P}_{uv1}
            $$
        '''

    def remap(self, _save_img: bool = True, _show_img: bool = False) -> None:
        _map1 = self._puv2[0, :].reshape(-1, self._width).astype(np.float32)
        _map2 = self._puv2[1, :].reshape(-1, self._width).astype(np.float32)
        # / _map1, _map2 = cv2.convertMaps(_map1, _map2, cv2.CV_32FC1)
        # / todo: 1. Use opencv build-in function to transform the _map1 and _map2 into CV_32FC2 or CV_16SC2type
        # / todo: 1. ((map1.type() == CV_32FC2 || map1.type() == CV_16SC2) && map2.empty()) ||
        # / todo: 1. (map1.type() == CV_32FC1 && map2.type() == CV_32FC1)
        self._image_b = cv2.remap(src=self._image_a, map1=_map1, map2=_map2, interpolation=cv2.INTER_LINEAR)
        if _save_img is True:
            cv2.imwrite(
                # *
                "./results/" + f"{self._image_a_name.split(os.sep)[-1].replace('.jpg', '_remap.jpg')}",  # % 'JPG'
                self._image_b)
        if _show_img is True:
            cv2.namedWindow('remap', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('remap', 640, 640)
            cv2.imshow('remap', np.concatenate([self._image_a, self._image_b], axis=1))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    # / todo: 1. Figure out how to pass the necessary info from CLASS Calibrator to CLASS Normalizer in a proper way.
    # / todo: 2. Figure out why self.mtx has a shape like (3, 3) ?

    def __call__(self, **kwargs):
        self.get_map()
        self.remap(_save_img=kwargs['save_img'],
                   _show_img=kwargs['show_img'])


class RealTime:
    def __init__(self):
        pass

    def get_files(self):
        # * 读取另外的本地文件夹，或从Calibrator类中获取
        pass
        # return original_img, given_img

    def get_parameters(self):
        # * 读取本地文件，或从Calibrator类中获取
        pass


if __name__ == '__main__':
    # c_test = Calibrator(7, 7, 25)
    # c_test(chessboard_image_folder='./hta0-horizontal-robot-arm/calibration_images',
    #        suffix='jpg',
    #        show_img=False)
    # * c1 as the original image, as the RGB channel, and of course c2 should be as the INFRARED channel.
    # * What this will do is to Remap the c1 according to the intrinsic & extrinsic of c2.
    c1 = Calibrator(7, 7, 25)
    c1(chessboard_image_folder='./hta0-horizontal-robot-arm/calibration_images',
       suffix='jpg',
       show_img=True)
    c2 = Calibrator(11, 8, 20)  # % VALUES TO BE CHANGED
    c2(chessboard_image_folder='./Calibration-ZhangZhengyou-Method/pic/RGB_camera_calib_img',  # % VALUES TO BE CHANGED
       suffix='png',
       show_img=True)
    for original_img, given_img, index in zip(c1.files, c2.files, range(len(c1.files))):
        n = Normalizer(original_img, given_img,
                       original_k=c1.mtx, given_k=c2.mtx,
                       original_rvecs=c1.rvecs[index], given_rvecs=c2.rvecs[index],
                       original_tvecs=c1.tvecs[index], given_tvecs=c2.tvecs[index])
        n(save_img=True, show_img=True)
