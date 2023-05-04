"""
To compute the Intrinsic Matrix K of the given camera and photo.
"""
from typing import Tuple, Any

import exifread
import numpy as np
import sympy


def exif_info(_photo_path: str) -> Tuple[Any, Any, Any, Any]:
    """
    Get the Exif information of the given photo.

    >>>print(tags)
    {'Image ImageWidth': (0x0100) Long=720 @ 18,
    'Image ImageLength': (0x0101) Long=960 @ 30,
    'Image ExifOffset': (0x8769) Long=94 @ 42,
    'Image Orientation': (0x0112) Short=Horizontal (normal) @ 54,
    'Image DateTime': (0x0132) ASCII=2017:06:17 00:15:52 @ 74,
    'EXIF DateTimeOriginal': (0x9003) ASCII=2017:06:17 00:15:52 @ 136,
    'EXIF UserComment': (0x9286) ASCII="ad22b568395d1fe185dc951a0c58e528d9819bef","ext":"jpg"} @ 156,
    'EXIF LightSource': (0x9208) Long=Unknown @ 128}
    """
    f = open(_photo_path, 'rb')
    tags: dict = exifread.process_file(f)

    _image_width = tags['Image ImageWidth'].values[0] or tags['EXIF ExifImageWidth']
    # ! Image Length = Image Height
    _image_length = tags['Image ImageLength'].values[0] or tags['EXIF ExifImageLength']

    return _image_width, \
        _image_length, \
        float(eval(repr(tags['EXIF FocalLengthIn35mmFilm'].values[0]))), \
        float(eval(repr(tags['EXIF FocalLength'].values[0])))


def compute_cx_cy(_image_width: int, _image_length: int) -> Tuple[float, float]:
    """
    Compute the center of the image.
    """
    cx = _image_width / 2
    cy = _image_length / 2

    return cx, cy


def compute_dx_dy(_image_width: int, _image_length: int,
                  _focal_length35: float, _focal_length: float) -> Tuple[float, float, float, float]:
    """
    Compute dx and dy.
    t = 27/(279/50)
    l = 36^2+24^2
    x/y = 36/24
    x^2+y^2=l/t
    solve([x/y - 36/24, x^2+y^2 - l/t], [x, y])
    """
    cmos_width = sympy.Symbol("cmos_width")
    cmos_length = sympy.Symbol("cmos_length")
    # * 基于等效焦距与实际焦距计算转换系数
    conversion_factor = _focal_length35 / _focal_length
    # * 计算全画幅底片的传感器对角线长度的平方
    full_frame_diagonal_square = 36 ** 2 + 24 ** 2
    # * 求解非其次二元方程
    solution = sympy.solve([cmos_width / cmos_length - 36 / 24,
                            cmos_width ** 2 + cmos_length ** 2 - full_frame_diagonal_square / conversion_factor],
                           [cmos_width, cmos_length])
    for sol in solution:
        cmos_width, cmos_length = sol
        if cmos_width > 0 and cmos_length > 0:
            break

    dx = cmos_width / _image_width
    dy = cmos_length / _image_length

    return dx, dy, cmos_width, cmos_length


def compute_intrinsic_matrix(_focal_length: float,
                             _cx: float, _cy: float,
                             _dx: float, _dy: float,
                             _beta: float = 0) -> np.ndarray:
    """
    Output the Intrinsic Matrix.
    :param _dy:
    :param _dx:
    :param _cy:
    :param _cx:
    :param _focal_length:
    :param _beta: The inclination parameter of the camera.
    """

    fx = _focal_length / _dx
    fy = _focal_length / _dy

    intrinsic_matrix = np.array([[fx, _beta, _cx], [0, fy, _cy], [0, 0, 1]], dtype=float)

    return intrinsic_matrix


def compute_k(_photo_path: str, show: bool = False) -> np.ndarray:
    _image_width, _image_length, focal_length35, focal_length = exif_info(_photo_path)
    cx, cy = compute_cx_cy(_image_width, _image_length)
    dx, dy, cmos_width, cmos_length = compute_dx_dy(_image_width, _image_length, focal_length35, focal_length)
    _k: np.ndarray = compute_intrinsic_matrix(focal_length, cx, cy, dx, dy)
    np.set_printoptions(suppress=True)
    print(f'image_width = {_image_width} pixels\n'
          f'image_length = {_image_length} pixels\n'
          f'focal_length = {focal_length} mm\n'
          f'sensor_size = {cmos_width:.3f} * {cmos_length:.3f} mm\n'
          f'k = {_k}') if show is True else None
    return _k


if __name__ == '__main__':
    # photo_path = './lv_miui_5_25.35_3472_4624.jpg'
    # photo_path = './wu_p30_2_17.35_3456_4608.jpg'
    # photo_path = './wu_p30_2_17.35_4608_3456.jpg'
    # photo_path = './wu_p30_6_27.35_2736_3648.jpg'
    photo_path = 'pictures/wu_p30_6_27.35_3648_2736.jpg'

    k = compute_k(photo_path, show=True)
    # >>>print(os.getcwd())
    # ? 'D:\OneDrive - USTC\Data\Pycharm\BinaryClassify\Double Channel Fire Detection'
