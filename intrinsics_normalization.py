"""
Normalize the intrinsics of two pictures.
Produce pic B according to the pic A and the given intrinsic matrix.
"""
import glob
import os
import shutil
import time

import cv2
from tqdm import tqdm

# * numpy has been imported as np in the following.
from intrinsics import *


def log_writer(_text: any, _new: bool = False) -> None:
    _text = _text if type(_text) is str else str(_text)
    if _new is True:
        with open('log.txt', 'w+') as _f:
            _f.write('=' * 20)
            _f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            _f.write('=' * 20)
            _f.write("\n" + _text + "\n")
    else:
        with open('log.txt', 'a') as _f:
            _f.write('=' * 20)
            _f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            _f.write('=' * 20)
            _f.write("\n" + _text + "\n")


def visualize(_img_array: np.ndarray, _frame_num: int, _video: bool = False) -> None:
    """
    Produce the image when function is called and encode them all into a video file if _video is True.
    :param _img_array:
    :param _frame_num:
    :param _video:
    :return:
    """
    _folder = './visualization'
    if os.path.exists(_folder) is False:
        os.makedirs(_folder)
    elif os.path.exists(_folder) is True and _frame_num == 1:
        shutil.rmtree(_folder)
        os.makedirs(_folder)
    _frame_name = f'./visualization/{str(_frame_num).zfill(4)}.jpg'
    cv2.imwrite(_frame_name, _img_array) if _frame_num != -1 else None
    if _video is True:
        _fps = 30.
        _size = (_img_array.shape[1], _img_array.shape[0])
        _fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # _film = cv2.VideoWriter('visualization01.avi', _fourcc, _fps, _size)
        _film = cv2.VideoWriter("appsrc ! filesink location=./visualization01.avi", _fourcc, _fps, _size)
        _frame_files = glob.glob('./visualization/*.jpg')
        for _frame_file in tqdm(_frame_files, desc=f'Encoding frame2video'):
            _img = cv2.imread(_frame_file)
            _film.write(_img)
        _film.release()
        cv2.destroyAllWindows()


def vectorized_offset(_width: int, _height: int, _k1: np.ndarray, _k2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # _idx_b = np.linspace(0, _height, _height, endpoint=False, dtype=int).reshape(-1, 1)
    # _idx_new = np.array([])
    # * 生成img_array的索引矩阵。以列向量的形式横向堆叠
    _i, _j = (np.arange(_width), np.arange(_height))
    _ij_1, _ij_2 = np.meshgrid(_i, _j)
    _ij_1, _ij_2 = (_ij_1.reshape(-1, 1), _ij_2.reshape(-1, 1))
    _ones = np.ones_like(_ij_1)
    _idx_new = np.concatenate((_ij_1, _ij_2, _ones), axis=1).T
    # * The following for-loop cost much more time than the above. So abandoned. =======================================
    # for _i in trange(0, _width, desc='Producing index array...'):
    #     _idx = _idx_new
    #     _idx_c = np.ones((_height, 1), dtype=int)
    #     _idx_a = np.multiply(_idx_c, _i)
    #     _idx_new = np.concatenate((_idx_a, _idx_b, _idx_c), axis=1)
    #     _idx_new = np.concatenate((_idx, _idx_new), axis=0) if _idx.size > 0 else _idx_new
    # * ================================================================================================================
    tqdm.write('Calculating the transformation for pixels coordinates...')
    _idx_offset: np.ndarray = np.dot(np.dot(_k2, np.linalg.inv(_k1)), _idx_new)
    assert _idx_offset.shape == (3, _width * _height)
    assert _idx_new.shape == _idx_offset.shape
    return _idx_new, _idx_offset


def vectorized_assignment(_array_a: np.ndarray, _array_b: np.ndarray,
                          _idx_a: np.ndarray, _idx_b: np.ndarray) -> np.ndarray:
    """
    Assigns the values to _array_b according to idx_b and corresponding values of _array_a.
    """
    _idx_b_copy = _idx_b.copy()
    _idx_b_copy[_idx_b > 0] = 0
    _idx_b_zero_sum = np.sum(_idx_b_copy, axis=0)
    _idx_b = _idx_b[:, _idx_b_zero_sum == 0]
    _idx_a = _idx_a[:, _idx_b_zero_sum == 0]
    _array_b[_idx_b[1, :], _idx_b[0, :]] = _array_a[_idx_a[1, :], _idx_a[0, :]]
    return _array_b


def intrinsic2image(_given_intrinsic_matrix: np.ndarray, _original_img_path: str):
    _k1 = compute_k(_original_img_path)
    _k2 = _given_intrinsic_matrix
    # ! cv2.imread() open the img with 'BGR' channel order.
    _img_array_1 = cv2.imread(_original_img_path)
    tqdm.write(f'{_original_img_path} is not valid.') if _img_array_1.size <= 0 else None
    _height, _width = _img_array_1.shape[0:2]
    _idx_new, _idx_offset = vectorized_offset(_width, _height, _k1, _k2)
    _ones = _idx_offset[2, :]
    _i_offset_max, _j_offset_max = \
        np.array(
            np.round(
                np.max(
                    np.divide(_idx_offset, _ones), axis=1)[0:2], decimals=0), dtype=int)
    log_writer(str(_i_offset_max) + '\n' + str(_j_offset_max))
    # ? _i_offset_max 与 _j_offset_max 都是从坐标中直接选取的最大值，并非就是shape，因此需要 +1
    # _idx_i_offset_max, _idx_j_offset_max = np.argmax(_idx_offset, axis=1)[0:2]
    # _one_i_offset_max = _idx_offset[2, _idx_i_offset_max]
    # _one_j_offset_max = _idx_offset[2, _idx_j_offset_max]
    # _i_offset_max = np.array(np.round(_i_offset_max / _one_i_offset_max, decimals=0), dtype=int)
    # _j_offset_max = np.array(np.round(_j_offset_max / _one_j_offset_max, decimals=0), dtype=int)
    _img_array_2 = np.zeros((_j_offset_max + 1, _i_offset_max + 1, 3), dtype=int)
    tqdm.write('Assigning values to the image pixels...')
    _idx_offset = np.array(np.round(np.divide(_idx_offset, _ones), decimals=0), dtype=int)
    _img_array_2 = vectorized_assignment(_img_array_1, _img_array_2, _idx_new, _idx_offset)
    # * The following for-loop also abandoned. =========================================================================
    # _pool = multiprocessing.Pool(multiprocessing.cpu_count())  # ! 调用全部cpu执行
    # _ones = np.array([])
    # for _ij, _ij_offset in tzip(np.split(_idx_new, _idx_new.shape[1], axis=1),
    #                             np.split(_idx_offset, _idx_offset.shape[1], axis=1)):
    #     '''
    #     _i, _j 始终与像素坐标系统一，即_i沿x轴，为横向; _j沿y轴，为纵向。
    #     但是在numpy中，索引为先纵着数后横着数，即(_j, _i)。
    #     '''
    #     _i, _j = _ij.squeeze()[0:2]
    #     # * 计算得到的并非刚好是整数，四舍五入得到float，再转化为int
    #     # _i_offset, _j_offset = np.array(np.round(_ij_offset.squeeze()[0:2], decimals=0), dtype=int)
    #     _i_offset, _j_offset, _one = _ij_offset.squeeze()
    #     np.append(_ones, _one)
    #     # ! _one 位置的值应当是1，但实际得到肯定不是1，因此需要归一化，如此得到的才是正确的坐标值
    #     _i_offset, _j_offset, _ = np.array(np.round([_i_offset / _one, _j_offset / _one, 1], decimals=0), dtype=int)
    #     # * 依据坐标变换对矩阵赋值
    #     _img_array_2[_j_offset, _i_offset, :] = _img_array_1[_j, _i, :] if _j_offset >= 0 and _i_offset >= 0 \
    #         else _img_array_2[_j_offset, _i_offset]
    #     Debug.
    #     _frame_num = _j + 1 + (_width - 1) * _i
    #     _pool.apply_async(func=visualize, args=(_img_array_2, int(_frame_num / _width), False)) \
    #         if _frame_num % _width == 0 else None
    #     text = f"j,i = {_j, _i} -> {int(_j_offset), int(_i_offset)}"
    #     new = True if _i == 0 and _j == 0 else False
    #     log_writer(text, new)
    tqdm.write('Writing into pictures...')
    # _pool.close()
    # _pool.join()
    # ! ================================================================= OLD VERSION START
    # _max_i, _max_j, _ = np.dot(np.dot(_k2, np.linalg.inv(_k1)),
    #                            np.array([_width, _height, 1]))
    # _img_array_2 = np.zeros((int(_max_j), int(_max_i), 3))
    # for _i in trange(0, _width, position=0, desc="X-axis of the image; 1-axis of the array"):
    #     for _j in range(0, _height):
    #         # * 计算K2·K1.I·[u1, v1, 1].T=[u2, v2, 1]
    #         # ? array()不同于matrix()，后者可以使用mat.I求逆，mat.H求共轭
    #         # ? 详情参考 https://blog.csdn.net/lfj742346066/article/details/77880668
    #         _transformed_i, _transformed_j, _ = np.dot(np.dot(_k2, np.linalg.inv(_k1)), np.array([_i, _j, 1]))
    #         # * 将A的_i, _j位置的值赋给B对应的位置
    #         # ! 图片中的像素坐标是（横，纵），但Array中的坐标是（纵，横）
    #         _img_array_2[int(_transformed_j), int(_transformed_i)] = _img_array_1[_j, _i]
    #
    #         frame_num = _j + 1 + (_width - 1) * _i
    #         visualize(_img_array_2, int(frame_num / _width), _video=False) \
    #             if frame_num % _width == 0 else None
    # ! ================================================================= OLD VERSION END
    # text = f"j,i = {_j, _i} -> {int(_transformed_j), int(_transformed_i)}"
    # new = True if _i == 0 and _j == 0 else False
    # log_writer(text, new)
    # print(int(_transformed_j), int(_transformed_i)) # if _i >= 2175 else None
    # visualize(_img_array_2, -1, _video=True)
    # * ================================================================================================================
    # cv2.namedWindow(f"normalized_image", cv2.WINDOW_NORMAL)
    cv2.imwrite(f"normalized_image_{_original_img_path.split('/')[-1]}", _img_array_2)
    # cv2.imshow('normalized image', _img_array_2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # * ./lv_miui_5_25.35_3472_4624.jpg
    # given_intrinsic_matrix = np.array([[1496.52936761968, 0, 2312.0],
    #                                    [0, 1685.53740193842, 1736.0],
    #                                    [0, 0, 1]])
    # * ./wu_p30_2_17.35_3456_4608.jpg
    # given_intrinsic_matrix = np.array([[595.04514114, 0., 1728.],
    #                                    [0., 1190.09028229, 2304.],
    #                                    [0., 0., 1.]])
    # * ./wu_p30_2_17.35_4608_3456.jpg
    # given_intrinsic_matrix = np.array([[793.39352153, 0., 2304.],
    #                                    [0., 892.56771172, 1728.],
    #                                    [0., 0., 1.]])
    # * ./wu_p30_6_27.35_2736_3648.jpg
    # given_intrinsic_matrix = np.array([[932.85162807, 0., 1368.],
    #                                    [0., 1865.70325615, 1824.],
    #                                    [0., 0., 1.]])
    # * ./wu_p30_6_27.35_3648_2736.jpg
    # given_intrinsic_matrix = np.array([[1243.80217076511, 0, 1824.0],
    #                                    [0, 1399.27744211075, 1368.0],
    #                                    [0, 0, 1]])
    # ! original_img_path 的路径中不可以出现中文字符，否则cv2.imread()读取不到
    given_intrinsic_matrix = compute_k('pictures/wu_p30_2_17.35_4608_3456.jpg')
    # original_img_path = 'pictures/lv_miui_5_25.35_3472_4624.jpg'
    # original_img_path = 'pictures/lv_miui_5_25.35_3472_4624.jpg'
    # original_img_path = 'pictures/wu_p30_2_17.35_4608_3456.jpg'
    # original_img_path = 'pictures/wu_p30_6_27.35_2736_3648.jpg'
    original_img_path = 'pictures/wu_p30_6_27.35_3648_2736.jpg'
    intrinsic2image(given_intrinsic_matrix, original_img_path)
