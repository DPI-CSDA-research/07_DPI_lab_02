import numpy as np
import matplotlib.pyplot as plt
import random


class ImageTransformer:
    @staticmethod
    def to_grayscale(img):
        if type(img) is not np.ndarray:
            img = np.array(img)
        result = np.zeros(shape=img.shape[0:2], dtype=img.dtype)
        if len(img.shape) == 3:
            for i in range(img.shape[2]):
                result = result / (i + 1) * i + img[..., i] / (i + 1)
        return result

    @staticmethod
    def binarize(img, threshold=128):
        if type(img) is not np.ndarray:
            img = np.array(img)
        return np.where(img < threshold, 0, 1)


class ImageAnalyser:
    @staticmethod
    def contours(img):
        if type(img) is not np.ndarray:
            img = np.ndarray(img)
        increments = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        prev_move = 0
        current_class = 2
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] == 1:
                    entrance = (x, y)
                    while True:
                        for i in range(prev_move, prev_move + len(increments)):
                            i = i % len(increments)
                            if img[x + increments[i][0]][y + increments[i][1]] == 1:
                                prev_move = i
                                break
                        else:
                            for i in range(prev_move, prev_move + len(increments)):
                                i = i % len(increments)
                                if img[x + increments[i][0]][y + increments[i][1]] == current_class:
                                    prev_move = i
                                    break
                            if x == entrance[0] and y == entrance[1]:
                                break
                        img[x][y] = current_class
                        x += increments[prev_move][0]
                        y += increments[prev_move][1]
                    current_class += 1
        return img, current_class

    @staticmethod
    def calculate_signs(img, obj_count=2):
        if type(img) is not np.ndarray:
            img = np.array(img)
        collection = np.zeros((obj_count-2, 7), dtype=float)
        for i in range(2, obj_count):
            mc = ImageAnalyser._mass_center(img, i)
            layout = ImageAnalyser._elongation(img, i, mc)
            collection[i-2][0] = ImageAnalyser._area(img, i)
            collection[i-2][1] = mc[0]
            collection[i-2][2] = mc[1]
            collection[i-2][3] = ImageAnalyser._perimeter(img, i)
            collection[i-2][4] = ImageAnalyser._density(img, i, collection[i-2][2], collection[i-2][0])
            collection[i-2][5] = layout[0]
            collection[i-2][6] = layout[1]
        return collection

    @staticmethod
    def cluster(signs, class_num: int):
        support = np.array(random.choices(range(signs.shape[0]), k=class_num))
        classes = np.zeros((signs.shape[0]), dtype=int)
        f = np.zeros(shape=support.shape, dtype=float)
        for i in range(len(classes)):
            dists = np.norm(signs[support] - signs[i], axis=0)
            classes[i] = np.argmin(dists)
            f[classes[i]] += dists[classes[i]]
        p_support = np.zeros(support.shape)
        f_map = {i: tuple((f[i], support[i])) for i in range(len(support))}
        while p_support != support:
            p_support = np.copy(support)
            for i in range(len(support)):
                population = np.argwhere(classes == i)
                for j in range(p_support[i], p_support[i] + len(population)):
                    _j = j % len(population)
                    f_value = np.sum(np.norm(signs[population]-signs[population[j]], axis=0))
                    if f_value < f_map[i][0]:
                        f_map[i] = (f_value, population[j])
            for i in range(len(support)):
                support[i] = f_map[i][1]
        return classes, support

    @staticmethod
    def _area(img, obj_num: int):
        if type(img) is not np.ndarray:
            img = np.ndarray
        return np.count_nonzero(img == obj_num)

    @staticmethod
    def _mass_center(img, obj_num: int):
        if type(img) is not np.ndarray:
            img = np.array(img)
        weights = np.indices((img.shape[0], img.shape[1]))
        area = ImageAnalyser._area(img, obj_num)
        x_mc = np.sum((img * weights[1])[img == obj_num], dtype=np.uint32) / area
        y_mc = np.sum((img * weights[0])[img == obj_num], dtype=np.uint32) / area
        return x_mc, y_mc

    @staticmethod
    def _perimeter(img, obj_num: int):
        if type(img) is not np.ndarray:
            img = np.array(img)
        origin = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img.dtype)
        masked = img == obj_num
        opened = ImageAnalyser._spatial_correlation(np.where(masked, 1, 0), origin)
        return np.count_nonzero(np.logical_and(not masked, opened))

    @staticmethod
    def _density(img, obj_num: int, p=None, s=None):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if p is None or type(p) is not int:
            p = ImageAnalyser._perimeter(img, obj_num)
        if s is None or type(s) is not int:
            s = ImageAnalyser._area(img, obj_num)
        return p**2/s

    @staticmethod
    def _elongation(img, obj_num: int, mc=None):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if mc is None or type(mc) is not tuple:
            mc = ImageAnalyser._mass_center(img, obj_num=obj_num)
        indicies = np.indicies((img.shape[0], img.shape[1]))
        m1 = (
            np.where(img == obj_num, indicies[1] - mc[0], 0),
            np.where(img == obj_num, indicies[0] - mc[1], 0)
        )
        mij = (np.sum(m1[0] ** 2), np.sum(m1[0] * m1[1]), np.sum(m1[1] ** 2))
        temp = np.sqrt(((mij[0] - mij[2]) ** 2 + 4 * mij[1] ** 2))
        return (mij[0] + mij[2] + temp) / (mij[0] + mij[2] - temp), (np.arctan(2*mij[1] / (mij[0] - mij[2]))) / 2

    @staticmethod
    def _spatial_correlation(img, origin):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if type(origin) is not np.ndarray:
            origin = np.array(origin)
        pads = (int(origin.shape[0] / 2), int(origin.shape[1] / 2))
        buf = np.pad(img, ((pads[0], origin.shape[0]-pads[0]), (pads[1], origin.shape[1]-pads[1])),
                     mode="constant", constant_values=0)
        result = np.zeros(img.shape, dtype=img.dtype)
        for x in range(buf.shape[0]-origin.shape[0]):
            for y in range(buf.shape[1]-origin.shape[1]):
                current_slice = buf[x:x+origin.shape[0], y:y+origin.shape[1]]
                result[x][y] = np.sum(np.multiply(current_slice, origin))
        return result
    pass


def lab():
    pass


if __name__ == '__main__':
    pass
