import numpy as np
import matplotlib.pyplot as plt
import random
import pathlib


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


class MinMaxFilter:
    @staticmethod
    def _apply_pooled(arr, call, fill_val, pool=3):
        if type(arr) is not np.ndarray:
            arr = np.array(arr)
        pool = int(pool) - 1
        result = np.full(shape=(arr.shape[0]-pool, *arr.shape[1:]), fill_value=fill_val, dtype=arr.dtype)
        for i in range(pool):
            result = call(result, arr[i:-(pool-i)])
        return result

    @staticmethod
    def minmax_filter(img, pool=(3, 3)):
        if type(img) is not np.ndarray:
            img = np.array(img)
        _hb = int(pool[0]/2)
        _vb = int(pool[1]/2)

        axes_t = np.arange(len(img.shape))
        axes_t[0] = 1
        axes_t[1] = 0
        result = np.pad(img, ((_hb, _hb), (_vb, _vb)), mode='constant', constant_values=np.iinfo(img.dtype).max)
        result = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[0])
        result = np.transpose(result, axes=axes_t)
        result = MinMaxFilter._apply_pooled(result, np.minimum, np.iinfo(result.dtype).max, pool[1])
        result = np.pad(result, ((_vb, _vb), (_hb, _hb)), mode='constant', constant_values=np.iinfo(img.dtype).min)
        result = MinMaxFilter._apply_pooled(result, np.maximum, np.iinfo(result.dtype).min, pool[1])
        result = np.transpose(result, axes=axes_t)
        result = MinMaxFilter._apply_pooled(result, np.maximum, np.iinfo(result.dtype).min, pool[0])
        return result


class ImageAnalyser:
    # TODO: add iterative object highlighting described in lab manual
    @staticmethod
    def contours(img):
        if type(img) is not np.ndarray:
            img = np.ndarray(img)
        pads = ((1, 1), (1, 1))
        img = np.pad(img, pads, mode="constant", constant_values=0)
        increments = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        prev_move = 0
        current_class = 2
        for x in range(pads[0][0], img.shape[0]-pads[0][1]):
            for y in range(pads[1][0], img.shape[1]-pads[1][1]):
                if img[x][y] == 1:
                    entrance = (x, y)
                    trace = current_class
                    while True:
                        for i in range(prev_move + len(increments) + 1, prev_move + 1, -1):
                            i = i % len(increments)
                            if img[x + increments[i][0]][y + increments[i][1]] == 1:
                                prev_move = i
                                break
                        else:
                            if img[x][y] == 1:
                                img[x][y] = trace
                                prev_move += int(len(increments) / 2)
                            while img[x][y] != current_class:
                                for i in range(prev_move - 1, prev_move + len(increments) - 1):
                                    i = i % len(increments)
                                    if img[x + increments[i][0]][y + increments[i][1]] == img[x][y] - 1:
                                        prev_move = i
                                        img[x][y] -= 1
                                        x += increments[prev_move][0]
                                        y += increments[prev_move][1]
                                        break
                                else:
                                    pass
                                for i in range(prev_move - 1, prev_move + len(increments) - 1):
                                    i = i % len(increments)
                                    if img[x + increments[i][0]][y + increments[i][1]] == 1:
                                        # img[x][y] += 1
                                        prev_move = i
                                        trace = img[x][y]
                                        break
                                else:
                                    continue
                                break
                            else:
                                break
                        img[x][y] = trace
                        x += increments[prev_move][0]
                        y += increments[prev_move][1]
                        trace += 1
                        pass
                    img[img >= current_class] = current_class
                    current_class += 1
        return img[pads[0][0]:-pads[0][1], pads[1][0]: -pads[1][1]], current_class

    @staticmethod
    def calculate_signs(img, obj_count=2):
        if type(img) is not np.ndarray:
            img = np.array(img)
        collection = np.zeros((obj_count-2, 5), dtype=float)

        weights = np.indices((img.shape[0], img.shape[1]))
        for i in range(2, obj_count):
            masked = np.where(img == i, 1, 0).astype(np.uint8)
            indices = np.argwhere(img == i)
            i_start = np.amin(indices, axis=0)
            i_stop = np.amax(indices, axis=0) + 1
            img_slice = np.pad(masked[i_start[0]:i_stop[0], i_start[1]:i_stop[1]], ((1, 1), (1, 1)),
                               mode='constant', constant_values=0)

            # mc = (
            #         np.sum(masked * weights[1], dtype=np.uint32) / collection[i - 2][1],
            #         np.sum(masked * weights[0], dtype=np.uint32) / collection[i - 2][1]
            # )

            origin = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img_slice.dtype)
            opened = ImageAnalyser._spatial_correlation(img_slice, origin)

            collection[i - 2][0] = \
                np.count_nonzero(np.logical_and(np.logical_not(img_slice.astype(bool)), opened.astype(bool)))
            collection[i - 2][1] = np.count_nonzero(img_slice)
            collection[i - 2][2] = collection[i - 2][0]**2/collection[i - 2][1]
            m1 = (
                (weights[1] - np.sum(masked * weights[1], dtype=np.uint32) / collection[i - 2][1]) * masked,
                (weights[0] - np.sum(masked * weights[0], dtype=np.uint32) / collection[i - 2][1]) * masked
            )
            mij = (np.sum(m1[0] ** 2), np.sum(m1[0] * m1[1]), np.sum(m1[1] ** 2))
            temp = np.sqrt(((mij[0] - mij[2]) ** 2 + 4 * mij[1] ** 2))

            collection[i - 2][3] = (mij[0] + mij[2] - temp) / (mij[0] + mij[2] + temp)
            collection[i - 2][4] = (np.arctan(2 * mij[1] / (mij[0] - mij[2]))) / 2
            if np.any(np.isnan(collection[i - 2][4])):
                collection[i - 2][4] = 0
        return collection

    @staticmethod
    def cluster(signs, class_num: int):
        support = np.array(random.sample(range(signs.shape[0]), k=class_num))
        classes = np.zeros((signs.shape[0]), dtype=int)
        f = np.zeros(shape=support.shape, dtype=float)
        for i in range(len(classes)):
            dists = np.linalg.norm(signs[support] - signs[i], axis=1)
            classes[i] = np.argmin(dists)
            f[classes[i]] += dists[classes[i]]
        p_support = np.zeros(support.shape)
        f_map = {i: tuple((f[i], support[i])) for i in range(len(support))}
        while np.any(p_support != support):
            p_support = np.copy(support)
            for i in range(len(support)):
                population = np.argwhere(classes == i)
                for j in range(p_support[i], p_support[i] + len(population)):
                    _j = j % len(population)
                    f_value = np.sum(np.linalg.norm(signs[population]-signs[population[_j]], axis=1))
                    if f_value < f_map[i][0]:
                        f_map[i] = (f_value, population[_j])
            for i in range(len(support)):
                support[i] = f_map[i][1]
        return classes, support

    @staticmethod
    def _area(img, obj_num: int):
        if type(img) is not np.ndarray:
            img = np.ndarray
        return np.count_nonzero(img == obj_num)

    @staticmethod
    def _mass_center(img, obj_num: int = 1, area: int = None, normalized: bool = False):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if obj_num == 0:
            raise ValueError
        if normalized and obj_num != 1:
            raise ValueError
        weights = np.indices((img.shape[0], img.shape[1]))
        if normalized:
            if area is None:
                area = np.sum(img)
            x_mc = np.sum(img * weights[1], dtype=np.uint32) / area
            y_mc = np.sum(img * weights[0], dtype=np.uint32) / area
        else:
            if area is None:
                area = ImageAnalyser._area(img, obj_num)
            x_mc = np.sum((img * weights[1])[img == obj_num], dtype=np.uint32) / area / obj_num
            y_mc = np.sum((img * weights[0])[img == obj_num], dtype=np.uint32) / area / obj_num
        return x_mc, y_mc

    @staticmethod
    def _perimeter(img, obj_num: int = 1, normalized: bool = False):
        if type(img) is not np.ndarray:
            img = np.array(img)
        if normalized and obj_num != 1:
            raise ValueError
        origin = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img.dtype)
        if normalized:
            opened = ImageAnalyser._spatial_correlation(img, origin)
            return np.count_nonzero(np.logical_and(np.logical_not(img.astype(bool)), opened.astype(bool)))
        masked = img == obj_num
        opened = ImageAnalyser._spatial_correlation(np.where(masked, 1, 0), origin)
        return np.count_nonzero(np.logical_and(np.logical_not(masked), opened.astype(bool)))

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
        indices = np.indices((img.shape[0], img.shape[1]))
        m1 = (
            np.where(img == obj_num, indices[1] - mc[0], 0),
            np.where(img == obj_num, indices[0] - mc[1], 0)
        )
        mij = (np.sum(m1[0] ** 2), np.sum(m1[0] * m1[1]), np.sum(m1[1] ** 2))
        temp = np.sqrt(((mij[0] - mij[2]) ** 2 + 4 * mij[1] ** 2))
        # TODO: returns NaN, should be float
        # return (mij[0] + mij[2] + temp) / (mij[0] + mij[2] - temp)
        result = tuple(((mij[0] + mij[2] - temp) / (mij[0] + mij[2] + temp), (np.arctan(2*mij[1] / (mij[0] - mij[2]))) / 2))
        if result[0] is np.NAN or result[1] is np.NAN:
            pass
        return result

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


def lab(path):
    params = [-1, 2]
    labels = [f"Binarization threshold [mean]: ", f"Number of classes [2]: "]

    img = plt.imread(path)
    # fig = plt.figure()
    # axes = fig.add_subplot()
    # axes.imshow(img)
    # axes.set_axis_off()
    # plt.show()

    for i in range(len(params)):
        try:
            temp = int(input(labels[i]))
            params[i] = temp if temp > 0 else params[i]
        except ValueError:
            continue

    img_g = ImageTransformer.to_grayscale(img)

    img_filtered = MinMaxFilter.minmax_filter(img_g.astype(np.uint8), (5, 5))
    img_n = ((img_filtered - np.amin(img_filtered)) *
             ((np.iinfo(img_filtered.dtype).max - np.iinfo(img_filtered.dtype).min) /
              (np.amax(img_filtered) - np.amin(img_filtered)))).astype(np.uint8)

    if params[0] == -1:
        params[0] = int(np.mean(img_n))
    img_b = ImageTransformer.binarize(img_n, params[0])

    segmented, obj_count = ImageAnalyser.contours(img_b)
    if obj_count > 2:
        sign_vectors = ImageAnalyser.calculate_signs(segmented, obj_count)
        classes, support_vectors = ImageAnalyser.cluster(sign_vectors, params[1])

        plot_content = [
            img, img_g, img_n,
            img_b,
            segmented,
        ]
        for _class in range(params[1]):
            _cluster = np.zeros(img_b.shape)
            for i in range(len(classes)):
                if classes[i] == _class:
                    _cluster += np.where(segmented == (i+2), 1, 0)
            plot_content.append(_cluster)

        figures = []
        for item in plot_content:
            fig = plt.figure()
            axes = fig.add_subplot()
            axes.imshow(item)
            axes.set_axis_off()
            figures.append(fig)
        plt.show()


if __name__ == '__main__':
    images = [p for p in pathlib.Path("img/easy").iterdir() if p.suffix in [".jpg", ".jpeg"]]
    for image in images:
        print(f"Proceed? {image}")
        if input("[y]/n: ") != "n":
            lab(image)
        # lab(image)
        if input("Quit? [y]/n: ") != "n":
            break
