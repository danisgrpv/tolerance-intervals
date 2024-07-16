import numpy as np
import scipy as sc
from scipy import stats


class ConfidenceCalculator:

    def __init__(self, distr):
        self.d = distr


    def search_nearest(self, array, target):
        """
        Возвращает индекс элемента массива,
        ближайшего к заданному

        Параметры:
            1. array - массив значений
            2. target - искомое значение

        Возвращаемые значения:
            1. out - индекс элемента массива,
            ближайшего к заданному
        """
        out = np.argmin(np.abs(array - target))
        return out


    def calc_coverages(self, sample_size, num_of_events,
                       tolerance_factor_lower, tolerance_factor_upper):
        """
        Вычисляет долю накрытия генеральной совокупности с 
        заданым законом распределения толерантными интервалами
        с заданными границами. Розыгрыш событий производится
        заданное число раз для выборки заданного размера

        Параметры:
            1. sample_size - размер выборки
            2. num_of_events - число статистических испытаний
            3. tolerance_factor_lower - массив нижних толерантных коэффициентов
            4. tolerance_factor_upper - массив верхних толерантных коэффициентов

        Вычисление производится для каждой пары коэффициентов

        Возвращаемые значения:
            1. true_coverages - статистически вычисленная доля накрытия
            генеральной совокупности (для каждого испытания)
            толерантным интервалом с коэффициентами
            tolerance_factor_lower и tolerance_factor_upper
                
            Размерность массива calc_coverages
            [len(tolerance_factor_lower)][num_of_events][len(tolerance_factor_upper)]
        """
        self.k1 = tolerance_factor_lower
        self.k2 = tolerance_factor_upper
        sample = self.d(0, 1).rvs(size=(sample_size, num_of_events))

        sample_loc = np.empty(shape=sample.shape[1])
        sample_scale = np.empty(shape=sample.shape[1])

        for si in range(sample.shape[1]):
            sample_loc[si], sample_scale[si] = self.d.fit(sample[:, si])
        
        loc_grid, k1_grid, k2_grid = np.meshgrid(sample_loc, self.k1, self.k2)
        scale_grid, k1_grid, k2_grid = np.meshgrid(sample_scale, self.k1, self.k2)
        
        l1 = loc_grid - k1_grid * scale_grid
        l2 = loc_grid + k2_grid * scale_grid
        self.true_coverages = self.d.cdf(l2) - self.d.cdf(l1)
        self.K1, self.K2 = np.meshgrid(self.k1, self.k2)
    

    def calc_confidence(self, check_coverages):
        """
        Вычисляет доверительную вероятность
        события A = {P_calc >= P_req} - того, что истинная
        доля накрытия не меньше некоторой заданной

        Параметры:
            1. check_coverages - доля накрытия
                (значение статистики для которой вычисляется
                доверительная вероятность)

        Возвращаемые значение:
            1. confidence - вычисленная доверительная вероятность

            Размерность массива confidence
            [len(tolerance_factor_lower)][len(tolerance_factor_upper)]
        """
        # Ось вдоль которой расположены результаты
        # каждого статистического испытания
        events_ax = 1

        if isinstance(check_coverages, list):
            rl = len(check_coverages)
        elif isinstance(check_coverages, np.ndarray):
            rl = check_coverages.shape[0]
        elif isinstance(check_coverages, np.float):
            rl = 1
            check_coverages = [check_coverages]

        cl = np.empty(shape=(rl), dtype=object)
        
        for i in range(rl):
            successes = (self.true_coverages >= check_coverages[i]).sum(axis=events_ax)
            conf = successes / self.true_coverages.shape[events_ax]
            cl[i] = conf
        
        self.confidence = cl
        self.check_coverages = check_coverages
        self.pair = [check_coverages, cl]


    def estimate(self, sample, required_confidence, lcb, ucb):
        """
        Возвращает значение доли накрытия P, для которого уровень доверия максимально близок к заданному

        Параметры:
            1. sample - массив выборочных значений по которым строится оценка
            2. required_confidence - требуемая доверительная вероятность
            3. lcb - нижняя граница параметра критерия годности
            4. ucb - верхняя граница параметра критерия годности

        Возвращаемое значение:
            1. out - граница толерантного интервала,
                содержащего значения заданной статистики
                с заданной доверительной вероятностью
        """
        # Оценки параметров сдвига и масштаба распределения 
        loc, scale = self.d.fit(sample)
        # Вычисление толерантных коэффициентов
        k1_estim = (loc - lcb) / scale
        k2_estim = (ucb - loc) / scale
        k1_indices = self.search_nearest(self.k1, k1_estim)
        k2_indices = self.search_nearest(self.k2, k2_estim)

        distance_init = 1e6
        covmesh, confs = self.pair

        for j, arr in enumerate(confs):
            confidence_level = arr[k1_indices][k2_indices]
            distance = np.abs(confidence_level - required_confidence)

            if distance <= distance_init:
                distance_init = distance
                coverages = covmesh[j]

        return coverages