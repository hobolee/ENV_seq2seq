import os.path

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import statsmodels.api as sm
import random
import netCDF4 as nc


class DataInterpolate:
    def __init__(self, dt, station_index):
        """
        :param dt: original data
        :param station_index: station number
        """
        print('Initialization...')
        self.origin_data = dt[:, 1:]
        self.origin_data[4:, :] = self.origin_data[4:, :].astype(float)
        self.origin_T = dt[:, 0]
        self.original_X = np.arange(len(self.origin_data))
        self.operate_data = self.origin_data
        self.operate_T = self.origin_T
        self.operate_X = self.original_X
        self.validate_X = None
        self.validate_data_true = None
        self.validate_data_period = []
        self.validate_data_linear = []
        self.valid_X = None
        self.valid_data = None
        self.invalid_X = None
        self.invalid_data = None
        self.index = station_index
        self.integrity = []
        self.day_data = None
        self.station_num = len(self.origin_data[0, :])
        self.invalid_data_period = None
        self.invalid_data_linear = None
        self.validate_data_period = []
        self.validate_data_linear = []
        self.operate_data_period = None
        self.operate_data_linear = None
        self.validate_X_index = []
        self.validate_data_true_index = []
        self.station_id = self.origin_data[0, :]
        self.location = (self.origin_data[1:3, :])

    def generate_dataset(self, interval, year=None):
        """
        To generate some subset dataset
            operate_data: [n, stations], the data will be operated
            valid_data: data in operate data which are not -99999
            invali_data: data in operate data which are -99999
            validate_data: [n, stations], the data will be used for validation.
            valid_data + invalid_data + validate_data = operate_data
        :param interval: int, length of validate_data
        :param year: list, which years are going to be operated. If None, all data.
        :return:
        """
        print('Generating dataset...')
        if year:
            s_t, e_t = self.search_year(start_year=year[0], end_year=year[1])
            self.operate_data = self.origin_data[s_t:e_t + 1, :]
            self.operate_T = self.origin_T[s_t:e_t + 1]
            self.operate_X = np.arange(len(self.operate_data))
        s_t, e_t = self.generate_val_data(self.operate_data[:, self.index], interval)
        total_num = min(40, len(s_t) // 4 * 4)
        num = random.sample(s_t, total_num)
        num = [i + 12 for i in num]
        for i in range(0, total_num // 4):
            self.validate_X_index.append(list(self.operate_X[num[i]:num[i] + 18]))
            self.validate_data_true_index.append((list(self.operate_data[num[i]:num[i] + 18, self.index])))
        for i in range(total_num // 4, total_num // 2):
            self.validate_X_index.append(list(self.operate_X[num[i]:num[i] + 6]))
            self.validate_data_true_index.append((list(self.operate_data[num[i]:num[i] + 6, self.index])))
        for i in range(total_num // 2, total_num // 4 * 3):
            self.validate_X_index.append(list(self.operate_X[num[i]:num[i] + 12]))
            self.validate_data_true_index.append((list(self.operate_data[num[i]:num[i] + 12, self.index])))
        for i in range(total_num // 4 * 3, total_num):
            self.validate_X_index.append(list(self.operate_X[num[i]:num[i] + 24]))
            self.validate_data_true_index.append((list(self.operate_data[num[i]:num[i] + 24, self.index])))
        self.validate_X = self.operate_X[num[0]:num[0] + 1]
        self.validate_data_true = self.operate_data[num[0]:num[0] + 1, self.index]
        for i in range(1, total_num // 4):
            self.validate_X = np.append(self.validate_X, [self.operate_X[num[i]:num[i] + 18]])
            self.validate_data_true = np.append(self.validate_data_true,
                                                [self.operate_data[num[i]:num[i] + 18, self.index]])
        for i in range(total_num // 4, total_num // 2):
            self.validate_X = np.append(self.validate_X, [self.operate_X[num[i]:num[i] + 6]])
            self.validate_data_true = np.append(self.validate_data_true,
                                                [self.operate_data[num[i]:num[i] + 6, self.index]])
        for i in range(total_num // 2, total_num // 4 * 3):
            self.validate_X = np.append(self.validate_X, [self.operate_X[num[i]:num[i] + 12]])
            self.validate_data_true = np.append(self.validate_data_true,
                                                [self.operate_data[num[i]:num[i] + 12, self.index]])
        for i in range(total_num // 4 * 3, total_num):
            self.validate_X = np.append(self.validate_X, [self.operate_X[num[i]:num[i] + 24]])
            self.validate_data_true = np.append(self.validate_data_true,
                                                [self.operate_data[num[i]:num[i] + 24, self.index]])
        data = list(zip(self.operate_X, self.operate_data[:, self.index]))[:]
        valid_data = [data[i] for i in range(len(self.operate_data)) if
                      data[i][1] != -99999 and i not in self.validate_X]
        invalid_data = [data[i] for i in range(len(self.operate_data)) if data[i][1] == -99999]
        self.valid_X, self.valid_data = zip(*valid_data)
        self.invalid_X, _ = zip(*invalid_data)
        # for i in range(len(self.validate_X)):
        self.operate_data[self.validate_X, :] = -99999
        self.day_data = np.array(self.operate_data).reshape([-1, 24, 14])

    @staticmethod
    def generate_val_data(t_data, interval):
        """
        To find continuous data clip which does not contain any -99999
        :param t_data: data
        :param interval: int, length of validate_data
        :return: start_t, end_t, index of the clip
        """
        interval = interval * 2  # To avoid the clip start after -99999
        start_t = []
        end_t = []
        count = 0
        state = 'invalid'
        flag = True
        for i, d in enumerate(t_data):
            if d != -99999:
                if state == 'invalid':
                    state = 'valid'
                elif count >= interval and flag is True:
                    start_t.append(i - interval)
                    flag = False
                count += 1
            else:
                if not flag:
                    flag = True
                    end_t.append(i - 1)
                count = 0
        if len(start_t) > len(end_t) and flag is False:
            end_t.append(len(t_data) - 1)
        return start_t, end_t

    def search_year(self, start_year=1983, end_year=None):
        """
        Return start year 01/01 00:00 ~ end_year-1 12/31 23:00
        :param start_year: int
        :param end_year: int or none if it is 2022
        :return: start year 01/01 00:00 ~ end_year-1 12/31 23:00
        """
        start_number, end_number = -1, -1
        s_flag, e_flag = True, True
        end_year = None if end_year == 2022 else end_year
        if end_year and (s_flag or e_flag):
            end_year += 1
            for i in range(len(self.origin_data)):
                if self.origin_T[i][:4] == str(start_year) and s_flag:
                    start_number = i
                    s_flag = False
                elif self.origin_T[i][:4] == str(end_year) and e_flag:
                    end_number = i
                    e_flag = False
        elif not end_year:
            end_number = len(self.origin_data)
            for i in range(len(self.origin_data)):
                if self.origin_T[i][:4] == str(start_year) and s_flag:
                    start_number = i
                    s_flag = False
        return start_number, end_number - 1

    def plot(self):
        """
        Plot data.
        :return:
        """
        print('Plotting...')
        fig, ax = plt.subplots()
        ax.plot(self.operate_X, self.operate_data_period[:, self.index], label='Period')
        ax.plot(self.operate_X, self.operate_data_linear, label='Linear')
        self.invalid_data_period = list(self.invalid_X)
        self.invalid_data_linear = list(self.invalid_X)
        for i in range(len(self.invalid_X)):
            self.invalid_data_period[i] = self.operate_data_period[self.invalid_X[i], self.index]
            self.invalid_data_linear[i] = self.operate_data_linear[self.invalid_X[i]]
        ax.scatter(self.invalid_X, self.invalid_data_period)
        ax.scatter(self.invalid_X, self.invalid_data_linear)
        for i in range(len(self.validate_X_index)):
            if i == len(self.validate_X_index) - 1:
                if len(self.validate_X_index[i]) == 1:
                    ax.scatter(self.validate_X_index[i], self.validate_data_true_index[i], marker='*', c='b', s=100)
                else:
                    ax.plot(self.validate_X_index[i], self.validate_data_true_index[i], 'b', label='Validation')
            else:
                if len(self.validate_X_index[i]) == 1:
                    ax.scatter(self.validate_X_index[i], self.validate_data_true_index[i], marker='*', c='b', s=100)
                else:
                    ax.plot(self.validate_X_index[i], self.validate_data_true_index[i], 'b')
        ax.legend()
        tick = [i[:4] for i in self.operate_T if i[4:] == '/1/1 00:00']
        tick.append(str(int(tick[-1]) + 1))
        year_number = len(self.operate_data) // (365 * 24) + 1
        if year_number >= 30:
            plt.xticks(np.arange(0, 8760 * year_number, 8760 * 4), [j for i, j in enumerate(tick) if i % 4 == 0])
        elif 15 < year_number < 30:
            plt.xticks(np.arange(0, 8760 * year_number, 8760 * 2), [j for i, j in enumerate(tick) if i % 2 == 0])
        else:
            plt.xticks(np.arange(0, 8760 * year_number, 8760), [j for i, j in enumerate(tick)])
        plt.ylabel('ppb')
        plt.xlabel('Time')
        plt.show()

    def interpolate(self, func, kind='linear'):
        """
        Interpolation
        :param func: Interpolation method
        :param kind: Interpolation kind
        :return:
        """
        print('Interpolating with ' + str(kind) + '...')
        if func == 'OLS':
            # To be finished.
            index = 0
            for i in range(len(self.invalid_X)):
                self.valid_X = list(self.valid_X)
                self.valid_data = list(self.valid_data)
                if self.invalid_X[i] not in self.validate_X:
                    index = self.valid_X[index:].index(self.invalid_X[i] - 1) + index
                    if index < 24:
                        continue
                    xx = np.array(self.valid_X[index - 24:index + 24])
                    xx = np.column_stack(
                        (xx ** 0.5, xx, xx ** 1.5, xx ** 2, xx ** 2.5, xx ** 3, xx ** 3.5, xx ** 4, xx ** 4.5))
                    xx = sm.add_constant(xx)
                    model = sm.OLS(self.valid_data[index - 24:index + 24], xx)
                    results = model.fit()
                    x_pre = np.array([1, self.invalid_X[i] ** 0.5, self.invalid_X[i], self.invalid_X[i] ** 1.5,
                                      self.invalid_X[i] ** 2,
                                      self.invalid_X[i] ** 2.5, self.invalid_X[i] ** 3, self.invalid_X[i] ** 3.5,
                                      self.invalid_X[i] ** 4,
                                      self.invalid_X[i] ** 4.5])
                    y_pre = results.predict(x_pre)
                    print(y_pre)
                    self.valid_X = np.append(self.valid_X, x_pre[2])
                    self.valid_data = np.append(self.valid_data, y_pre)
                    tmp = list(zip(self.valid_X, self.valid_data))
                    tmp.sort()
                    self.valid_X, self.valid_data = zip(*tmp)
        elif func == 'interpolate.interp1d':
            if kind == 'linear':
                self.valid_X = list(self.valid_X)
                self.valid_data = list(self.valid_data)
                self.valid_X.append(0)
                self.valid_data.append(0)
                f = interpolate.interp1d(self.valid_X, self.valid_data, kind=kind)
                self.operate_data_linear = f(self.operate_X)
                self.validate_data_linear = f(self.validate_X)
            else:
                # Window slide, it is not necessary if interpolate method is linear
                index = 0
                for i in range(0, len(self.invalid_X)):
                    if i == 0:
                        y_pre = 0
                        self.valid_X = np.append(self.valid_X, self.invalid_X[i])
                        self.valid_data = np.append(self.valid_data, y_pre)
                        tmp = list(zip(self.valid_X, self.valid_data))
                        tmp.sort()
                        self.valid_X, self.valid_data = zip(*tmp)
                        continue
                    self.valid_X = list(self.valid_X)
                    self.valid_data = list(self.valid_data)
                    if self.invalid_X[i] not in self.validate_X:
                        index = self.valid_X[index:].index(self.invalid_X[i] - 1) + index
                        if index < 24:
                            y_pre = 0
                        else:
                            xx = np.array(self.valid_X[index - 24:index + 24])
                            f = interpolate.interp1d(xx, self.valid_data[index - 24:index + 24], kind=kind)
                            y_pre = f(self.invalid_X[i])
                        self.valid_X = np.append(self.valid_X, self.invalid_X[i])
                        self.valid_data = np.append(self.valid_data, y_pre)
                        tmp = list(zip(self.valid_X, self.valid_data))
                        tmp.sort()
                        self.valid_X, self.valid_data = zip(*tmp)
                index = 0
                for i in range(len(self.validate_X)):
                    self.valid_X = list(self.valid_X)
                    self.valid_data = list(self.valid_data)
                    index = self.valid_X[index:].index(self.validate_X[i] - 1) + index
                    if index < 24:
                        y_pre = 0
                    else:
                        xx = np.array(self.valid_X[index - 24:index + 24])
                        f = interpolate.interp1d(xx, self.valid_data[index - 24:index + 24], kind=kind)
                        y_pre = f(self.validate_X[i])
                    self.validate_data_linear.append(y_pre)
                    self.valid_X = np.append(self.valid_X, self.validate_X[i])
                    self.valid_data = np.append(self.valid_data, y_pre)
                    tmp = list(zip(self.valid_X, self.valid_data))
                    tmp.sort()
                    self.valid_X, self.valid_data = zip(*tmp)
                self.operate_data_linear = self.valid_data

    def period_factor(self, index=None):
        """
        Interpolation with the method of period factor
        :param index: station number list
        :return:
        """
        print('Interpolating with period factor...')
        for day, val_day in enumerate(self.day_data):
            for hour, value in enumerate(val_day):
                for station in index:
                    if value[station] == -99999:
                        self.day_data[day, hour, station] = self.cal_prediction(day, hour, station)

        self.operate_data_period = self.day_data.reshape(-1, self.station_num)
        for i in range(len(self.validate_X)):
            self.validate_data_period.append(self.operate_data_period[self.validate_X[i], self.index])

    def cal_prediction(self, day, hour, station):
        """
        The process of the method period facto
        :param day: The day to be interpolated
        :param hour: The hour to be interpolated
        :param station: The station index
        :return: prediction value
        """

        # To be developed
        # if day < 5, just predict as 0
        if day < 5 or day > len(self.day_data) - 5:
            if hour > 0:
                return self.day_data[day, hour - 1, station]
            elif day > 0:
                return self.day_data[day - 1, 23, station]
            else:
                return 0

        pre_data, suf_data = [], []
        count = 0
        for i in range(day - 1):
            if sum(self.day_data[day - i - 1, :, station]) != -99999 * 24 and count < 5 and sum(self.day_data[day - i - 1, :, station]) != 0:
                pre_data.append(self.day_data[day - i - 1, :, station])
                count += 1
        count = 0
        for i in range(day + 1, len(self.day_data)):
            if sum(self.day_data[i, :, station]) > 0 and count < 5:
                suf_data.append(self.day_data[i, :, station])
                count += 1
        pre_data_ratio = pre_data[0] / np.mean(pre_data[0])
        suf_data_ratio = suf_data[0] / np.mean(suf_data[0])
        for i in pre_data[1:]:
            pre_data_ratio = np.append(pre_data_ratio, i / np.mean(i))
        for i in suf_data[1:]:
            suf_data_ratio = np.append(suf_data_ratio, i / np.mean(i))
        pre_data_ratio = pre_data_ratio.reshape(-1, 24)
        suf_data_ratio = suf_data_ratio.reshape(-1, 24)
        pre_base = np.mean(pre_data[0])
        suf_base = np.mean(suf_data[0])
        pre_median = [np.median(i) for i in np.transpose(pre_data_ratio)]
        suf_median = [np.median(i) for i in np.transpose(suf_data_ratio)]
        pre_mean = [np.mean(i) for i in np.transpose(pre_data_ratio)]
        suf_mean = [np.mean(i) for i in np.transpose(suf_data_ratio)]
        pre_factor, suf_factor = [], []
        for i in range(len(pre_mean)):
            pre_factor.append(pre_mean[i] / 2 + pre_median[i] / 2)
            suf_factor.append(suf_mean[i] / 2 + suf_median[i] / 2)
        # result = [pre_factor[hour] * pre_base / 2 + suf_factor[hour] * suf_base / 2]
        # result = [pre_factor[hour] * pre_base]
        # result = [suf_factor[hour] * suf_base]

        res = [pre_factor[i] * pre_base / 2 + suf_factor[i] * suf_base /2 for i in range(len(pre_factor))]
        # res = [pre_factor[i]*pre_base*0.99 + suf_factor[i]*suf_base*0.01 for i in range(len(pre_factor))]

        # If the true values of other hours' data in the same day are known, use them to correct the prediction value
        bias = 0

        count = 0
        # Values near 0 cannot be interpolated with this method, use the minimum value of the local area instead
        less6_count = 0
        min_val = 99999
        max_val = 0
        for i in range(24):
            if self.day_data[day, i, station] != -99999:
                bias += self.day_data[day, i, station] - res[i]
                # bias = 0
                count += 1
                if self.day_data[day, i, station] < min_val:
                    min_val = self.day_data[day, i, station]
                if self.day_data[day, i, station] > max_val:
                    max_val = self.day_data[day, i, station]
                if self.day_data[day, i, station] < 6:
                    less6_count += 1
        if res[hour] + 1 * bias < 1 and count:
            result = max(res[hour] + 1 * bias / count, min_val)
        elif res[hour] > 2 * max_val and count:
            result = min(res[hour] + 1 * bias / count, max_val)
        else:
            result = res[hour]
        if less6_count >= 10 and self.day_data[day, max(hour - 1, 0), station] < 6:
            'For the near 0 situation'
            if hour > 0:
                result = max(self.day_data[day, hour - 1, station], result)
            else:
                result = max(self.day_data[day - 1, 23, station], result)
            # tmp_d = day
            # tmp_h = hour
            # count = 0
            # for i in range(1, 9999):
            #     tmp_h += 1
            #     if tmp_h > 23:
            #         tmp_h -= 24
            #         day += 1
            #     count += 1
            #     if self.day_data[tmp_d, tmp_h, station] != -99999:
            #         break
            # previous = self.day_data[day, hour-1, station]
            # future = self.day_data[tmp_d, tmp_h, station]
            # result = self.day_data[day, hour-1, station] +
            #        ((self.day_data[tmp_d, tmp_h, station] - self.day_data[day, hour-1, station]) / (count + 1))
            # print(result)
            # c = count
        return result

    @staticmethod
    def cal_rmse(arr1, arr2):
        """
        Calculate the root mean square error.
        """
        print('Calculating RMSE...')
        if type(arr1) is not np.ndarray:
            arr1 = np.array(arr1)
        if type(arr2) is not np.ndarray:
            arr2 = np.array(arr2)
        if len(arr1) != len(arr2):
            raise Exception('Length of two arrays are different.')
        return np.sqrt(np.mean(np.square(arr1 - arr2)))

    @staticmethod
    def cal_r2(arr1, arr2):
        """
        Calculate the r square.
        """
        print('Calculating R square...')
        if type(arr1) is not np.ndarray:
            arr1 = np.array(arr1)
        if type(arr2) is not np.ndarray:
            arr2 = np.array(arr2)
        if len(arr1) != len(arr2):
            raise Exception('Length of two arrays are different.')

        x_bar, y_bar = np.mean(arr1), np.mean(arr2)
        ssr, var_x, var_y = 0, 0, 0
        for i in range(0, len(arr1)):
            diff_xx_bar = arr1[i] - x_bar
            diff_yy_bar = arr2[i] - y_bar
            ssr += (diff_xx_bar * diff_yy_bar)
            var_x += diff_xx_bar ** 2
            var_y += diff_yy_bar ** 2
        sst = np.sqrt(var_x * var_y)
        return (ssr / sst) ** 2

    def cal_integrity(self):
        """
        Calculate the data integrity
        :return:
        """
        print('Calculating data integrity...')
        self.integrity = []
        station_number = []
        count = 0
        index = []
        for i in range(len(self.operate_data)):
            if sum(self.operate_data[i, :]) == -99999 * 29:
                self.integrity.append(0)
                station_number.append([-1])
                continue
            for j in range(29):
                if self.operate_data[i, j] != -99999:
                    count += 1
                    index.append(j)
            self.integrity.append(count)
            count = 0
            station_number.append(index)
            index = []

        plt.figure()
        plt.scatter(np.arange(len(self.integrity)), self.integrity, marker='.')
        tick = [i[:4] for i in self.operate_T if i[4:] == '/1/1 00:00']
        tick.append(str(int(tick[-1]) + 1))
        year_number = len(self.operate_data) // (365 * 24) + 1
        if year_number >= 30:
            plt.xticks(np.arange(0, 8760 * year_number, 8760 * 4), [j for i, j in enumerate(tick) if i % 4 == 0])
        elif 15 < year_number < 30:
            plt.xticks(np.arange(0, 8760 * year_number, 8760 * 2), [j for i, j in enumerate(tick) if i % 2 == 0])
        else:
            plt.xticks(np.arange(0, 8760 * year_number, 8760), [j for i, j in enumerate(tick)])
        plt.yticks(np.arange(0, 26, 2))
        plt.title('Data integrity')
        plt.xlabel('Time')
        plt.ylabel('Index of station')
        # plt.show()

    def cal_missing_length(self):
        """
        Calculate the length of continuous missing data.
        :return:
        """
        print('Calculating length of missing data...')
        xx = []
        yy = []
        length = []
        count = 0
        for j in range(len(self.operate_data[1, :])):
            length.append([])
            flag = 'valid'
            for i in range(len(self.operate_data)):
                xx.append(i)
                if self.operate_data[i, j] == -99999:
                    yy.append(j + 1)
                    if flag == 'valid':
                        count = 1
                        flag = 'invalid'
                    elif flag == 'invalid':
                        count += 1
                else:
                    yy.append(0)
                    if flag == 'invalid':
                        length[j].append(count)
                        flag = 'valid'
            if flag == 'invalid':
                length[j].append(count)
            length[j].sort(reverse=True)
        tmp = []
        for i in range(len(length)):
            tmp.append(length[i][0:10])
        print('Max length of each station:', tmp)
        plt.figure()
        plt.scatter(xx, yy, marker='_')
        plt.title('Data missing length')
        plt.xlabel('Time')
        plt.ylabel('Index of station')
        # plt.show()


def search_year(dt, t, start_year=1983, end_year=None):
    """
    Return start year 01/01 00:00 ~ end_year-1 12/31 23:00
    :param dt: data
    :param t: time
    :param start_year: int
    :param end_year: int or none if it is 2022
    :return: start year 01/01 00:00 ~ end_year-1 12/31 23:00
    """
    start_number, end_number = -1, -1
    s_flag, e_flag = True, True
    end_year = None if end_year == 2022 else end_year
    if end_year and (s_flag or e_flag):
        end_year += 1
        for i in range(len(dt)):
            if t[i][:4] == str(start_year) and s_flag:
                start_number = i
                s_flag = False
            elif t[i][:4] == str(end_year) and e_flag:
                end_number = i
                e_flag = False
    elif not end_year:
        end_number = len(dt)
        for i in range(len(dt)):
            if t[i][:4] == str(start_year) and s_flag:
                start_number = i
                s_flag = False
    return start_number, end_number - 1


def cal_valid_time(t_data, interval=72):
    """
    Find the missing data interval longer than interval. Return valid segmentation
    :param t_data: time series data with missing data marked as -99999
    :param interval: interval, hours
    :return: 2 lists, [start_1, ..., start_n], [end_1, ..., end_n]
    """
    state = 'invalid'
    state_count = 0
    start_t = []
    end_t = []
    for i, d in enumerate(t_data):
        if d != -99999:
            if state == 'invalid':
                state = 'valid'
                start_t.append(i)
            state_count = 0
        else:
            if state == 'invalid':
                continue
            else:
                if state_count >= interval:
                    state = 'invalid'
                    state_count = 0
                    end_t.append(i - interval - 1)
                else:
                    state_count += 1
    if len(start_t) > len(end_t):
        end_t.append(len(t_data) - 1)
    return start_t, end_t


class nc_reader:
    def __init__(self, nc_path):
        self.ds = nc.Dataset(nc_path)
        # self.names = {'O3': 'Dataset1', 'O2': 'Dataset2', 'NOx': 'Dataset3', 'PM2.5': 'Dataset4', 'PM10': 'Dataset5', 'SO2': 'Dataset6'}

        # print(self.ds)
        # print('16', ''.join([i.decode("utf-8") for i in ds['Output_Options_Description'][1, :]]))

    def read_value(self, name, start=None, end=None):
        """
        Read value from nc file
        :param name: the name of pollutant: refer to adms_fcstout_notes.md.
        :param start: start index. if None: 0.
        :param end: end index. if None: len(data).
        :param mode: if 0: without XY. Else: with XY
        :return: masked array(end-start+1) or masked array(end-start+1, 3) if mode.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self.ds.dimensions['datalen'])
        return self.ds[name][start:end]


def get_filelist(path, filter=None):
    """
    Get filename list of target file.
    :param path: data folder
    :param filter: the years wanted
    :return: filename list
    """
    filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if not filter:
                filelist.append(os.path.join(home, filename))
            else:
                if filename[:4] in filter:
                    filelist.append(os.path.join(home, filename))

    return filelist


if __name__ == "__main__":
    nc_path = '/Volumes/8T/AQMS/data/2022/202201/2022010112.2/2022-01-02_05_04_con.nc'
    nc_path = '/Volumes/8T/AQMS1/2022/202201/20220101/2022010100.nc'
    wrf_path = '/Volumes/8T/WRF/data/2021/202101/2021010312/wrfout_d01_2021-01-03_12:00:00'
    cmaq_path = '/Volumes/8T/CMAQ/1km/CCTM_V5g_ebi_cb05cl_ae5_aq_mpich2.ACONC.2021002'
    cmaq_ave_path = '/Volumes/8T/CMAQ/average/D1/NO2/000.nc'
    ds = nc_reader(cmaq_ave_path)

    # ds = nc_reader(wrf_path)
    # a = ds.read_value('LU_INDEX', start=0, end=19)
    # b = ds.ds['LU_INDEX'][:]
    pass
