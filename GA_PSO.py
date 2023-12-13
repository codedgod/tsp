import matplotlib.pyplot as plt
import numpy as np
from math import floor
import pandas as pd
import random
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


class GA_pso(object):  # 粒子群计算
    def __init__(self, num_cities, num_particles, generations,name, select_prob=0.8, cross_prob=0.80):  # 种群数量；
        self.generations = generations
        self.num_cities = num_cities
        self.num = num_cities  # 城市数量；
        self.num_particles = num_particles
        self.size_pop = num_particles
        ## 两个数据包
        coordinates = self.data(name)  # path_data
        self.distance_matrix = self.clac_distance(num_cities, coordinates)  # 距离；
        self.particles = np.array([np.random.permutation(num_cities) for _ in range(num_particles)])
        self.personal_best_positions = self.particles.copy()  # 初始化个体最佳位置
        self.personal_best_values = np.array(
            [self.tsp_objective_function(p, self.distance_matrix) for p in self.particles])  # 每个历史最优的距离。
        self.global_best_position = self.particles[np.argmin(self.personal_best_values)]  # 全局最优的粒子；
        self.global_best_value = np.min(self.personal_best_values)
        self.global_best_value_lst = []
        ##遗传变异
        self.chrom = np.array([0] * self.size_pop * self.num).reshape(self.size_pop,
                                                                      self.num)  # 父 print(chrom.shape)(200, 14)
        self.fitness = np.zeros(self.size_pop)
        # 选择概率
        self.select_prob = select_prob
        # 通过选择概率确定子代的选择个数
        self.select_num = max(floor(self.size_pop * self.select_prob + 0.5), 2)
        self.cross_prob = cross_prob
        self.sub_sel = np.array([0] * int(self.select_num) * self.num).reshape(self.select_num, self.num)  # 子 (160, 14)
        self.place=self.particles
        """
        固定的参数；
        """
        self.tsp_best_path=self.global_best_position
        self.tsp_best_value=self.global_best_value
        self.tsp_best_value_lst=self.global_best_value_lst
    def get_ss(self, x_best, x_i, r):
        """
        计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
        :param x_best: pbest or gbest [1,31]
        :param x_i: 粒子当前的解 [1,31]
        :param r: 随机因子
        :return:
        """
        velocity_ss = []
        for i in range(len(x_i)):  #
            if x_i[i] != x_best[i]:
                j = np.where(x_i == x_best[i])[0][0]  # 找到与x_best[i]相同的x_i
                so = (i, j, r)  # 得到交换子；r:随机数；赋值元组。
                velocity_ss.append(so)  # 添加元组。
                x_i[i], x_i[j] = x_i[j], x_i[i]  # 执行交换操作
        return velocity_ss

    # 定义位置更新函数
    def do_ss(self, x_i, ss):
        """
        执行交换操作
        :param x_i:
        :param ss: 由交换子组成的交换序列
        :return:
        """
        for i, j, r in ss:
            rand = np.random.random()
            if rand <= r:
                x_i[i], x_i[j] = x_i[j], x_i[i]
        return x_i

    # TSP问题的目标函数
    def tsp_objective_function(self, tour, distance_matrix):  # 计算适应度。
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        # 回到起点
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance

    def clac_distance(self, city_num, city):
        distance_matrix = np.zeros((city_num, city_num))  # 31*31的矩阵
        for i in range(city_num):
            for j in range(city_num):
                if i == j:
                    continue
                distance = np.sqrt((city[i, 0] - city[j, 0]) ** 2 + (city[i, 1] - city[j, 1]) ** 2)
                distance_matrix[i][j] = distance
        return distance_matrix

    def data(self,name):
        coord = []
        name=name+".txt"
        with open(name, "r") as lines:
            lines = lines.readlines()
        for line in lines:
            xy = line.split()
            coord.append(xy)
        coord = np.array(coord)
        w, h = coord.shape
        coordinates = np.zeros((w, h))
        for i in range(w):  # 将str转化成float
            for j in range(h):
                coordinates[i, j] = float(coord[i, j])
        return coordinates

    ####################遗传变异
    def intercross(self, ind_a, ind_b):  # ind_a，ind_b 父代染色体 shape=(1,14) 14=14个城市
        r1 = np.random.randint(self.num)  # 在num内随机生成一个整数 ，num=14.即随机生成一个小于14的数
        r2 = np.random.randint(self.num)
        while r2 == r1:  # 如果r1==r2
            r2 = np.random.randint(self.num)  # r2重新生成
        left, right = min(r1, r2), max(r1, r2)  # left 为r1,r2小值 ，r2为大值
        ind_a1 = ind_a.copy()  # 父亲
        ind_b1 = ind_b.copy()  # 母亲
        for i in range(left, right + 1):
            ind_a2 = ind_a.copy()
            ind_b2 = ind_b.copy()
            ind_a[i] = ind_b1[i]  # 交叉 （即ind_a  （1,14） 中有个元素 和ind_b互换
            ind_b[i] = ind_a1[i]
            x = np.argwhere(ind_a == ind_a[i])
            y = np.argwhere(ind_b == ind_b[i])

            """
                   下面的代码意思是 假如 两个父辈的染色体编码为【1234】，【4321】 
                   交叉后为【1334】，【4221】
                   交叉后的结果是不满足条件的，重复个数为2个
                   需要修改为【1234】【4321】（即修改会来
            """
            if len(x) == 2:
                ind_a[x[x != i]] = ind_a2[i]  # 查找ind_a 中元素=- ind_a[i] 的索引
            if len(y) == 2:
                ind_b[y[y != i]] = ind_b2[i]
        return ind_a, ind_b

    # 交叉
    def cross_sub(self):
        if self.select_num % 2 == 0:  # select_num160
            num = range(0, int(self.select_num), 2)
        else:
            num = range(0, int(self.select_num - 1), 2)
        for i in num:
            if self.cross_prob >= np.random.rand():
                self.sub_sel[i, :], self.sub_sel[i + 1, :] = self.intercross(self.sub_sel[i, :], self.sub_sel[i + 1, :])

    # 变异模块  在变异概率的控制下，对单个染色体随机交换两个点的位置。
    def mutation_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            if np.random.rand() <= self.cross_prob:  # 如果随机数小于变异概率
                r1 = np.random.randint(self.num)  # 随机生成小于num==可设置 的数
                r2 = np.random.randint(self.num)
                while r2 == r1:  # 如果相同
                    r2 = np.random.randint(self.num)  # r2再生成一次
                self.sub_sel[i, [r1, r2]] = self.sub_sel[i, [r2, r1]]  # 随机交换两个点的位置。

    # 进化逆转  将选择的染色体随机选择两个位置r1:r2 ，将 r1:r2 的元素翻转为 r2:r1 ，如果翻转后的适应度更高，则替换原染色体，否则不变
    def reverse_sub(self):
        for i in range(int(self.select_num)):  # 遍历每一个 选择的子代
            r1 = np.random.randint(self.num)  # 随机生成小于num==14 的数
            r2 = np.random.randint(self.num)
            while r2 == r1:  # 如果相同
                r2 = np.random.randint(self.num)  # r2再生成一次
            left, right = min(r1, r2), max(r1, r2)  # left取r1 r2中小值，r2取大值
            sel = self.sub_sel[i, :].copy()  # sel 为父辈染色体 shape=（1,14）

            sel[left:right + 1] = self.sub_sel[i, left:right + 1][::-1]  # 将染色体中(r1:r2)片段 翻转为（r2:r1)
            if self.comp_fit(sel) < self.comp_fit(self.sub_sel[i, :]):  # 如果翻转后的适应度小于原染色体，则不变
                self.sub_sel[i, :] = sel

    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.distance_matrix[one_path[i], one_path[i + 1]]  # matrix_distance n*n, 第[i,j]个元素表示城市i到j距离
        res += self.distance_matrix[one_path[-1], one_path[0]]  # 最后一个城市 到起点距离
        return res

    def reins(self):
        index = np.argsort(self.fitness)[::-1]  # 替换最差的（倒序）
        self.chrom[index[:self.select_num], :] = self.sub_sel

    def select_sub(self):
        fit = 1. / (self.fitness)  # 适应度函数
        cumsum_fit = np.cumsum(fit)  # 累积求和   a = np.array([1,2,3]) b = np.cumsum(a) b=1 3 6
        pick = cumsum_fit[-1] / self.select_num * (
                np.random.rand() + np.array(range(int(self.select_num))))  # select_num  为子代选择个数 160
        i, j = 0, 0
        index = []
        while i < self.size_pop and j < self.select_num:
            if cumsum_fit[i] >= pick[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.sub_sel = self.chrom[index, :]  # chrom 父

    def tsp(self):
        for _ in range(self.generations):
            self.place=self.particles
            for i in range(self.num_particles):
                # 更新粒子速度和位置
                r1, r2 = 0.7, 0.8  # 0-1之间的value;
                ss1 = self.get_ss(self.personal_best_positions[i], self.place[i], r1)  # [31,1],[31，1],常数；
                ss2 = self.get_ss(self.global_best_position, self.place[i], r2)  # [31，1],[31,1],常数；
                ss = ss1 + ss2
                self.place[i] = self.do_ss(self.place[i], ss)
                # 更新个体最佳位置和适应值
                current_value = self.tsp_objective_function(self.place[i], self.distance_matrix)
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.place[i]
            # min_index = np.argmin(self.personal_best_values)
            # if self.personal_best_values[min_index] < self.global_best_value:
            #     self.global_best_value = self.personal_best_values[min_index]
            #     self.global_best_position = self.personal_best_positions[min_index]
            # self.global_best_value_lst.append(self.global_best_value)
            ##########################遗传与变异；
            self.chrom = self.particles  # 将粒子群传递给父代；
            for j in range(self.size_pop):
                self.fitness[j] = self.comp_fit(self.chrom[j, :])
            # print(self.chrom)
            self.select_sub()  # 选择子代
            # 交叉操作
            self.cross_sub()  # 交叉
            # 变异操作
            self.mutation_sub()  # 变异
            # return self.sub_sel
            self.reverse_sub()  # 净化逆转
            self.reins()  # 插入父代
            self.particles = self.chrom  # 变异好了，传回去
            for i in range(self.num_particles):
                current_value = self.tsp_objective_function(self.particles[i], self.distance_matrix)
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.particles[i]
            min_index = np.argmin(self.personal_best_values)
            if self.personal_best_values[min_index] < self.global_best_value:
                self.global_best_value = self.personal_best_values[min_index]
                self.global_best_position = self.personal_best_positions[min_index]
            self.global_best_value_lst.append(self.global_best_value)
            print("pso_value=", self.global_best_value)
            print("pso_path=", self.global_best_position)
            self.tsp_best_path = self.global_best_position
            self.tsp_best_value = self.global_best_value
            self.tsp_best_value_lst = self.global_best_value_lst
# num_cities, num_particles, generations,name = 31, 1000, 1000,"data/data"
# ga_pso_tsp = GA_pso(num_cities, num_particles, generations,name)  # num_cities,num_particles,generations
# ga_pso_tsp.tsp()
# plt.figure()
# plt.plot(range(len(ga_pso_tsp.global_best_value_lst)), ga_pso_tsp.global_best_value_lst)
# # 设置X轴标签
# plt.xlabel('迭代次数')
# # 设置Y轴标签
# plt.ylabel('路径的总长度')
# # 设置图表标题
# plt.title('ga_pso算法解决tsp问题')
# # 显示网格（可选）
# plt.grid(True)
# # 保存图形为PDF
# plt.savefig('ga_pso.pdf')
# num_cities, num_particles, generations,name = 31, 1000, 1000,"data/data"
# ga_pso_tsp = GA_pso(num_cities, num_particles, generations,name)  # num_cities,num_particles,generations
# ga_pso_tsp.tsp()
# plt.figure()
# plt.plot(range(len(ga_pso_tsp.global_best_value_lst)), ga_pso_tsp.global_best_value_lst)
# # 设置X轴标签
# plt.xlabel('迭代次数')
# # 设置Y轴标签
# plt.ylabel('路径的总长度')
# # 设置图表标题
# plt.title('ga_pso算法解决tsp问题')
# # 显示网格（可选）
# plt.grid(True)
# # 保存图形为PDF
# plt.savefig('ga_pso.pdf')