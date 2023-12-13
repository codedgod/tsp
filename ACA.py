# -*- coding: utf-8 -*-
"""
蚁群算法求解TSP问题
随机在（0,101）二维平面生成20个点
距离最小化
"""
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
class Ant_colony(object):
    def __init__(self,city_num,antNum,generation,name):
        self.CityNum = city_num #城市数量
        #ACO参数
        self.antNum =antNum #蚂蚁数量
        self.generation=generation #迭代次数
        self.alpha = 2 #信息素重要程度因子
        self.beta = 1.5 #启发函数重要程度因子
        self.rho = 0.2 #信息素挥发因子
        self.Q = 100.0 #常数
        self.best_fit = math.pow(10,10) #较大的初始值，存储最优解
        self.best_line = []#存储最优路径
        self.bestfit_lst=[]
        self.dataset,self.CityCoordinates=self.data(name) #城市的坐标
        self.dis_matrix=self.clac_distance(self.CityNum,self.dataset) #城市之间的距离；
        self.pheromone = pd.DataFrame(data=self.Q,columns=range(len(self.CityCoordinates)),index=range(len(self.CityCoordinates)))
        #data初始值为100.0
        self.trans_p = self.calTrans_p(self.pheromone,self.alpha,self.beta,self.dis_matrix,self.Q) #计算初始转移概率矩阵；
        """
        固定的参数；
        """
        self.tsp_best_path = self.best_line #存储最优路径
        self.tsp_best_value = self.best_fit #最优解；
        self.tsp_best_value_lst = self.bestfit_lst #最优解的list；
    def tsp_objective_function(self,tour, distance_matrix): #计算适应度。
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        # 回到起点
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance

    #计算路径距离，即评价函数
    def calFitness(self,line,dis_matrix): #antCityList[i]-路线,dis_matrix-矩阵的大小；
        dis_sum = 0
        dis = 0
        for i in range(len(line)-1):
            dis = dis_matrix.loc[line[i],line[i+1]]#计算距离
            dis_sum = dis_sum+dis
        dis = dis_matrix.loc[line[-1],line[0]] #闭环的距离；
        dis_sum = dis_sum+dis
        return dis_sum
    def select(self,antCityList,antCityTabu,trans_p):
        '''
        轮盘赌选择，根据出发城市选出途径所有城市
        输入：trans_p-概率矩阵;antCityTabu-城市禁忌表，即未经过城市;
        输出：完整城市路径-antCityList;
        '''
        while len(antCityTabu) > 0: #迭代；
            if len(antCityTabu) == 1:
                nextCity = antCityTabu[0] #下一城市就是第0个的城市；
            else:
                fitness = []
                for i in antCityTabu:#逐个添加[开头，剩下的城市]的城市转移概率
                    #antCityTabu-剩下的城市；
                    fitness.append(trans_p.loc[antCityList[-1],i])#antCityList[-1]-最后一个城市；
                    #取出antCityTabu对应的城市转移概率
                sumFitness = sum(fitness) #城市转移lv的概率的总和；
                randNum = random.uniform(0, sumFitness)
                #随机生成一个在[0到sumFitness)之间的均匀分布的随机浮点数。
                accumulator = 0.0  #累积量；
                for i, ele in enumerate(fitness):
                    accumulator += ele
                    if accumulator >= randNum:
                        nextCity = antCityTabu[i]
                        break
            antCityList.append(nextCity)
            antCityTabu.remove(nextCity)

        return antCityList #蚁群跑的链式线状；


    def calTrans_p(self,pheromone,alpha,beta,dis_matrix,Q):  #转移概率；
        #Q=100,alpha=2,beta=1;
        #计算信息素=历史累计信息素-信息素挥发量+蚂蚁行走释放量
        '''
        根据信息素计算转移概率
        输入：pheromone-当前信息素；alpha-信息素重要程度因子；beta-启发函数重要程度因子；dis_matrix-城市间距离矩阵；Q-信息素常量；
        输出：当前信息素+增量-transProb
        '''
        transProb = Q/dis_matrix #初始化transProb存储转移概率，同时计算增量
        for i in range(len(transProb)):
            for j in range(len(transProb)):
                transProb.iloc[i,j] = pow(pheromone.iloc[i,j], alpha) * pow(transProb.iloc[i,j], beta)

        return transProb


    def updatePheromone(self,pheromone,fit,antCity,rho,Q):
        '''
        更新信息素，蚁周算法
        输入：pheromone-当前信息素；fit-路径长度；antCity-路径；rho-ρ信息素挥发因子；Q-信息素常量
        输出：更新后信息素-pheromone
        '''
        for i in range(len(antCity)-1):
            pheromone.iloc[antCity[i],antCity[i+1]] += Q/fit
        pheromone.iloc[antCity[-1],antCity[0]] += Q/fit

        return pheromone
    def intialize(self,CityCoordinates,antNum):
        """
        初始化，为蚂蚁分配初始城市
        输入：CityCoordinates-城市坐标;antNum-蚂蚁数量
        输出：cityList-蚂蚁初始城市列表，记录蚂蚁初始城市;cityTabu-蚂蚁城市禁忌列表，记录蚂蚁未经过城市
        """
        cityList,cityTabu = [None]*antNum,[None]*antNum#初始化
        for i in range(len(cityList)):
            city = random.randint(0, len(CityCoordinates)-1)#初始城市，默认城市序号为0开始计算
            cityList[i] = [city]
            cityTabu[i] = list(range(len(CityCoordinates)))
            cityTabu[i].remove(city)
        return cityList,cityTabu #初始城市列表[50,1];[50,num-1];
    # def clac_distance(self,city_num,X, Y):
    #     """
    #     计算两个城市之间的欧氏距离，二范数
    #     :param X: 城市X的坐标.np.array数组
    #     :param Y: 城市Y的坐标.np.array数组
    #     :return:
    #     """
    #
    #     distance_matrix = np.zeros((city_num, city_num))#31*31的矩阵
    #     for i in range(city_num):
    #         for j in range(city_num):
    #             if i == j:
    #                 continue
    #             distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
    #             distance_matrix[i][j] = distance
    #     return distance_matrix
    def data(self,name):
        coord = []
        name=name+".txt"
        with open(name, "r") as lines:
            lines = lines.readlines()
        for line in lines:
            xy = line.split()
            coord.append(xy)
        coord = np.array(coord)
        w,h=coord.shape
        coordinates=np.zeros((w,h))
        CityCoordinates=[]
        for i in range(w):#将str转化成float(31,2)
            CityCoordinates.append((float(coord[i, 0]),float(coord[i, 1])))
            for j in range(h):
                coordinates[i, j] = float(coord[i, j])
        return coordinates,CityCoordinates
    def clac_distance(self,city_num,city):
        distance_matrix = np.zeros((city_num, city_num))#31*31的矩阵
        for i in range(city_num):
            for j in range(city_num):
                if i == j:
                    continue
                distance = np.sqrt((city[i,0] - city[j,0]) ** 2 + (city[i,1] - city[j,1]) ** 2)
                distance_matrix[i][j] = distance
        return pd.DataFrame(distance_matrix)
    def tsp(self):
        '''
        每一代更新一次环境因素导致的信息素减少，每一代中的每一个蚂蚁完成路径后，都进行信息素增量更新（采用蚁周模型）和转移概率更新；
        每一代开始都先初始化蚂蚁出发城市；
        '''
        for i in range(self.generation):
            antCityList,antCityTabu = self.intialize(self.CityCoordinates,self.antNum)
            #初始化城市;CityCoordinates-城市的位置；antNum-蚂蚁数量；antCityList-初始城市位置,antCityTabu-在1-31的列表里面移除初始城市；
            fitList = [None]*self.antNum #适应值列表(每个尝试的距离)；[50,31];
            for i in range(self.antNum): #根据转移概率选择后续途径城市，并计算适应值；antNum-蚁群大小；fitList-[50,1]:50个蚂蚁每个蚂蚁爬的长度
                antCityList[i] = self.select(antCityList[i],antCityTabu[i],self.trans_p)
                fitList[i] = self.tsp_objective_function(antCityList[i],self.dis_matrix)#适应度，即路径长度
                self.pheromone = self.updatePheromone(self.pheromone,fitList[i],antCityList[i],self.rho,self.Q)#更新当前蚂蚁信息素增量
                self.trans_p = self.calTrans_p(self.pheromone,self.alpha,self.beta,self.dis_matrix,self.Q)
            if self.best_fit >= min(fitList):
                self.best_fit = min(fitList)
                self.best_line = antCityList[fitList.index(min(fitList))] #[50,31]的矩阵；
            self.bestfit_lst.append(self.best_fit)
            self.pheromone = self.pheromone*(1-self.rho)
            self.tsp_best_path = self.best_line  # 存储最优路径
            self.tsp_best_value = self.best_fit  # 最优解；
            self.tsp_best_value_lst = self.bestfit_lst  # 最优解的list；
            print("ACO_value=",self.best_fit,"ACO_path=",self.best_line)
# num_cities,num_particles,generation=31,1000,30
# ACA_tsp=Ant_colony(num_cities,num_particles,generation)
# ACA_tsp.tsp()
# plt.figure()
# plt.plot(range(len(ACA_tsp.bestfit_lst)), ACA_tsp.bestfit_lst)
# # 设置X轴标签
# plt.xlabel('迭代次数')
# # 设置Y轴标签
# plt.ylabel('路径的总长度')
# # 设置图表标题
# plt.title('ACA算法解决tsp问题')
# # 显示网格（可选）
# plt.grid(True)
# # 保存图形为PDF
# plt.savefig('aca.pdf')
