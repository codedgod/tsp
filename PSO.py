import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
class particles_swarm_optimization(object): #粒子群计算
    def __init__(self,num_cities,num_particles,generation,name): #种群数量；
        self.num_cities=num_cities
        self.num_particles=num_particles
        self.generation=generation
        coordinates=self.data(name) #data
        self.distance_matrix=self.clac_distance(num_cities,coordinates)#距离；
        self.particles = np.array([np.random.permutation(num_cities) for _ in range(num_particles)])
        self.personal_best_positions = self.particles.copy() # 初始化个体最佳位置
        self.personal_best_values = np.array([self.tsp_objective_function(p, self.distance_matrix) for p in self.particles])#每个历史最优的距离。
        self.global_best_value = np.min(self.personal_best_values)
        self.global_best_position = self.particles[np.argmin(self.personal_best_values)]#最终的结果；
        self.global_best_value_lst=[] ##是最优值的list；
        """
        固定的参数；
        """
        self.tsp_best_path=self.global_best_position
        self.tsp_best_value=self.global_best_value
        self.tsp_best_value_lst=self.global_best_value_lst
    def get_ss(self,x_best, x_i, r):
        """
        计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
        :param x_best: pbest or gbest [1,31]
        :param x_i: 粒子当前的解 [1,31]
        :param r: 随机因子
        :return:
        """
        velocity_ss = []
        for i in range(len(x_i)): #
            if x_i[i] != x_best[i]: #与历史最优与全局最优不相等；
                j = np.where(x_i == x_best[i])[0][0] #从x_i中找到与x_best[i]相等的值；
                #print(j)
                so = (i, j, r)  #得到交换子；r:随机数；赋值元组。
                velocity_ss.append(so) #添加元组。
                x_i[i], x_i[j] = x_i[j], x_i[i]  # 执行交换操作
        return velocity_ss
    # 定义位置更新函数
    def do_ss(self,x_i, ss):
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
    def tsp_objective_function(self,tour, distance_matrix): #计算适应度。
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += distance_matrix[tour[i]][tour[i + 1]]
        # 回到起点
        total_distance += distance_matrix[tour[-1]][tour[0]]
        return total_distance
    def clac_distance(self,city_num,city):
            distance_matrix = np.zeros((city_num, city_num))#31*31的矩阵
            for i in range(city_num):
                for j in range(city_num):
                    if i == j:
                        continue
                    distance = np.sqrt((city[i,0] - city[j,0]) ** 2 + (city[i,1] - city[j,1]) ** 2)
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
        w,h=coord.shape
        coordinates=np.zeros((w,h))
        for i in range(w):#将str转化成float
            for j in range(h):
                coordinates[i, j] = float(coord[i, j])
        return coordinates
    def tsp(self):
        for _ in range(self.generation):
            for i in range(self.num_particles):
                # 更新粒子速度和位置
                r1, r2 = 0.7,0.8 #0-1之间的value;
                ss1 = self.get_ss(self.personal_best_positions[i], self.particles[i], r1) #[31,1],[31，1],常数；
                ss2 = self.get_ss(self.global_best_position, self.particles[i], r2) #[31，1],[31,1],常数；
                #print(ss1,ss2)
                ss = ss1 + ss2
                #print(ss)
                self.particles[i] = self.do_ss(self.particles[i], ss)
                # 更新个体最佳位置和适应值
                current_value = self.tsp_objective_function(self.particles[i], self.distance_matrix)
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.particles[i]
            min_index = np.argmin(self.personal_best_values)
            if self.personal_best_values[min_index] < self.global_best_value:
                self.global_best_value = self.personal_best_values[min_index]
                self.global_best_position = self.personal_best_positions[min_index]
            self.global_best_value_lst.append(self.global_best_value)
            self.tsp_best_path = self.global_best_position
            self.tsp_best_value = self.global_best_value
            self.tsp_best_value_lst = self.global_best_value_lst
            print("global_best_value=",self.global_best_value,"global_best_position=",self.global_best_position)
# num_cities,num_particles,generation=31,1000,1000
# PSO_tsp=particles_swarm_optimization(num_cities,num_particles,generation)
# PSO_tsp.tsp()
# plt.figure()
# plt.plot(range(len(PSO_tsp.global_best_value_lst)),PSO_tsp.global_best_value_lst)
# # 设置X轴标签
# plt.xlabel('迭代次数')
# # 设置Y轴标签
# plt.ylabel('路径的总长度')
# # 设置图表标题
# plt.title('PSO算法解决tsp问题')
# # 显示网格（可选）
# plt.grid(True)
# # 保存图形为PDF
# plt.savefig('pso.pdf')