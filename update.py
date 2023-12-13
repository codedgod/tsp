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
        self.w=0.8
        self.c1=0.7
        self.c2=0.6
        self.velocity=np.zeros(num_cities)
        """
        固定的参数；
        """
        self.tsp_best_path=self.global_best_position
        self.tsp_best_value=self.global_best_value
        self.tsp_best_value_lst=self.global_best_value_lst
    def update_particle(self,position,best_position,global_best_position):
        inertia = self.w * self.velocity
        cognitive = self.c1 * np.random.rand(self.num_cities) * (best_position - best_position)
        social = self.c2 * np.random.rand(self.num_cities) * (global_best_position - position)
        self.velocity = inertia + cognitive + social
        position = np.argsort(position + self.velocity)
        return position
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
                #print("before=",self.particles[i])
                self.particles[i]=self.update_particle(self.particles[i],self.personal_best_positions[i],self.global_best_position)
                #print("pass=",self.particles[i])
                # 更新个体最佳位置和适应值
                current_value = self.tsp_objective_function(self.particles[i], self.distance_matrix)
                print("now_value=",current_value,"pass_value",self.personal_best_values[i])
                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.particles[i]
            print(self.personal_best_values)
            min_index = np.argmin(self.personal_best_values)
            if self.personal_best_values[min_index] < self.global_best_value:
                self.global_best_value = self.personal_best_values[min_index]
                self.global_best_position = self.personal_best_positions[min_index]
            self.global_best_value_lst.append(self.global_best_value)
            self.tsp_best_path = self.global_best_position
            self.tsp_best_value = self.global_best_value
            self.tsp_best_value_lst = self.global_best_value_lst
            print("global_best_value=",self.global_best_value,"global_best_position=",self.global_best_position)
num_cities,num_particles,generation=31,5,100
name="data/data"
PSO_tsp=particles_swarm_optimization(num_cities,num_particles,generation,name)
PSO_tsp.tsp()
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