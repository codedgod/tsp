import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
class MOpso(object): #粒子群计算
    def __init__(self,num_cities,num_particles,generation,name1,name2): #种群数量；
        self.generation=generation #迭代次数；
        self.num_cities=num_cities
        self.num_particles=num_particles
        coordinates=self.data(name1) #data
        self.distance_matrix=self.clac_distance(num_cities,coordinates)#距离；
        self.particles = np.array([np.random.permutation(num_cities) for _ in range(num_particles)])
        self.timetxt=self.timable(name2) #[31,31]
        ##历史位置
        self.pbpath_positions = self.particles.copy() # 初始化个体最佳位置
        self.pbpath_values = np.array([self.tsp_objective_function(p, self.distance_matrix) for p in self.particles])#每个历史最优的距离。
        ##历史时间
        self.pbtime_positions = self.particles.copy() # 初始化个体最佳位置
        self.pbtime_values = np.array([self.tsp_objective_function(p, self.timetxt) for p in self.particles])#每个历史最优的距离。
        ##历史最佳位置与时间；
        self.pb_positions = self.particles.copy() # 初始化个体最佳位置
        self.pb_path = np.array([self.tsp_objective_function(p, self.distance_matrix) for p in self.particles])#每个distance历史最优的距离。
        self.pb_time = np.array([self.tsp_objective_function(p, self.timetxt) for p in self.particles])#每个time历史最优的距离。
        ##全局的距离。
        self.gb_position = self.particles[0] #全局最优的粒子；
        self.gb_time = np.math.pow(10,10)
        self.gb_path = np.math.pow(10,10)
        self.gb_time_lst=[]
        self.gb_path_lst=[]
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
            if x_i[i] != x_best[i]:
                j = np.where(x_i == x_best[i])[0][0] #找到与x_best[i]相同的x_i
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
    def tsp_objective_function(self,tour,distance_matrix): #计算适应度。
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
    def timable(self,name):
        name=name+".txt"
        df = pd.read_csv(name, sep="\s+", header=None)
        matrix = df.values
        matrix = np.array(matrix, dtype=float)
        return matrix
    def grid(self,grid_point,n,x_low,x_upper,y_low,y_upper):
        #划分的格子大小；
        x_grid=abs(x_low-x_upper)/n
        y_grid=abs(y_low-y_upper)/n
        #print(n,x_low,y_low,x_upper,y_low,y_upper) #格子大小；
        #print(x_grid,y_grid)
        #对点进行遍历；
        grid_list=np.zeros((9,len(grid_point))) #存储的list
        grid_num=np.zeros(9).astype(int) #存储的数量
        #print("grid_num=",grid_num)
        for j,point in enumerate(grid_point):
            flag=0
            if point[0]<x_low:
                point[0]=x_low+x_grid/2
            if point[1]<y_low:
                point[1]=y_low+y_grid/2
            for i in range(9):
                if (point[0]>=(x_low+x_grid*(i%n)) and point[0]<=(x_low+x_grid*((i%n)+1))) and ((point[1]>=(y_low+y_grid*(n-(i//n)-1))) and point[1]<=(y_low+y_grid*(n-(i//n)))):
                    grid_list[i][grid_num[i]]=grid_num[i] #将所在位置的一次储存；
                    grid_num[i]+=1 #该区域内的数量+1；
                    flag=1
            if flag==0:
                print("point=",point)
        probability=np.array(grid_num)/sum(grid_num)
        return grid_list,grid_num,probability
    def select(self,probabilities):
        cumulative_probabilities = [0] * len(probabilities)
        cumulative_probabilities[0] = probabilities[0]
        for i in range(1, len(probabilities)): #累计概率；
            cumulative_probabilities[i] = cumulative_probabilities[i-1] + probabilities[i]
        # 生成一个0到1之间的随机数
        random_number = random.uniform(0, 1)
        # 使用轮盘赌方法选择一个网格
        selected_grid = -1
        for i in range(len(cumulative_probabilities)):
            if random_number <= cumulative_probabilities[i]:
                selected_grid = i
                break
        return selected_grid
    def tsp(self):
        for j in range(self.generation):
            for i in range(self.num_particles):
                # 更新粒子速度和位置
                r1, r2 = 0.7,0.8 #0-1之间的value;
                ss1 = self.get_ss(self.pb_positions[i], self.particles[i], r1) #[31,1],[31，1],常数；
                ss2 = self.get_ss(self.gb_position, self.particles[i], r2) #[31，1],[31,1],常数；
                ss = ss1 + ss2
                self.particles[i] = self.do_ss(self.particles[i], ss)
                # 更新个体最佳位置和适应值(多目标优化)1.path的距离；2.time的时间；
                current_path_value= self.tsp_objective_function(self.particles[i],self.distance_matrix)
                current_time_value=self.tsp_objective_function(self.particles[i],self.timetxt)
                if current_time_value<self.pbtime_values[i]: #对时间;
                    self.pbtime_values[i]=current_time_value
                    self.pbtime_positions[i]=self.particles[i]
                if current_path_value < self.pbpath_values[i]: #对路程；
                    self.pbpath_values[i] =current_path_value
                    self.pbpath_positions[i] = self.particles[i]
                #p_best求解；随机赋值；
                """
                self.pb_path：距离；
                self.pb_time：时间;
                self.pb_positions:路径；
                """
                if current_time_value<self.pbtime_values[i] and current_path_value < self.pbpath_values[i]:
                    self.pb_path[i]=self.pbpath_values[i]
                    self.pb_time[i]=self.pbtime_values[i]
                    self.pb_positions[i]=self.pbtime_positions[i]
                else:
                    #if (current_time_value+current_path_value)<(self.pbtime_values[i]+self.pbpath_values[i]):
                    randn=np.random.random()
                    if randn>0.5: #随机赋值；
                        self.pb_path[i]=self.pbpath_values[i] #[num_particles,1]
                        self.pb_time[i]=self.pbtime_values[i] #[num_particles,1]
                        self.pb_positions[i]=self.pbtime_positions[i] #[num_particles,31]
            #g_best求解；网格法；
            #min_index = np.argmin(self.personal_best_values)
            ## 1.边界；
            min_path,max_path=min(self.pb_path),max(self.pb_path)
            min_time,max_time=min(self.pb_time),max(self.pb_time)
            f1_b,f1_u,f2_b,f2_u=min_path+0.5,max_path+0.5,min_time+0.5,max_time+0.5
            #print(f1_b,f1_u,f2_b,f2_u)
            ## 2.划分格子；n=3。初始化点；
            grid_point=[[x,y] for x,y in zip(self.pb_path,self.pb_time)]
            #print(np.array(grid_point)[:,0])
            #print(np.array(grid_point)[:,1])
            ## 3.网格中存储的个体总数## 4.计算每个格子的概率；
            """
            grid_num：每个格子里面有多少点；
            grid_list：每个格子里面的点的位置；
            """
            #print(len(grid_point))
            grid_list,grid_num,probability=self.grid(grid_point,3,f1_b,f1_u,f2_b,f2_u)
            #print(grid_num,sum(grid_num))
            ## 5.赌盘选择；select_num-赌盘选择的值；
            select_num=self.select(probability)
            ## 6.在select_num的选择集里面随机选择一个；
            grid_set=grid_list[select_num][0:grid_num[select_num]] #最好的集合；
            grid_set=grid_set.astype(int)
            #print(grid_set)
            grid_best=np.random.choice(grid_set) #最好的序列；可以读取position;
            ## 7.读取最好；value;
            a,b=grid_point[grid_best] #[gb_path,gb_time]
            if (a < self.gb_path or b < self.gb_time) and (a+b<self.gb_path+self.gb_time):
                self.gb_path,self.gb_time=a,b
                self.gb_position=self.pb_positions[grid_best] #[1,31]
            self.gb_path_lst.append(self.gb_path)
            self.gb_time_lst.append(self.gb_time)
            print("gb_path=",self.gb_path,"gb_time=",self.gb_time)
            print("gb_position=",self.gb_position)

# num_cities,num_particles,generation=31,1000,1000
# mopso_tsp=MOpso(num_cities,num_particles,generation)
# mopso_tsp.tsp()
# plt.figure()
# plt.plot(range(len(mopso_tsp.gb_path_lst)), mopso_tsp.gb_path_lst)
# plt.plot(range(len(mopso_tsp.gb_time_lst)), mopso_tsp.gb_time_lst)
# # 设置X轴标签
# plt.xlabel('迭代次数')
# # 设置Y轴标签
# plt.ylabel('路径的总长度')
# # 设置图表标题
# plt.title('mopso算法解决tsp问题')
# # 显示网格（可选）
# plt.grid(True)
# # 保存图形为PDF
# plt.savefig('mopso.pdf')
