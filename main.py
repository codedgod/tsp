from PSO import particles_swarm_optimization #粒子群算法
from GA import Genetic #遗传算法
from ACA import Ant_colony #蚁群算法
from GA_PSO import GA_pso #粒子群_遗传算法
from MOPSO import MOpso #多目标粒子群算法
from MOGA_PSO import MOGA_pso #多目标粒子群_遗传算法
"""
matplotlib:绘图；
"""
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


if __name__=='__main__':
    num_cities,num_particles,generation=31,500,20 ##城市的数量，种群的大小，迭代的次数；
    path_name="data/data"
    time_name="data/timable"
    ###类赋值；
    """
    单目标优化
    """
    pso_tsp=particles_swarm_optimization(num_cities,num_particles,generation,path_name) #粒子群算法0
    ga_tsp=Genetic(num_cities,num_particles,generation,path_name) #遗传算法1
    aca_tsp=Ant_colony(num_cities,num_particles,generation,path_name) #蚁群算法2
    ga_pso_tsp=GA_pso(num_cities,num_particles,generation,path_name) #粒子群_遗传算法3
    """
    多目标优化
    """
    mopso=MOpso(num_cities,num_particles,generation,path_name,time_name) #多目标粒子群算法4
    moga_pso_tsp=MOGA_pso(num_cities,num_particles,generation,path_name,time_name) #多目标粒子群_遗传算法5

    class_list=['pso_tsp','ga_tsp','aca_tsp','ga_pso_tsp','mopso','moga_pso_tsp'] #名字集
    ### 通用类；
    tsp_class=aca_tsp #(替换掉它，就可以运行不同的数)
    number=2
    ### 开始解决tsp问题
    tsp_class.tsp()
    """
    单目标优化：tsp的最短距离：
    tsp_best_path:最优的路径
    tsp_best_value:最优路径的值
    tsp_best_value_lst:遍历过程中最优的值的list
    """
    """
        多目标优化：tsp的最短距离与最短时间的权衡；
        gb_position:位置；
        gb_path:最短距离；
        gb_time:最短时间
        gb_path_lst:最短距离的list；
        gb_time_lst:最短时间的List；
        """
    if number<=3:
        #单目标；
        plt.figure()
        plt.plot(range(len(tsp_class.tsp_best_value_lst)),tsp_class.tsp_best_value_lst)
        # 设置X轴标签
        plt.xlabel('迭代次数')
        # 设置Y轴标签
        plt.ylabel('路径的总长度')
        # 设置图表标题
        plt.title(class_list[number]+'单目标优化的tsp问题')
        # 显示网格（可选）
        plt.grid(True)
    else:
        plt.figure()
        plt.plot(range(len(tsp_class.gb_path_lst)), tsp_class.gb_path_lst)
        plt.plot(range(len(tsp_class.gb_time_lst)), tsp_class.gb_time_lst)
        # 设置X轴标签
        plt.xlabel('迭代次数')
        # 设置Y轴标签
        plt.ylabel('路径的总长度/时间')
        # 设置图表标题
        plt.title(class_list[number]+'多目标优化的tsp问题')
        # 显示网格（可选）
        plt.grid(True)
    ####保存图片为pdf格式
    plt.savefig('pdf/'+class_list[number]+'1.pdf')







