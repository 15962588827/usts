import numpy as np
import  pandas as pd
import torch.nn

MAX_STEPS = 5000
class single_agent_Env(object):
    def __init__(self, Hour,pre_demand,grid):
        self.real_load =grid
        # self.real_load = self.real_load.reshape(-1,1)
        self.demand_load = pre_demand
        # self.demand_load = self.demand_load.reshape(-1,1)
        self.dt = 15 / 60  # 15分钟
        self.Hour = Hour  # 当前时刻
        # self.PV_load = PV
        # 将奖励归一化
        self.alpha = 0.9  # 衰减系数，可以根据需要调整
        self.mean_reward1 = 0
        self.mean_reward2 = 0
        self.mean_reward3 = 0
        """
        energy storage device (ESD) 电池设备参数
        """
        self.P_B_charge = 3   # 充电功率
        self.P_B_discharge = 3  # 放电功率
        self.n_b = 0.95  # 单向效率 (放电/充电)
        self.soc_min = 0  # 最小可用容量/kWh
        self.soc_max = 10  # 最大可用容量/kWh

        self.soc = np.zeros(len(self.demand_load))  # 一个二维数组，表示电池的状态（State of Charge）。第一行被初始化为0。
        self.soc[0] = 0
        self.price = np.zeros(len(self.demand_load))
        self.sell_money = 0
        self.buy_money = 0
        self.soc_min = 0  # 最小可用容量/kWh
        self.soc_max = 10  # 最大可用容量/kWh

        self.part_pv = np.zeros(len(self.demand_load))
        self.part_pv[0] = 0

    # 充电的时候 a_P_B > 0 放电 a_P_B < 0
    def step(self, t, battery_action):
        bool_B, a_P_B = battery_action[0], battery_action[1]
        """
        ESS状态选择
        """
        # self.part_pv[t] = 0

        part_load = 0 #电网给电池冲的电量
        charge_load = 0
        PV = 0
        # 如果光伏有盈余
        # if -0.75 <= self.demand_load[t] <= 0 :
        #     self.soc[t+1] = self.soc[t] + abs(self.demand_load[t]) * self.n_b
        #
        # elif self.demand_load[t] < -0.75:
        #     self.soc[t+1] = self.soc[t] + 0.7125
            #判断是否充满
        # if self.soc[t] > self.soc_max:
        #     self.soc[t] = self.soc_max

        # if -0.75 <= self.demand_load[t] <= 0:
        #     self.part_pv[t] = abs(self.demand_load[t]) * self.n_b
        #
        # if self.demand_load[t] < -0.75:
        #     self.part_pv[t] = 0.7125
        #
        # if self.soc[t + 1] + self.part_pv[t] <= self.soc_max:
        #     self.soc[t] = self.soc[t] + self.part_pv[t]
        # else:
        #     self.part_pv[t] = self.soc_max - self.soc[t]
        #     self.soc[t] = self.soc_max
#----------------------------------------------------------------------------------
        # if self.part_pv[t] != 0:
        #     if self.soc[t] + self.part_pv[t] * self.n_b <= self.soc_max:
        #         self.soc[t] = self.part_pv[t] * self.n_b + self.soc[t]
        #     else:
        #         self.soc[t] = self.soc_max
        # else:
        #     self.soc[t] = self.soc[t]

        if self.part_pv[t] > 0:
            PV = self.part_pv[t]
            self.soc[t ] = self.soc[t] + self.part_pv[t] * self.n_b
            # 目前电量+充电量 < 最大

        if bool_B == 2 :#充电






            part_load = min(self.dt * self.n_b * a_P_B, self.soc_max - self.soc[t])
            self.soc[t + 1] = self.soc[t] + part_load * self.n_b





            # if self.soc[t]  + self.dt * self.n_b * a_P_B < self.soc_max:
            #     part_load =  self.dt * self.n_b * a_P_B
            # else: #如果能充满
            #     # self.soc[t + 1]= self.soc_max
            #     part_load = self.soc_max - self.soc[t]

            # self.soc[t+1] = self.soc[t] + part_load


            # print(t,part_pv,part_load,self.soc[t+1])
        # else:
        #     self.soc[t + 1][0] = self.soc_max


            # E_B = a_P_B * self.dt
        elif bool_B == 0 : #放电
            charge_load = min(abs(a_P_B) * self.n_b * self.dt,self.soc[t]*self.n_b)


            self.soc[t + 1] = self.soc[t] - charge_load

        else: #没在用
            self.soc[t + 1] = self.soc[t]
            # E_B = 0




        """
        电价
        """

        if 8 <= self.Hour[t] < 11:
            self.price[t] = 0.6
        elif 11 <= self.Hour[t] < 14:
            self.price[t] = 0.85
        elif 14 <= self.Hour[t] < 18:
            self.price[t] = 0.6
        elif 18 <= self.Hour[t] < 22:
            self.price[t] = 0.85
        else:
            self.price[t] = 0.3

        # price_PV = 0.1

        if 8 <= self.Hour[t + 1] < 11:
            self.price[t + 1] = 0.6
        elif 11 <= self.Hour[t + 1] < 14:
            self.price[t + 1] = 0.85
        elif 14 <= self.Hour[t + 1]< 18:
            self.price[t + 1] = 0.6
        elif 18 <= self.Hour[t + 1] < 22:
            self.price[t + 1] = 0.85
        else:
            self.price[t + 1] = 0.3




        if self.demand_load[t + 1] < 0:
            self.part_pv[t + 1] = min(abs(self.demand_load[t + 1]),self.soc_max - self.soc[t + 1],0.75)
        else:
            self.part_pv[t + 1] = 0






        state_ = np.array([self.soc[t + 1], self.price[t + 1], self.demand_load[t + 1],self.part_pv[t + 1]],dtype=object)
        # state_ = np.array([self.soc[t + 1][0], self.price[t + 1][0]])
        return state_,part_load,charge_load,a_P_B,self.soc[t],PV

    # def mean_load(self,grid):
    #     positive_values = []
    #     for value in real_load:
    #         if value > 0:
    #             positive_values.append(value)
    #
    #     average = sum(positive_values) / len(positive_values)
    #     return average

  # 充电的时候 a_P_B > 0 放电 a_P_B < 0
    def reward(self, t, battery_action,part_load,charge_load,PV):
        reward1 = 0
        reward2 = 0
        reward3 = 0
        pv = PV
        E_B = 0
        A_grid = 0
        bool_B, a_P_B = battery_action[0], battery_action[1]

        if bool_B == 2: #充电 a_P_B > 0
            # if self.soc[t] + self.dt * self.n_b * a_P_B < self.soc_max:
            #     E_B = a_P_B * self.dt
            # else:
            #     E_B = (self.soc_max - self.soc[t])

            E_B = part_load

        elif bool_B == 0: #放电
            # E_B = a_P_B * self.d
            # E_B = - min(self.dt * abs(a_P_B) / (1 / self.n_b), self.soc[t])
            E_B = - charge_load
        else: #不用
            E_B = 0




        a_grid = self.real_load[t] + pv + E_B# 预测负荷在大于0的时候不会出现part_pv 也大于0
        # print(a_grid)
        # 充电的时候E_B < 0 self.real_load[t] - E_B > 0
        # if self.demand_load[t] > 0 :
        #     a_grid = self.demand_load[t] + E_B
        # else:
        #     a_grid =  E_B #盈余部分已经被加到电池里
        # 加入电池后需要的负荷
        # 如果a_grid < 0 则说明可以通过光伏和储能设备自给自足


        # a_grid = self.demand_load[t] + E_B
        #

        # if a_grid < 0:
        #     a_grid = E_B
        # else:
        #     a_grid += E_B



        if a_grid > 0:
            reward1 = - a_grid * self.price[t]
            money_ = abs(reward1)

        else:
            reward1 = 0

            money_ = 0


        A_grid += a_grid
        # print(a_grid, E_B,money_)
        "如果可以出售电"
        """
        price_sell = 0.1
        real_money = self.real_load[t] * self.price[t][0]
        if a_grid < 0:
            # reward1 = abs(a_grid) * price_sell
            self.sell_money = abs(a_grid) * price_sell
            reward1 = self.sell_money
            money_ = -reward1
        else:
            # reward1 = - a_grid * self.price[t][0]
            self.buy_money = a_grid * self.price[t][0]
            reward1 = - self.buy_money
            money_ = abs(reward1)
        """
        "如果不考虑出售电"
        # for day in (1,54)
        total_load = sum(self.real_load[:t])+self.demand_load[t]


        # total_load = sum(self.demand_load[:t + 1])
        average_load = total_load / (t + 1)
        # print(average_load)

        line_load = average_load * 1.3
        line_load_low = average_load * 0.7


        # if abs(a_grid - average_load) > abs(average_load) * 0.15 :
        # # #if (a_grid - average_load)/(average_load) >  0.2:
        #     reward2 = - abs(a_grid - average_load)
        # else:
        #     reward2 = 0
        # reward2 = - abs(a_grid - average_load)

        # reward2 = -abs( a_grid - A_grid/(t+1) )

        if a_grid > 0.2 :
            reward2 = -a_grid
        else:
            reward2 = 0

        reward1_nor = (reward1+1)
        reward2_nor = (reward2+3)/3



        reward_ =   reward1+ reward2


        # reward1_nor = (reward1+1200)/600
        # reward2_nor = (reward2+2700)/800

        reward =  0.1* reward1_nor  +  0.9 * reward2_nor
        # reward = reward2_nor
        # done = False

        # if t >= MAX_STEPS:
        #     done = True

        return reward1,reward2,reward,reward_,money_,a_grid,average_load






    def reset(self,pre_demand):

        self.demand_load = pre_demand

        self.soc = np.zeros(len( self.demand_load))
        self.soc[0] = 0
        # self.price = np.zeros((len( self.demand_load ), 1))
        self.price = np.zeros(len( self.demand_load))
        self.price[0] = 0.3

        self.part_pv = np.zeros(len(self.demand_load))
        self.part_pv[0] = 0
        # self.real_load = grid
        # self.PV_load = PV
        # self.demand_load[0] = 0.551966682


        state_init = np.array([self.soc[0], self.price[0],self.demand_load[0],self.part_pv[0]],dtype=object)

        return state_init