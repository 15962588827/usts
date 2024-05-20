import numpy as np
import torch
from DQN_AGENT import DQN2
import random
import pandas as pd
import time
from env1 import single_agent_Env
import matplotlib.pyplot as plt
from sklearn import preprocessing

if __name__ == '__main__':

    data4 = pd.read_csv('data4.csv')

    data4_ = np.array(data4)

    Hour = data4_[:, 0] #时间
    grid = data4_[:, 1] #真实负荷
    grid = grid.reshape(-1, 1)
    pre_demand = data4_[:, 2] #预测负荷
    pre_demand = pre_demand.reshape(-1, 1)


    start_time = time.time()
    seed = 2024010318
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 设置 CPU 生成随机数的 种子

#状态应该增加一个维度
    dqn = DQN2(state_dim=4,
               action_dim=13,
               gamma=0.99,
               memory_size=10000,
               learning_rate=0.001,#0.0001
               epsilon=0.99,
               batch_size=32,
               delay_update=4,
               l2_reg=0.001)
    #初始化预测网络  ，加载pth 网络参数
    # model = torch.load('model.pth')
    # model.eval()


    # 加载特征的csv文件，归一化操作等



    terminal = 5087  # 96个15min步长 = 24h


    # 这里先把动作空间设置好，如下
    action_ = []  # 所有的动作空间
    for i in np.arange(-3, 3.5, 0.5):
        action_.append(i)

    env1 = single_agent_Env(Hour, pre_demand,grid)
    #每一轮的奖励、轮数、电费
    Episode_rewards = []
    Episode_numbers = []
    Episode_money = []
    #每一轮中一天的奖励、天数、电费
    Day_Money = []
    Day_REWARD = []
    Day_number = []
    Average_load = []
    Grid = []
    Money = []
    return_avg = []
    Grid_7day = []
    Grid_epis5 = []
    Grid_epis10 = []
    Grid_epis15 = []

    date = 53 #总的是53天\
    Soc_action = []
    Soc_state = []

    #加一个维度  用于存放预测值
    for episode in range(20): # 训练150轮
        s = env1.reset(pre_demand)
         #每一轮环境都初始化
        episode_reward_sum = 0 #每一轮的奖励总和
        episode_reward_sum_ = 0
        episode_money = 0   #每一轮的钱总和
        episode_reward1 = 0
        episode_reward2 = 0
        reward_sum = 0
        reward_sum_ = 0
        reward1 = 0
        reward2 = 0
        money = 0
        """
        for day in range(0,date):
            day_reward_sum = 0 #每一天的奖励总和
            day_reward_sum_ = 0
            day_money = 0   #每天的钱总和
            day_reward1 = 0
            day_reward2 = 0
            regluar_load = 0
        """
        for t in range(0, terminal):

            #if
                #pre load  =model（[now load，归一化的特征]）
                #s=[now load,pre load]
            # print(s)
            # print("******")
            a = dqn.choose_action(s)  #输出的是动作索引
            #print(a)
            true_action = action_[a]


            battery_a = true_action

            #判断battery的充放电
            #[0,1] 0代表充放电状态，1代表充放电的功率
            #if bool_B == 2:  # 充电
            battery_action = np.array([0.0, 0.0])


            if battery_a < 0: # 放电
                    battery_action[0] = 0
            elif battery_a == 0: #不用
                    battery_action[0] = 1
            else:  #充电
                    battery_action[0] = 2

            battery_action[1] = battery_a

            # battery_action[0,1]  0 放电 1功率

            s_,part_load,charge_load,soc_action,soc_state,PV= env1.step(t, battery_action)

            r1, r2, r,r_,rmb,a_grid,average_load = env1.reward(t, battery_action,part_load,charge_load,PV)

            dqn.store_memory(s, a, r, s_)

            if dqn.index_memory > dqn.memory_size: #大于10000开始学习
                    dqn.learn()

            s = s_

            reward_sum += r #这一轮的累积奖励
            reward_sum_ += r_
            reward1 += r1
            reward2 += r2
            money += rmb #这一轮的累积钱
            regluar_load = a_grid
            if episode == 99  :
                Grid.append(regluar_load)
                Money.append(rmb)
                Average_load.append(average_load)
                Soc_action.append(soc_action)
                Soc_state.append(soc_state)
            if episode == 5 :
                Grid_epis5.append(regluar_load)
            if episode == 5:
                Grid_epis10.append(regluar_load)




        Episode_money.append(money) # 每回合的钱
        Episode_rewards.append(reward_sum_) #每回合的总奖励
        Episode_numbers.append(episode) #回合数
        return_avg.append(sum(Episode_rewards) / (episode + 1)) #每回合的平均奖励
        print("episode:{}, reward:{}, ave_return: {},reward1:{},reward2:{},money{}".format(episode + 1, reward_sum_,sum(Episode_rewards) / (episode + 1), reward1, reward2,money))
        # if episode == 1:
        #     while day == 50:
        #         Grid.append(regluar_load)
        #         break
        # print(Money_)

    # torch.save(dqn.Q_eval_network.state_dict(), "Q_eval_network.pth")

    # 记录52天所有步的调峰负荷
    Grid1 = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Grid]
    df1 = pd.DataFrame(Grid1)
    csv_file_path = "Result/Grid_last.csv"
    df1.to_csv(csv_file_path,index=False)

    # 记录52天每一个时间步的钱
    standard_list = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Money]
    df2 = pd.DataFrame( standard_list)
    csv_file_path = "Result/52day_money_last.csv"
    df2.to_csv(csv_file_path, index=False)

    #记录每一回合的奖励
    Episode_rewards1 = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Episode_rewards]
    df3 = pd.DataFrame(Episode_rewards1)
    csv_file_path = "Result/Reward_last.csv"
    df3.to_csv(csv_file_path,index=False)

    # 记录每一回合的平均奖励
    return_avg = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in return_avg]
    df4= pd.DataFrame(return_avg)
    csv_file_path = 'Result/Reward_average_last.csv'
    df4.to_csv(csv_file_path,index=False)

    # 记录每一回合的总电费
    Episode_money = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Episode_money]
    df5 = pd.DataFrame(Episode_money)
    csv_file_path = 'Result/Episode_money_last.csv'
    df5.to_csv(csv_file_path,index=False)

    #记录52天的所有时间步的电池动作
    Soc_action = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Soc_action]
    df6 = pd.DataFrame(Soc_action)
    csv_file_path = "Result/Soc_action_last.csv"
    df6.to_csv(csv_file_path, index=False)

    #记录52天的所有时间步的电池状态
    Soc_state = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Soc_state]
    df7 = pd.DataFrame(Soc_state)
    csv_file_path = "Result/Soc_state_last.csv"
    df7.to_csv(csv_file_path, index=False)

    plt.plot(Episode_numbers, Episode_rewards, color='red', label='Episode Reward')  # 指定颜色为红色
    plt.plot(Episode_numbers,return_avg,color = 'blue',label = 'Avgerage Reward')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Reward vs. Episode')
    plt.legend()  # 添加图例
    plt.grid(True)  # 添加网格线
    plt.show()


    plt.plot(Episode_numbers,Episode_money,color = 'blue',label = 'Episode Money')
    plt.xlabel('Episode')
    plt.ylabel('Episode Money')
    plt.title('Episode Money vs. Episode')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 把需要的处理和分析的数据保存，例如奖赏，温度，储能等数据
    # 把网络参数模型保存用于测试

    plt.figure(figsize=(10, 5))
    plt.plot(range(0,5087),Grid[:],color = 'green',label = 'regular',alpha = 0.8)
    plt.plot(range(0,5087),grid[:5087],color = 'red',label = 'real',alpha = 0.5)
    plt.plot(range(0,5087),Average_load[:],color ='blue',label = 'average',alpha = 1)
    plt.xlabel('step')
    plt.ylabel('Gird')
    plt.title('peak_regular vs. Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, 672), Grid[960:1632], color='green', label='regular', alpha=0.8)
    plt.plot(range(0, 672), grid[960:1632], color='red', label='real', alpha=0.5)
    plt.xlabel('step')
    plt.ylabel('7_dayGird')
    plt.title('7day_peak_regular vs. Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

