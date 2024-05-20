import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from copy import deepcopy
from torch.distributions import  Normal
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_Network(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)     # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)     # 权重初始化 (均值为0，方差为0.1的正态分布)
        # self.fc1.weight.data.normal_(0, 0.1)      # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc1.bias.data.normal_(0.1)         # 偏置初始化 (均值为0，方差为0.1的正态分布)
        # self.fc2 = nn.Linear(128, 64)            ## 设置第二个全连接层: 50个神经元到动作数个神经元
        # self.fc2.weight.data.normal_(0, 0.1)     ## 权重初始化 (均值为0，方差为0.1的正态分布)
        # self.fc2.bias.data.normal_(0.1)        # 偏置初始化 (均值为0，方差为0.1的正态分布)
        self.fc2 = nn.Linear(64, action_dim)    # 设置第三个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.fc2.weight.data.normal_(0, 0.1)    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.fc2.bias.data.normal_(0.1)         # 偏置初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x),negative_slope=0.01)
        #out = F.relu(self.fc2(out))
        out = self.fc2(out)

        #out = torch.tanh(out)
        return out

class DQN2(object):

    def __init__(self, state_dim, action_dim, gamma, memory_size, learning_rate, epsilon, batch_size, delay_update,l2_reg):

        self.state_dim = state_dim                            #状态维度
        self.action_dim = action_dim                          #动作维度
        self.gamma = gamma
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        # self.epsilon_decay = 0.0001 #0.00003
        self.epsilon_decay = 0.00001#0.00003
        self.batch_size = batch_size
        self.delay_update = delay_update
        self.loss = []
        self.counter = 0    #计数
        self.Q_eval_network, self.Q_target_network = Q_Network(self.state_dim, self.action_dim), \
                                                     Q_Network(self.state_dim, self.action_dim)
        self.replay_memory = np.zeros((self.memory_size, self.state_dim * 2 + 2))
        self.index_memory = 0      #表示初始时位置位于经验回放缓存的开头。
        self.optimizer = torch.optim.Adam(self.Q_eval_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.l2_reg = l2_reg

    def choose_action(self, observation):
        obs_array = np.array(observation, dtype=np.float)
        s = torch.tensor(obs_array,dtype=torch.float)
        #torch.FloatTensor(observation)：这部分将观察数据转换为 PyTorch 的浮点数张量（Tensor）。
        #torch.unsqueeze(..., 0)：这是一个操作，用于在张量中增加一个维度。
        a = self.Q_eval_network(s)  #把s状态放到评价网络中
        if self.epsilon > 0.01:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon =0.01
        if np.random.uniform() > self.epsilon:
            #基于一个随机数，与 ε 进行比较，以决定是进行探索还是利用。np.random.uniform() 生成一个 0 到 1 之间的随机数。
            action = torch.argmax(a).item()
            # print(action)
            # action = torch.max(a, 1)[1].detach().numpy()[0]
            #表示进行利用，将根据当前的 Q 值网络输出 a 来选择具有最大 Q 值的动作。
        else:
            action = np.random.randint(0, self.action_dim )
            #在探索时，随机选择一个动作，从动作空间中选择一个索引，

        return action
    #返回一个动作

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, [a, r],  s_))
        index1 = self.index_memory % self.memory_size  # %是取余数 index1 在0到 memory_size-1 之间
        self.replay_memory[index1, :] = memory       #将memory 根据索引index1 将整行赋给replay_memory
        self.index_memory += 1                     #行数+1

    def sample_memory(self):
        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.replay_memory[sample_memory_index, :]

        sample_memory_s = torch.FloatTensor(sample_memory[:, : self.state_dim])
        sample_memory_a = torch.LongTensor(sample_memory[:, self.state_dim: 1 + self.state_dim].astype(int))
        sample_memory_r = torch.FloatTensor(sample_memory[:,- self.state_dim - 2: - self.state_dim - 1])
        sample_memory_s_ = torch.FloatTensor(sample_memory[:, - self.state_dim - 1: -1])
        # sample_memory_d = torch.FloatTensor(sample_memory[:, -1:])

        # return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_d
        return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_


    def learn(self):

        # 每过delay_update步长跟新target_Q
        if self.counter % self.delay_update == 0:
            for parm, target_parm in zip(self.Q_eval_network.parameters(), self.Q_target_network.parameters()):
                target_parm.data.copy_(parm.data)
        self.counter += 1
        s, a, r, s_ = self.sample_memory() #从经验回放内存中随机抽取一批样本，分别获取当前状态 s、动作 a、奖励 r、下一个状态 s_ 和结束标志 d。
        # 根据a来选取对应的Q(s,a)
        q_eval = self.Q_eval_network(s).gather(1, a)
        # 计算target_Q
        # q_target = self.gamma * self.Q_target_network(s_).detach().max(1)[0].view(self.batch_size, 1)
        q_target = self.gamma * torch.max(self.Q_target_network(s_), 1)[0].reshape(self.batch_size, 1)
        y = r + q_target  #d为结束标志 未结束是0  结束是1 结束后 y = r
        # l2_reg_loss = 0.5 * sum(p.norm(2) ** 2 for p in self.Q_eval_network.parameters())
        # 网络跟新
        loss = self.loss_function(y, q_eval)
        # loss = self.loss_function(y, q_eval) + self.l2_reg * l2_reg_loss
        self.loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()