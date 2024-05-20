# 引入依赖库
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
from LSTM_models import *
from LSTM_utils import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import sys
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 必要参数定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 训练设备,如果NVIDIA GPU已配置，会自动使用GPU训练
train_ratio = 0.7  # 训练集比例
val_ratio = 0.15  # 验证集比例
test_ratio = 0.15  # 测试集比例
batch_size = 24  # 批大小，若用CPU，建议为1
input_length = 4  # 每个batch的输入数据长度，多步预测建议长，单步预测建议短
output_length = 1 # 每个batch的输出数据长度，1为单步预测，1以上为多步预测
loss_function = 'MSE'  # 损失函数定义
learning_rate = 0.001  # 基础学习率
weight_decay = 0.0001  # 权重衰减系数
num_blocks = 1  # lstm堆叠次数
dim = 32  # 隐层维度
interval_length =  34364  # 2977  # 预测数据长度，最长不可以超过总数据条数
scalar = True  # 是否使用归一化
scalar_contain_labels = True  # 归一化过程是否包含目标值的历史数据
target_value = 'grid'  # 需要预测的列名，可以在excel中查看
Loss1 = []
Loss2 = []
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用第一个可用的GPU
    print("GPU 可用")
# 多步，单步标签
if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

#  读取数据
df = pd.read_excel("data5_gridalone.xlsx")
df = df[:interval_length]
features_num = 1  # 输入特征维度数量
if features_num > 1:
    features_ = df.values
else:
    features_ = df[target_value].values  #目标预测列的整列值
labels_ = df[target_value].values
# 初步划分训练集、验证集、测试集  train_ratio = 0.7 val_ratio = 0.15
split_train_val, split_val_test = int(len(features_)*train_ratio),\
                                  int(len(features_)*train_ratio)+int(len(features_)*val_ratio)
# print(split_train_val)   24054
# print(split_val_test)    29208
#split_train_val 是标记训练集结束和验证集开始的索引。
#split_val_test 是标记验证集结束和测试集开始的索引。

#  数据标准化
if scalar:
    #min-max scalar
    train_features_ = features_[:split_train_val]

    val_test_features_ = features_[split_train_val:]

    scalar = preprocessing.MinMaxScaler()  #用于进行最小-最大标准化
    if features_num == 1:
        train_features_ = np.expand_dims(train_features_, axis=1)
        val_test_features_ = np.expand_dims(val_test_features_, axis=1)
    train_features_ = scalar.fit_transform(train_features_)  #使用 scalar.fit_transform 对训练集的特征进行最小-最大标准化，
    val_test_features_ = scalar.transform(val_test_features_) #对验证/测试集的特征使用 scalar.transform 进行相同的处理。
    # 重新将数据进行拼接
    features_ = np.vstack([train_features_, val_test_features_])
    if scalar_contain_labels:
        labels_ = features_[:, -1] #表示标签也需要进行归一化处理

if len(features_.shape) == 1: #代码检查 features_ 的形状是否是一维的。
    features_ = np.expand_dims(features_,0).T #用 np.expand_dims 对数据进行维度变换，使其变成二维的。
features, labels = get_rolling_window_multistep(output_length, 0, input_length,
                                                features_.T, np.expand_dims(labels_, 0))

#  构建数据集
labels = torch.squeeze(labels, dim=1) #将 labels 张量中的所有维度为1的维度去除，以确保 labels 张量的形状适用于后续的操作。
features = features.to(torch.float32)
labels = labels.to(torch.float32)  #这两行代码将 features 和 labels 张量的数据类型转换为 torch.float32，以确保数据类型的一致性。
split_train_val, split_val_test = int(len(features)*train_ratio), int(len(features)*train_ratio)+int(len(features)*val_ratio)
train_features, train_labels = features[:split_train_val], labels[:split_train_val]  #训练集的特征和标签

val_features, val_labels = features[split_train_val:split_val_test], labels[split_train_val:split_val_test]

test_features, test_labels = features[split_val_test:], labels[split_val_test:]


#  数据管道构建，此处采用torch高阶API
train_Datasets = TensorDataset(train_features.to(device), train_labels.to(device))  #这是一个 TensorDataset 对象，用于包装训练数据。TensorDataset 将特征和标签组合成一个数据集。
train_Loader = DataLoader(batch_size=batch_size, dataset=train_Datasets)  #train_Loader：这是一个 DataLoader 对象，用于批处理训练数据。
val_Datasets = TensorDataset(val_features.to(device), val_labels.to(device))
val_Loader = DataLoader(batch_size=batch_size, dataset=val_Datasets)
test_Datasets = TensorDataset(test_features.to(device), test_labels.to(device))
test_Loader = DataLoader(batch_size=batch_size, dataset=test_Datasets)


#  模型定义
AttentionLSTM_model = AttentionLSTM(input_features_num=features_num, output_len=output_length,
                                  lstm_hidden=dim, lstm_layers=num_blocks, batch_size=batch_size, device=device)
AttentionLSTM_model.to(device)
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')  #nn.MSELoss 是 PyTorch 中的均方误差损失函数，它用于度量模型的预测值与实际标签之间的差异。

#  训练代数定义
epochs = 400
#  优化器定义，学习率衰减定义
optimizer = torch.optim.AdamW(AttentionLSTM_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//3, eta_min=0.00001)

#  训练及验证循环
print("——————————————————————Training Starts——————————————————————")
for epoch in range(epochs):
    # 训练
    AttentionLSTM_model.train()
    train_loss_sum = 0 #用于累积每个训练周期内的总损失
    step = 1
    for step, (feature_, label_) in enumerate(train_Loader): #这是训练数据加载器的循环，它会遍历训练数据集并执行以下操作：

        optimizer.zero_grad() #重置优化器的梯度
        feature_ = feature_.permute(0,2,1) #调整特征数据的维度
        prediction = AttentionLSTM_model(feature_) #通过模型进行前向传播，生成预测值
        # print("prediction:{}".format(prediction))
        loss = loss_func(prediction, label_) #计算损失
        loss.backward() #反向传播误差
        torch.nn.utils.clip_grad_norm_(AttentionLSTM_model.parameters(), 0.15) #对梯度进行裁剪，以避免梯度爆炸问题
        optimizer.step() #更新模型参数
        train_loss_sum+=loss.item() #累计训练损失
        Loss1.append(train_loss_sum)
        Loss1 = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Loss1]
        df1 = pd.DataFrame(Loss1)
        csv_file_path = "Loss.csv"
        df1.to_csv(csv_file_path, index=False)
    print("epochs = " + str(epoch))
    print('train_loss = ' + str(train_loss_sum))

    #  验证
    AttentionLSTM_model.eval()
    val_loss_sum = 0
    val_step = 1
    for val_step, (feature_, label_) in enumerate(val_Loader):
        feature_ = feature_.permute(0, 2, 1)
        with torch.no_grad():
            prediction = AttentionLSTM_model(feature_)
            val_loss = loss_func(prediction, label_)
        val_loss_sum += val_loss.item()
        Loss2.append(val_loss_sum)
        Loss2 = [np.array([item]) if not isinstance(item, np.ndarray) else item for item in Loss2]
        df2 = pd.DataFrame(Loss2)
        csv_file_path = "Loss2.csv"
        df2.to_csv(csv_file_path, index=False)
    if epoch == 0:
        val_best = val_loss_sum
        print('val_loss = ' + str(val_loss_sum))
    else:
        print('val_loss = ' + str(val_loss_sum))
        if val_best > val_loss_sum:
            val_best = val_loss_sum
            torch.save(AttentionLSTM_model.state_dict(), 'model_AttentionLSTM_weights')  # 保存最好权重
            torch.save(AttentionLSTM_model,'model.pth')
            #并保存以下网络模型！ .pth格式
            print("val_best change")
print("best val loss = " + str(val_best))

print("——————————————————————Training Ends——————————————————————")

#  测试集预测
AttentionLSTM_model.load_state_dict(torch.load('model_AttentionLSTM_weights'))  # 调用权重
test_loss_sum = 0
#  测试集inference
print("——————————————————————Testing Starts——————————————————————")
for step, (feature_, label_) in enumerate(test_Loader):
    # torch.set_printoptions(profile="full")
    # print(feature_)

    feature_ = feature_.permute(0, 2, 1)
    # print(len(features_)) 34364
    with torch.no_grad():
         if step ==0:
            prediction = AttentionLSTM_model(feature_)
            pre_array = prediction.cpu()
            label_array = label_.cpu()
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
         else:
            prediction = AttentionLSTM_model(feature_)
            pre_array = np.vstack((pre_array, prediction.cpu()))
            label_array = np.vstack((label_array, label_.cpu()))
            loss = loss_func(prediction, label_)
            test_loss_sum += loss.item()
print("test loss = " + str(test_loss_sum))
print("——————————————————————Testing Ends——————————————————————")

# 数据后处理，单步预测绘制全部预测值的图像，多步预测仅绘制第一个batch的输出图像
#  逆归一化过程及绘制图像
print("——————————————————————Post-Processing——————————————————————")
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)  # 用于筛选出 y_true 中大于零的元素的索引
    y_true = y_true[non_zero_index]  # 筛选出大于零的真实值和对应的预测值
    y_pred = y_pred[non_zero_index]
    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0  # 无穷大值（如果存在）替换为零
    return np.mean(mape) * 100

if scalar_contain_labels and scalar:
    pre_inverse = []
    test_inverse = []
    if features_num == 1 and output_length == 1:  #不看
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    elif features_num>1:
        if isinstance(pre_array, np.ndarray):
            pre_array = torch.from_numpy(pre_array)
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(torch.cat((torch.zeros(pre_array[0].shape[0], features_num-1),torch.unsqueeze(pre_array[pre_slice], dim=1)), 1))[:,-1]
            test_inverse_slice = scalar.inverse_transform(torch.cat((torch.zeros(test_labels[0].shape[0], features_num-1), torch.unsqueeze(test_labels[pre_slice], dim=1)), 1))[:,-1]
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse)
        test_labels = np.array(test_inverse)
    else:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(label_array[pre_slice,:], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    plt.figure(figsize=(10,5))
    df = pd.DataFrame(pre_array)
    csv_file_path = "pre_grid_alone.csv"
    df.to_csv(csv_file_path, index=False)
    print(pre_array)
    print(test_labels)
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g',label = 'pred')
        plt.plot(test_labels[0], "r", label = 'real')
        plt.show()
    else:
        plt.plot(pre_array[:], 'g',label = 'pred')
        plt.plot(test_labels[:], "r",label = 'real', alpha=0.5)
        plt.legend()
        plt.show()
    #  计算衡量指标
    MSE_l = mean_squared_error(test_labels[:], pre_array[:])
    MAE_l = mean_absolute_error(test_labels[:], pre_array[:])
    MAPE_l = mape(test_labels[:],pre_array[:])
    # MAPE_l = mean_absolute_percentage_error(test_labels, pre_array)
    R2 = r2_score(test_labels[:], pre_array[:])
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)
"""
else:
    plt.figure(figsize=(40,20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g')
        plt.plot(test_labels[0].cpu(), "r")
        plt.show()
    else:
        plt.plot(pre_array, 'g')
        plt.plot(test_labels.cpu(), "r")
        plt.show()
    MSE_l = mean_squared_error(test_labels.cpu(), pre_array)
    MAE_l = mean_absolute_error(test_labels.cpu(), pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels.cpu(), pre_array)
    R2 = r2_score(test_labels.cpu(), pre_array)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)
    """