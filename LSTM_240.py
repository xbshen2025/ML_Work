import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

# 加载数据
train_dataset = pd.read_csv('data/train_data.csv')
test_dataset = pd.read_csv('data/test_data.csv')
train_dataset = train_dataset[0:-3]



device = "cuda:0"
def process_data(data):
    data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)
    data = np.array(data.values)
    return data

# train_cnt = train_dataset['cnt']
# train_scaler = MinMaxScaler(feature_range=(0, 1))
# train_values = train_cnt.values.astype(int)
# train_scaled_values = train_scaler.fit_transform(train_values.reshape(-1, 1))
# train_dataset['cnt'] = train_scaled_values
#
# test_scaler = MinMaxScaler(feature_range=(0, 1))
# test_cnt = test_dataset['cnt']
# test_values = test_cnt.values
# test_scaled_values = test_scaler.fit_transform(test_values.reshape(-1, 1))
# test_dataset['cnt'] = test_scaled_values

scaler = MinMaxScaler(feature_range=(0, 1))
train_cnt = train_dataset['cnt']
test_cnt = test_dataset['cnt']

train_values = train_cnt.values.reshape(-1)
test_values = test_cnt.values.reshape(-1)

values = np.concatenate((train_values, test_values))
scaled_values = scaler.fit_transform(values.reshape(-1, 1))

train_dataset['cnt'] = scaled_values[0:len(train_values)]
test_dataset['cnt'] = scaled_values[len(train_values):]



train_dataset = process_data(train_dataset)
train_label = train_dataset[:, -1].reshape(-1)


test_dataset = process_data(test_dataset)
test_label = test_dataset[:, -1].reshape(-1)

def create_sequences(feature, label, past_steps, future_steps):
    temp = []
    for i in range(0, len(feature) - past_steps - future_steps + 1):
        X = feature[i:i+past_steps].reshape(-1, 13)
        y = label[i+past_steps:i+past_steps+future_steps].reshape(-1, 1)
        temp.append((X, y))
    return temp


# 设置时间窗口
past_steps = 96
future_steps = 240

# 创建输入和输出序列
train_seq = create_sequences(train_dataset, train_label, past_steps, future_steps)


class MyDataset(Dataset):
    def __init__(self, data):

        self.data = data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, index):
        input, label = self.data[index]
        return input, label

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# 超参数
input_size = 13
hidden_size = 1024
output_size = future_steps
seq_length = future_steps
num_epochs = 30
batch_size = 8
learning_rate = 0.001

train_data = MyDataset(train_seq)
# 数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True
)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, output_size, seq_length).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in tqdm(range(num_epochs), desc='正在训练'):
    for batch in train_loader:
        inputs, targets = batch
        inputs = inputs.to(torch.float32).to(device)
        targets = targets.to(torch.float32).reshape(-1).to(device)
        # 前向传播
        outputs = model(inputs)
        outputs = outputs.reshape(-1)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


test_inputs = train_dataset[-96:].tolist()
test_inputs.extend(test_dataset.tolist())

pred = []
# 评估模型
model.eval()
for i in range(0, len(test_dataset), future_steps-past_steps):
    seq = torch.FloatTensor(test_inputs[i:i+96]).reshape(1, 96, 13).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(2, 1, model.hidden_size),
                        torch.zeros(2, 1, model.hidden_size))

        #actual_predictions = train_scaler.inverse_transform(np.array(model(seq)[0].cpu()).reshape(-1, 1))
        actual_predictions = np.array(model(seq)[0].cpu())
        actual_predictions = actual_predictions.reshape(-1)
        pred.extend(actual_predictions.reshape(-1))
pred = pred[0:2160]
# 计算 MSE 和 MAE
mse = mean_squared_error(test_label, pred)
mae = mean_absolute_error(test_label, pred)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')

# 绘制图像并保存
plt.figure(figsize=(12, 6))
a = max(pred)
b = max(test_values)
plt.ylim(0, 1)
plt.plot(pred, label='Predicted')
plt.plot(test_label, label='Actual')

plt.legend()
plt.title('Prediction vs Actual for One Sample')
plt.xlabel('Hour')
plt.ylabel('Count')

# 保存图像
output_dir = 'output'  # 指定输出目录
os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
plt.savefig(os.path.join(output_dir, 'prediction_vs_actual_240.png'))  # 保存图像
plt.show()
