import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt

train_data = pd.read_csv('train_data.csv')
# 随机挑选5%的数据作为验证集
val_data = train_data.sample(frac=0.05, random_state=42)

# 剩余的95%数据作为训练集
train_data = train_data.drop(val_data.index)
test_data = pd.read_csv('test_data.csv')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#假设你的输入DataFrame的名字时XXX
#可以使用下面这个语句来判断输入数据中是否包含隐蔽的Nan
train_data.isnull().any()
#更加具体地，可以使用下面这个语句来输出具体时哪些列包含nan
train_data.isnull()

patience = 15
epochs = 10  # The number of epochs

# # 96
# window_size = 96
# batch_size = 16
# lr = 1e-5
# pt_path = './models/trm_96sw.pt'
# pic_path = './res/sw96_'

# 336
window_size = 96
batch_size = 16  # batch size
lr = 1e-5
pt_path = '../models/trm_96sw.pt'
pic_path = '../res/sw96_'

'''
    model
'''


class Trm(nn.Module):
    def __init__(self, feature_size=15, num_layers=3, dropout=0):
        super(Trm, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=15, dropout=dropout)  # 编码
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)  # 解码
        self.decoder = nn.Linear(feature_size, feature_size)
        self.relu = nn.ReLU()  # 激活函数
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):  # mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, mask)
        output = self.relu(output)  # 激活函数
        output = self.decoder(output)
        return output


'''
    数据集的构建 滑动窗口
'''


def create_inout_sequences(input_data, window_size):
    inout_seq = []
    L = len(input_data)
    for i in range(L - window_size):
        if (i + window_size + window_size) > len(input_data):
            break
        train_seq = input_data[i:i + window_size]  # input
        train_label = input_data[i + window_size:i + window_size + window_size]  # output
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


'''
    数据划分和预处理
'''


def get_data(train_data, val_data, test_data, window_size):
    # 数据长度
    train_len = len(train_data)
    val_len = len(val_data)
    test_len = len(test_data)

    # 合并所有数据
    data = np.concatenate((train_data, val_data, test_data), axis=0)

    # 去除时间数据
    data = data[:, 2:]

    # 对原始数据进行归一化处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    processed_data = scaler.fit_transform(data)  # 对全量数据归一化

    # 数据集划分
    train_data = processed_data[:train_len]
    val_data = processed_data[train_len: train_len + val_len]
    test_data = processed_data[train_len + val_len:]

    # Train data
    train_data = create_inout_sequences(train_data, window_size)
    # Validation data
    val_data = create_inout_sequences(val_data, window_size)
    # Test data
    test_data = create_inout_sequences(test_data, window_size)

    return train_data.to(device), val_data.to(device), test_data.to(device), scaler

'''
def get_data(train_data, val_data, test_data):
    # 数据长度
    train_len = len(train_data)
    val_len = len(val_data)
    test_len = len(test_data)
    data = np.concatenate((train_data, val_data, test_data), axis=0)
    # 去除时间数据
    data = data[:, 2:]
    # 差分处理
    diff = np.diff(data, axis=0)
    # 归一化处理
    scaler = MinMaxScaler(feature_range=(0,1))
    processed_data = scaler.fit_transform(diff)

    # 数据集划分
    train_data = processed_data[:train_len]
    val_data = processed_data[train_len: train_len + val_len]
    test_data = processed_data[train_len + val_len:]

    # train data
    train_data = create_inout_sequences(train_data, window_size)
    # val data
    val_data = create_inout_sequences(val_data, window_size)
    # test_data
    test_data = create_inout_sequences(test_data, window_size)

    return train_data.to(device), val_data.to(device), test_data.to(device), scaler
'''

'''
    train
'''


def train(model,train_data):
    model.train()
    total_loss = 0.0
    cnt = 0
    for batch, i in enumerate(range(0, len(train_data), batch_size)):
        cnt += 1
        # 设置上界
        border = min(i + batch_size, len(train_data) - 1)
        data, targets = train_data[border, 0, :, :], train_data[border, 1, :, :]

        optimizer.zero_grad()
        # 数据输入
        output = model(data).to(device)

        # loss
        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()  # 优化

        total_loss += loss.item()
    return total_loss / cnt


'''
    val
'''


def evaluate(eval_model, val_data):
    total_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            cnt += 1
            # 设置上界
            border = min(i + batch_size, len(val_data) - 1)
            data, targets = val_data[border, 0, :, :], val_data[border, 1, :, :]

            # 评估模型
            output = eval_model(data)
            loss = criterion(output, targets)

            total_loss += loss.item()
    return total_loss / cnt


'''
    test
'''


def predict(eval_model, test_data, cnt, scaler):
    # 存储mse、mae数据
    mse = [0] * cnt
    mae = [0] * cnt
    run = 0

    eval_model.eval()
    with torch.no_grad():
        # 随机选择测试数据段
        for i in range(0, len(test_data), 300):
            predictions = []  # 预测将来值
            true_values = []  # 实际将来值
            pre_values = []  # 实际当前值

            data, targets = test_data[i, 0, :, :], test_data[i, 1, :, :]
            pre_targets = test_data[i - window_size, 1, :, :]

            # 解决PyTorch张量在cpu和gpu的存储访问问题
            data.cpu().data.numpy()
            targets.cpu().data.numpy()
            pre_targets.cpu().data.numpy()

            output = eval_model(data)

            predictions.extend(output)
            true_values.extend(targets)
            pre_values.extend(pre_targets)

            predictions = torch.cat(predictions, dim=0).cpu().numpy()
            true_values = torch.cat(true_values, dim=0).cpu().numpy()
            pre_values = torch.cat(pre_values, dim=0).cpu().numpy()

            # 归一化的还原
            predictions = scaler.inverse_transform((np.array(predictions)).reshape(-1, 15))
            true_values = scaler.inverse_transform((np.array(true_values)).reshape(-1, 15))
            pre_values = scaler.inverse_transform((np.array(pre_values)).reshape(-1, 15))
            # print(true_values.shape, len(true_values))

            ### 差分还原
            # 未来预测值
            re_predictions = np.cumsum(predictions, axis=0)
            rediff_pred = []
            for i in range(len(re_predictions)):
                rediff_pred.append(predictions[0] + re_predictions[i])
            predictions = np.array(rediff_pred)

            # 未来真实值
            re_true_values = np.cumsum(true_values, axis=0)
            rediff_true = []
            for i in range(len(re_true_values)):
                rediff_true.append(true_values[0] + re_true_values[i])
            true_values = np.array(rediff_true)

            # 当前真实值
            re_pre_true = np.cumsum(pre_values, axis=0)
            rediff_pretrue = []
            # rediff_pretrue.append(pre_values[0])
            for i in range(len(re_pre_true)):
                rediff_pretrue.append(pre_values[0] + re_pre_true[i])
            pre_values = np.array(rediff_pretrue)

            # 计算mse和mae
            mse[run] = MSE(predictions, true_values)
            mae[run] = MAE(predictions, true_values)

            print(f"MSE: {mse[run]:.5f}")
            print(f"MAE: {mae[run]:.5f}")

            # 获取最后一列OT的数据
            true_values = true_values[:, -1]
            print('true----------\n', true_values.shape, true_values)
            predictions = predictions[:, -1]
            print('pred----------\n', predictions.shape, predictions)
            pre_values = pre_values[:, -1]
            print('pre----------\n', pre_values.shape, pre_values)

            # 将当前真实值和未来真实值合并
            truth = np.concatenate((pre_values, true_values), axis=0)
            # print('----------\n', truth.shape, truth)

            # 预测结果横坐标平移
            x = np.linspace(window_size, window_size * 2, window_size)

            plt.figure(figsize=(12, 6))
            plt.plot(truth, label='true_values')  # 真实值
            plt.plot(x, predictions, label='Predictions')  # 预测值

            plt.ylabel('Value')
            plt.title('Predictions vs Ground Truth')
            plt.legend()

            # 图片保存路径
            pic_file = pic_path + str(run + 1) + '.jpg'
            print(pic_file)
            # 如果文件夹不存在，则创建它
            os.makedirs(os.path.dirname(pic_file), exist_ok=True)
            plt.savefig(pic_file)
            plt.close()

            run += 1
            if run == cnt:  # 测试5轮
                break

        # mse和mae的平均和标准差
        print('avg mse: ', np.mean(mse), '   std mse: ', np.std(mse))
        print('avg mae: ', np.mean(mae), '   std mae: ', np.std(mae))




'''
    MAE
'''


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


'''
    MSE
'''


def MSE(pred, true):
    return np.mean((pred - true) ** 2)

if __name__ == '__main__':
    if not os.path.exists(pt_path):
        print(f"Error: The model file at {pt_path} does not exist.")

    # 获取数据
    train_data, val_data, test_data, scaler = get_data(train_data, val_data, test_data,window_size)
    criterion = nn.MSELoss()

    # 初始化模型、优化器和学习率调度器
    model = Trm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3.0, gamma=0.98)

    best_val_loss = np.inf
    best_model = None
    cur_wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        cnt = 0
        for batch, i in enumerate(range(0, len(train_data), batch_size)):
            cnt += 1
            border = min(i + batch_size, len(train_data) - 1)
            data, targets = train_data[border, 0, :, :], train_data[border, 1, :, :]

            optimizer.zero_grad()
            output = model(data).to(device)

            loss = criterion(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / cnt

        # validation
        total_loss = 0.0
        cnt = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                cnt += 1
                border = min(i + batch_size, len(val_data) - 1)
                data, targets = val_data[border, 0, :, :], val_data[border, 1, :, :]

                output = model(data)
                loss = criterion(output, targets)

                total_loss += loss.item()
        val_loss = total_loss / cnt

        print('-' * 89)
        print('| end of epoch {:3d} | train loss {:.5f} | valid loss {:.5f} '.format(epoch, train_loss, val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            cur_wait = 0
        else:
            cur_wait += 1
            if cur_wait >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    # 保存最好的模型
    torch.save(best_model.state_dict(), pt_path)
    print('finish.')

    # 加载最好的模型
    model = Trm().to(device)
    model.load_state_dict(torch.load(pt_path))
    model.eval()  # 设置为评估模式

    # 进行预测
    predict(model, test_data, 5, scaler)
