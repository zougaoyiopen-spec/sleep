import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 时序注意力机制模块 (针对 LSTM 优化维度)
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        # 学习一个权重向量，用来评估每个时间步的重要性
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        # ⚠️ 注意：LSTM 的输出直接就是 [Batch, Seq_Len, Features]
        # 所以这里不需要像原来那样做 permute 转置了

        # 计算每个时间步的注意力分数
        attn_weights = self.attention(x)  # shape: [Batch, Seq_Len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 沿时间维度做 Softmax归一化

        # 将分数乘回特征上，并求和 (加权平均)
        context_vector = torch.sum(attn_weights * x, dim=1)  # shape: [Batch, Features]
        return context_vector


# ==========================================
# 2. 终极优化版模型：MultiScale-CNN + BiLSTM + Attention
# ==========================================
class Sleep_Model(nn.Module):
    def __init__(self, num_classes=5, in_channels=1):
        super(Sleep_Model, self).__init__()

        # ==========================================
        # 模块 1：多尺度 CNN (Multi-scale CNN)
        # ==========================================
        # 分支 A：小卷积核 (捕获高频：纺锤波 12-14Hz, Alpha波 8-13Hz)
        self.branch_small = nn.Conv1d(in_channels, 32, kernel_size=64, stride=4, padding=30)
        # 分支 B：大卷积核 (捕获低频：Delta慢波 0.5-4Hz)
        self.branch_large = nn.Conv1d(in_channels, 32, kernel_size=256, stride=4, padding=126)

        self.bn_multi = nn.BatchNorm1d(64)  # 32+32=64
        self.pool_multi = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        # ==========================================
        # 模块 2：深层特征提取
        # ==========================================
        self.cnn_deep = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
        )

        # ==========================================
        # 模块 3：Bi-LSTM 替换 TCN
        # ==========================================
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # 双向LSTM，输出维度变回 64*2 = 128
        )

        # ==========================================
        # 模块 4：时序注意力机制
        # ==========================================
        self.attention = TemporalAttention(feature_dim=128)

        # ==========================================
        # 模块 5：分类器
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 1. 多尺度特征提取 & 拼接
        out_small = self.branch_small(x)  # [Batch, 32, Seq_Len_1]
        out_large = self.branch_large(x)  # [Batch, 32, Seq_Len_1]
        x = torch.cat([out_small, out_large], dim=1)  # [Batch, 64, Seq_Len_1]

        x = F.relu(self.bn_multi(x))
        x = self.pool_multi(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # 2. 深度 CNN
        x = self.cnn_deep(x)  # [Batch, 128, Seq_Len_Final]

        # 3. Bi-LSTM 时序建模
        # ⚠️ 注意：CNN 输出是 [Batch, Channels, Seq_Len]
        # LSTM 需要输入 [Batch, Seq_Len, Channels]，所以做一次 permute
        x = x.permute(0, 2, 1)

        # lstm_out 的 shape: [Batch, Seq_Len, 128]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 4. 注意力机制加权
        # 丢弃 h_n，用 Attention 评估 lstm_out 中每一步的价值
        final_state = self.attention(lstm_out)  # [Batch, 128]

        # 5. 分类
        out = self.classifier(final_state)
        return out

