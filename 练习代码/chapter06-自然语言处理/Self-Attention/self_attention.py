import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from emo_utils import *

# 设置随机种子
torch.manual_seed(1)
np.random.seed(1)


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状: (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状: (batch_size, seq_len, d_model)
        # 使用前x.size(1)个位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        # 线性变换并分头
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_linear(output)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # 自注意力层
        src2 = self.self_attention(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SelfAttentionModel(nn.Module):
    """基于自注意力机制的情感分析模型"""

    def __init__(self, vocab_size, embedding_dim, word_to_vec_map, word_to_index, max_len, num_classes=5):
        super(SelfAttentionModel, self).__init__()

        self.original_embedding_dim = embedding_dim  # 保存原始维度
        self.max_len = max_len

        # 调整embedding_dim使其能被num_heads整除
        num_heads = 4
        if embedding_dim % num_heads != 0:
            self.embedding_dim = (embedding_dim // num_heads) * num_heads
            print(f"Adjusted embedding_dim from {embedding_dim} to {self.embedding_dim} to be divisible by {num_heads}")
        else:
            self.embedding_dim = embedding_dim

        # 初始化嵌入层 - 使用原始维度
        self.embedding = nn.Embedding(vocab_size, self.original_embedding_dim)

        # 加载预训练权重
        self._init_embedding_weights(word_to_vec_map, word_to_index, vocab_size)

        # 如果维度需要调整，添加投影层
        if self.original_embedding_dim != self.embedding_dim:
            self.projection = nn.Linear(self.original_embedding_dim, self.embedding_dim)
            print(f"Added projection layer from {self.original_embedding_dim} to {self.embedding_dim}")
        else:
            self.projection = nn.Identity()

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.embedding_dim, max_len)

        # Transformer编码器层
        self.transformer1 = TransformerEncoderLayer(self.embedding_dim, nhead=num_heads, dim_feedforward=128,
                                                    dropout=0.1)
        self.transformer2 = TransformerEncoderLayer(self.embedding_dim, nhead=num_heads, dim_feedforward=128,
                                                    dropout=0.1)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _init_embedding_weights(self, word_to_vec_map, word_to_index, vocab_size):
        """初始化嵌入层权重"""
        emb_matrix = np.zeros((vocab_size, self.original_embedding_dim))

        for word, index in word_to_index.items():
            if index < vocab_size and word in word_to_vec_map:
                emb_matrix[index, :] = word_to_vec_map[word]

        # 将numpy数组转换为torch tensor并设置为嵌入层权重
        self.embedding.weight.data = torch.FloatTensor(emb_matrix)
        # 冻结嵌入层，不进行训练
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        # 嵌入层 - 输出形状: (batch_size, seq_len, original_embedding_dim)
        x = self.embedding(x)

        # 投影到调整后的维度 - 输出形状: (batch_size, seq_len, embedding_dim)
        x = self.projection(x)

        # 位置编码
        x = self.pos_encoder(x)

        # Dropout
        x = self.dropout(x)

        # Transformer编码器
        x = self.transformer1(x)
        x = self.transformer2(x)

        # 全局平均池化
        x = torch.mean(x, dim=1)  # (batch_size, embedding_dim)

        # 分类
        x = self.classifier(x)

        return x


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    """
    m = X.shape[0]
    X_indices = np.zeros((m, max_len), dtype=np.int64)

    for i in range(m):
        sentence_words = (X[i].lower()).split()
        j = 0
        for w in sentence_words:
            if w in word_to_index and j < max_len:
                X_indices[i, j] = word_to_index[w]
                j += 1
    return X_indices


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """训练模型"""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return train_losses, val_accuracies


def evaluate_model(model, test_loader):
    """评估模型"""
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return all_predictions, all_labels, accuracy


if __name__ == "__main__":
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 读取数据
    X_train, Y_train = read_csv('train_emoji.csv')
    X_test, Y_test = read_csv('test_emoji.csv')
    maxLen = len(max(X_train, key=len).split())

    print(f"Max sentence length: {maxLen}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # 读取GloVe词向量
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

    # 词汇表大小（添加1是为了包含索引0）
    vocab_size = len(word_to_index) + 1
    original_embedding_dim = 50  # GloVe 50d的维度
    num_classes = 5

    # 将句子转换为索引
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)

    # 将标签转换为tensor（注意：PyTorch使用类别索引而不是one-hot）
    Y_train_tensor = torch.LongTensor(Y_train)
    Y_test_tensor = torch.LongTensor(Y_test)

    # 创建数据加载器
    train_dataset = TensorDataset(torch.LongTensor(X_train_indices), Y_train_tensor)
    test_dataset = TensorDataset(torch.LongTensor(X_test_indices), Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型
    model = SelfAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=original_embedding_dim,
        word_to_vec_map=word_to_vec_map,
        word_to_index=word_to_index,
        max_len=maxLen,
        num_classes=num_classes
    ).to(device)

    print("Model architecture:")
    print(model)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 训练模型
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, num_epochs=100)

    # 评估模型
    print("\nEvaluating model...")
    predictions, true_labels, test_accuracy = evaluate_model(model, test_loader)

    # 比较预测和期望的emoji
    print("\nMisclassified examples:")
    with torch.no_grad():
        X_test_tensor = torch.LongTensor(X_test_indices).to(device)
        test_outputs = model(X_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()

        misclassified_count = 0
        for i in range(len(X_test)):
            if test_predictions[i] != Y_test[i]:
                print(
                    f'Expected emoji: {label_to_emoji(Y_test[i])} prediction: {X_test[i]} {label_to_emoji(test_predictions[i]).strip()}')
                misclassified_count += 1
        print(f"Total misclassified: {misclassified_count}/{len(X_test)}")

    # 测试自定义句子
    print("\nTesting custom sentences:")
    test_sentences = ['very happy', 'I am so sad', 'I love you', 'This is terrible', 'Let us play baseball']

    for sentence in test_sentences:
        x_test = np.array([sentence])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
        X_test_tensor = torch.LongTensor(X_test_indices).to(device)

        with torch.no_grad():
            output = model(X_test_tensor)
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]

        print(f'{sentence} {label_to_emoji(prediction)}')

    # 保存模型
    torch.save(model.state_dict(), 'self_attention_model.pth')
    print("\nModel saved as 'self_attention_model.pth'")