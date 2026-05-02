import time
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.configuration import config
from src.model.classifier import BertTitleClassifier
from src.preprocess.dataset import get_dataloader, DataType


def train_one_epoch(model, dataloader, device, optimizer, loss_fn):
    """
    执行一次训练轮次（epoch）

    :param model: 当前模型
    :param dataloader: 训练数据加载器
    :param device: 使用的设备（CPU/GPU）
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :return: 平均训练损失
    """
    model.train()  # 切换到训练模式
    epoch_loss = 0

    for batch in tqdm(dataloader, desc="训练"):
        # 数据移至设备
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 前向传播 + 损失计算
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积损失
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)  # 返回平均损失


def train():
    """
    模型训练主函数：
    - 初始化模型和优化器
    - 执行多轮训练与验证
    - 使用 EarlyStopping 保存最佳模型
    - 写入 TensorBoard 日志
    """
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 设置 TensorBoard 日志目录
    log_dir = config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    # 初始化模型（不冻结 BERT 参数）
    model = BertTitleClassifier(freeze_bert=False).to(device)

    # 加载训练和验证数据
    train_loader = get_dataloader(DataType.TRAIN)

    # 设置损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 训练过程
    best_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== Epoch {epoch} ==========")

        # 训练
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)

        # 打印训练与验证信息
        print(f"训练集-loss: {train_loss:.4f}")

        # 记录日志
        writer.add_scalar("Loss/train", train_loss, epoch)

        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), config.MODELS_DIR / "model.pt")

    # 关闭日志记录器
    writer.close()
