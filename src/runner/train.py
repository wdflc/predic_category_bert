import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.configuration import config
from src.model.classifier import BertTitleClassifier
from src.preprocess.dataset import get_dataloader, DataType


# 提前终止训练的工具类（EarlyStopping）
class EarlyStopping:
    def __init__(self, patience=2, path=None):
        """
  初始化早停机制。
  :param patience: 容忍验证集 loss 连续几轮不下降（超过即停止）
  :param path: 模型保存路径
  """
        self.patience = patience  # 允许的“无改进”轮数
        self.counter = 0  # 当前连续“无改进”轮数计数器
        self.best_score = None  # 当前最佳得分（验证 loss 的负数）
        self.early_stop = False  # 是否触发早停
        self.path = path  # 最佳模型保存路径

    def __call__(self, val_loss, model):
        """
  每轮验证后调用，判断是否早停
  """
        score = -val_loss  # 损失越小越好，取负值用于比较
        if self.best_score is None or score > self.best_score:
            # 当前是最优模型，更新并保存
            self.best_score = score
            self.counter = 0
            self.save_model(model)
        else:
            # 模型未提升，计数器加 1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        """保存当前模型权重"""
        torch.save(model.state_dict(), self.path)


def run_one_epoch(model, dataloader, device, loss_fn, optimizer=None, is_train=True):
    """
 执行一次训练轮次（epoch），同时支持训练和验证模式
 """
    epoch_loss = 0
    model.train() if is_train else model.eval()

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="训练" if is_train else "验证"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            # 训练模式：反向传播 + 更新参数
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train():
    """
 模型训练主函数
 """
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'训练设备: {device}')

    # 日志
    log_dir = config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    # 模型
    model = BertTitleClassifier(freeze_bert=False).to(device)

    # 数据
    train_loader = get_dataloader(DataType.TRAIN)
    valid_loader = get_dataloader(DataType.VALID)

    # 优化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 早停
    early_stopping = EarlyStopping(
        patience=2,
        path=config.MODELS_DIR / 'model.pt'
    )

    # 开始训练
    for epoch in range(1, config.EPOCHS + 1):
        print(f'\n========== Epoch {epoch} ==========')

        # 训练
        train_loss = run_one_epoch(model, train_loader, device, loss_fn, optimizer, is_train=True)

        # 验证
        valid_loss = run_one_epoch(model, valid_loader, device, loss_fn, is_train=False)

        # 打印
        print(f"训练 Loss: {train_loss:.4f}")
        print(f"验证 Loss: {valid_loss:.4f}")

        # 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        # 早停
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("✅ 早停触发，训练结束")
            break

    writer.close()
