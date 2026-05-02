import time
import torch
from torch import nn, GradScaler
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
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.save_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model(self, model):
        torch.save(model.state_dict(), self.path)


def run_one_epoch(model, dataloader, device, loss_fn, scaler, optimizer=None, is_train=True):
    epoch_loss = 0.0
    model.train() if is_train else model.eval()

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, desc="训练" if is_train else "验证"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 混合精度前向传播
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

            # 训练阶段：反向传播
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'训练设备: {device}')

    log_dir = config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    model = BertTitleClassifier(freeze_bert=False).to(device)

    train_loader = get_dataloader(DataType.TRAIN)
    valid_loader = get_dataloader(DataType.VALID)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = GradScaler()

    early_stopping = EarlyStopping(
        patience=2,
        path=config.MODELS_DIR / 'model.pt'
    )

    start_epoch = 1
    checkpoint_path = config.MODELS_DIR / 'checkpoint.pt'

    # 断点续训
    if checkpoint_path.exists():
        print(f"检测到断点，恢复训练中...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        early_stopping.best_score = checkpoint.get('early_stopping_best_score', None)
        early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
        start_epoch = checkpoint['epoch'] + 1
        print(f"从 epoch {start_epoch} 继续训练")
    else:
        print("未检测到断点，从头开始训练")

    # 正式训练循环
    for epoch in range(start_epoch, config.EPOCHS + 1):
        print(f'\n========== Epoch {epoch} ==========')

        # 训练
        train_loss = run_one_epoch(model, train_loader, device, loss_fn, scaler, optimizer, is_train=True)
        # 验证
        valid_loss = run_one_epoch(model, valid_loader, device, loss_fn, scaler, is_train=False)

        print(f"训练 Loss: {train_loss:.4f}")
        print(f"验证 Loss: {valid_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        # 早停
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("✅ 早停触发，训练结束")
            break

        # 保存断点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'early_stopping_best_score': early_stopping.best_score,
            'early_stopping_counter': early_stopping.counter,
        }
        torch.save(checkpoint, checkpoint_path)

    writer.close()
