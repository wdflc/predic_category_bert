from enum import StrEnum

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.configuration import config
from src.preprocess.dataset import get_dataloader, DataType
from src.model.classifier import BertTitleClassifier


class Metric(StrEnum):
    ACCURACY = "accuracy"  # 准确率
    PRECISION = "precision"  # 精确率
    RECALL = "recall"  # 召回率
    F1 = "f1"  # F1 分数

def evaluate_model(model, dataloader, device, metrics):
        """
        对模型在指定数据上进行评估，返回指定指标的值以及平均损失（可选）。
       
        :param model: 要评估的模型
        :param dataloader: 用于加载评估数据的 DataLoader
        :param device: 使用的计算设备（CPU 或 GPU）
        :param metrics: 需要评估的指标列表（Metric 枚举值）
        :return: 包含各个评估指标的 dict，如果传入 loss_fn，还会返回平均 loss
        """
        model.eval()  # 设置为评估模式，关闭 dropout 等
        all_preds = []  # 所有预测结果
        all_labels = []  # 所有真实标签

        # 遍历整个数据集
        for batch in tqdm(dataloader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():  # 禁用梯度计算，节省显存和计算
                outputs = model(input_ids, attention_mask)  # 模型输出 logits
            preds = torch.argmax(outputs, dim=1)  # 取最大概率对应的类别作为预测结果

            # 收集预测值和真实标签，用于后续评估
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        # 初始化评估结果字典
        results = {}

        # 逐一计算指定的指标
        for metric in metrics:
            if metric == Metric.ACCURACY:
                results[Metric.ACCURACY] = accuracy_score(all_labels, all_preds)
            elif metric == Metric.F1:
                results[Metric.F1] = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            elif metric == Metric.PRECISION:
                results[Metric.PRECISION] = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            elif metric == Metric.RECALL:
                results[Metric.RECALL] = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        return results

def run_evaluation():
        """
        载入模型并在测试集上进行评估，输出各项指标的结果。
        """
        # 选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型并加载已训练权重
        model = BertTitleClassifier().to(device)
        model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

        # 获取测试数据加载器
        dataloader = get_dataloader(DataType.TEST)

        # 指定要评估的指标
        metrics = [Metric.ACCURACY, Metric.F1, Metric.PRECISION, Metric.RECALL]

        # 调用评估函数
        results = evaluate_model(model, dataloader, device, metrics)

        # 打印评估结果
        print("======= 评估结果 =======")
        for name, value in results.items():
            print(f"{name}: {value:.4f}")
        print("========================")
