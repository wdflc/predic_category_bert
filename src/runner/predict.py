import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.configuration import config
from src.model.classifier import BertTitleClassifier


def predict_batch(model, input_ids, attention_mask):
    """
    对一批输入进行预测，返回类别索引和置信度。

    :param model: 已加载好的分类模型
    :param input_ids: 输入 token 的 id（tensor，shape: [batch_size, seq_len]）
    :param attention_mask: 注意力 mask（tensor，shape 同上）
    :return: (预测的类别索引, 置信度) 两个 tensor
    """
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, dim=1)
    return preds, confidences


def predict_text(text, model, tokenizer, device, label_feature):
    """
    对单条文本进行预测，返回类别 ID、标签名和置信度。

    :param text: 输入文本字符串
    :param model: 已加载好的模型
    :param tokenizer: 用于编码输入文本的 tokenizer
    :param device: 使用的计算设备
    :param label_feature: label 特征映射（HuggingFace 的 ClassLabel 对象）
    :return: (类别 ID, 类别名称, 置信度)
    """
    # 文本编码成输入格式（padding、截断等）
    encoded = tokenizer(
        [text],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=config.SEQ_LEN
    )

    # 将输入数据移动到计算设备
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # 调用批量预测函数
    preds, confidences = predict_batch(model, input_ids, attention_mask)

    # 转换为类别 id 和标签名
    pred_id = preds[0].item()
    pred_label = label_feature.int2str(pred_id)
    confidence = confidences[0].item()

    return pred_id, pred_label, confidence


def run_predict():
    """
    命令行交互式预测函数，用户输入标题，输出预测分类。
    """
    # 初始化设备、模型、tokenizer、标签映射
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载已训练好的模型权重
    model = BertTitleClassifier().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))
    model.eval()

    # 加载 tokenizer 和标签信息
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))
    label_feature = load_from_disk(str(config.PROCESSED_DATA_DIR / 'train')).features['label']

    # 命令行交互
    print("请输入商品标题（输入 q 或 quit 退出）：")
    while True:
        text = input("> ").strip()
        if text.lower() in ['q', 'quit']:
            break
        if not text:
            continue

        # 调用预测函数
        pred_id, pred_label, confidence = predict_text(text, model, tokenizer, device, label_feature)
        print(f"预测类别ID: {pred_id}，类别名称: {pred_label}，置信度: {confidence:.4f}")
