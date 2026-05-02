import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.configuration import config
from src.model.classifier import BertTitleClassifier
from src.runner.predict import predict_text

# 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'设备：{device}')

# 模型加载
print('加载模型...')
model = BertTitleClassifier().to(device)
model.load_state_dict(torch.load(
 config.MODELS_DIR / "model.pt",
 map_location=device
))
model.eval()
print('模型加载完成！')

# Tokenizer加载
print('加载 Tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
 str(config.PRE_TRAINED_DIR / "bert-base-chinese")
)
print('Tokenizer 加载完成！')

# 标签映射加载
label_feature = load_from_disk(
 str(config.PROCESSED_DATA_DIR / 'train')
).features['label']

def predict_service(text):
 return predict_text(text, model, tokenizer, device, label_feature)
