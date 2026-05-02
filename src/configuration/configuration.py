from pathlib import Path

# 项目根目录（自动定位到 project_classify_bert 文件夹）
BASE_DIR = Path(__file__).parent.parent.absolute()

# 数据相关路径
RAW_DATA_DIR = BASE_DIR / "data"          # 存放 train.txt / test.txt / valid.txt
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"  # 存放处理好的数据集

# 预训练模型路径
PRE_TRAINED_DIR = BASE_DIR / "pretrained"  # 存放 bert-base-chinese 等预训练模型

# 模型与日志路径
MODELS_DIR = BASE_DIR / "models"           # 存放训练好的模型、checkpoint
LOGS_DIR = BASE_DIR / "logs"               # 存放 TensorBoard 日志

# 训练超参数
SEQ_LEN = 64                               # 文本最大长度
BATCH_SIZE = 32                            # 批次大小
LEARNING_RATE = 2e-5                       # 学习率（BERT 微调推荐值）
EPOCHS = 10                                # 最大训练轮数
PATIENCE = 2                               # 早停容忍轮数

# 设备配置
DEVICE = "cuda" if Path.cwd().exists("/dev/nvidia0") else "cpu"

# 自动创建目录（防止运行时报错）
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PRE_TRAINED_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)