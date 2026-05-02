from enum import StrEnum
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.configuration import config


# 数据类型枚举
class DataType(StrEnum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


def get_dataset(type):
    dataset = load_from_disk(str(config.PROCESSED_DATA_DIR / type))
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])   # 设置为 torch 格式
    return dataset


def get_dataloader(type=DataType.TRAIN):
    dataset = get_dataset(type)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
