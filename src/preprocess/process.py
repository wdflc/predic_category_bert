import datasets
from datasets import ClassLabel
from transformers import AutoTokenizer

from src.configuration import config


def process():
    # 读取数据
    dataset_dic = datasets.load_dataset('csv', data_files={
        'train': str(config.RAW_DATA_DIR / 'train.txt'),
        'test': str(config.RAW_DATA_DIR / 'test.txt'),
        'valid': str(config.RAW_DATA_DIR / 'valid.txt')
    }, delimiter='\t')

    # 过滤数据
    dataset_dic = dataset_dic.filter(lambda x: x['text_a'] is not None and x['label'] is not None)

    # 处理类别
    all_labels = sorted(set(dataset_dic['train']['label']))
    dataset_dic = dataset_dic.cast_column('label', ClassLabel(names=all_labels))

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))

    def tokenize(example):
        encoded = tokenizer(
            example['text_a'],
            truncation=True,
            padding='max_length',
            max_length=config.SEQ_LEN,
        )
        example['input_ids'] = encoded['input_ids']
        example['attention_mask'] = encoded['attention_mask']
        return example

    # 编码
    dataset_dic = dataset_dic.map(tokenize, batched=True, remove_columns=['text_a'])

    # 保存数据集
    dataset_dic['train'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'train'))
    dataset_dic['test'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'test'))
    dataset_dic['valid'].save_to_disk(str(config.PROCESSED_DATA_DIR / 'valid'))
