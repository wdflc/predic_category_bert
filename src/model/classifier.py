from torch import nn
from transformers import AutoModel

from src.configuration import config


class BertTitleClassifier(nn.Module):
    def __init__(self, freeze_bert=True):
        """
        初始化分类模型。
       :param freeze_bert: 是否冻结 BERT 编码器的参数（默认冻结，仅训练分类器）
        """
        super().__init__()

        # 加载预训练的 BERT 模型
        self.bert = AutoModel.from_pretrained(str(config.PRE_TRAINED_DIR / 'bert-base-chinese'))

        # 定义分类器：将 BERT 输出的 CLS 向量映射到类别空间
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)

        # 决定是否冻结 BERT 参数（只训练最后的分类层）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播过程。

        :param input_ids: 输入 token 的 id 序列（batch_size, seq_len）
        :param attention_mask: 注意力掩码（同 shape），标识 padding 的位置
        :return: 分类结果的 logits（batch_size, num_classes）
        """
        # 获取 BERT 输出：last_hidden_state 为所有 token 的表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 取 [CLS] token 对应的表示，作为句子的整体表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)

        # 通过线性分类层得到最终 logits
        logits = self.classifier(cls_output)  # shape: (batch_size, num_classes)

        return logits
