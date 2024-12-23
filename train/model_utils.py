import torch
from transformers import AutoModel
from torch import nn


class CustomSequenceClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(CustomSequenceClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}
