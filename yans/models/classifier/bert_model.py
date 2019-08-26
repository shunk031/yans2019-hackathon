from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.token_embedders.bert_token_embedder import (
    PretrainedBertModel
)
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel


@Model.register("bert_based_model")
class BertModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        dropout: float = 0.0,
        index: str = "bert",
        trainable: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self.bert_model = self.init_pert_model(bert_model, trainable)

        num_dense = self.bert_model.config.hidden_size * 2
        self._classification_layer = nn.Sequential(
            nn.BatchNorm1d(num_dense, num_dense),
            nn.Dropout(p=dropout),
            nn.Linear(num_dense, num_dense),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_dense, num_dense),
            nn.Dropout(p=dropout),
            nn.Linear(num_dense, 1),
            nn.Sigmoid(),
        )
        self.lossfun = nn.BCEWithLogitsLoss()
        self.metrics = {"accuracy": CategoricalAccuracy()}
        self._index = index

        initializer(self)

    def init_pert_model(self, bert_model, trainable) -> Model:
        if isinstance(bert_model, str):
            bert_model = PretrainedBertModel.load(bert_model)

        for param in bert_model.parameters():
            param.requires_grad = trainable

        return bert_model

    @overrides
    def forward(
        self,
        question1: Dict[str, torch.Tensor],
        question2: Dict[str, torch.Tensor],
        pos_weight: torch.Tensor = None,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        input_q1_ids = question1[self._index]
        input_q2_ids = question2[self._index]

        token_type_q1_ids = question1[f"{self._index}-type-ids"]
        token_type_q2_ids = question2[f"{self._index}-type-ids"]

        mask_q1 = (input_q1_ids != 0).long()
        mask_q2 = (input_q2_ids != 0).long()

        _, pooled_q1 = self.bert_model(
            input_ids=input_q1_ids,
            token_type_ids=token_type_q1_ids,
            attention_mask=mask_q1,
        )
        _, pooled_q2 = self.bert_model(
            input_ids=input_q2_ids,
            token_type_ids=token_type_q2_ids,
            attention_mask=mask_q2,
        )
        h = torch.cat((pooled_q1, pooled_q2), dim=1)
        logits = self._classification_layer(h)

        output_dict = {"logits": logits}
        if label is not None:
            label = label.float()

            bce_loss = self.lossfun(logits, label.unsqueeze(dim=-1))
            weight = label * pos_weight.transpose(0, 1) + (1 - label)
            loss = (bce_loss * weight).mean(dim=1).sum()
            output_dict["loss"] = loss

            one_minus_logits = 1 - logits
            probs = torch.stack((one_minus_logits, logits), dim=-1)
            for metricfun in self.metrics.values():
                metricfun(probs, label.unsqueeze(dim=-1))

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metric_scores = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        for key, value in metric_scores.items():
            if isinstance(metric_scores[key], tuple):
                # f1 get_metric returns (precision, recall, f1)
                metric_scores[key] = metric_scores[key][2]

        return metric_scores
