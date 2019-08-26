from typing import Dict

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides


@Model.register("base")
class BaseModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        dropout_proba: float = 0.5,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: RegularizerApplicator = None,
    ) -> None:
        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        num_dense = encoder.get_output_dim() * 2
        self.projection = nn.Sequential(
            nn.BatchNorm1d(num_dense, num_dense),
            nn.Dropout(p=dropout_proba),
            nn.Linear(num_dense, num_dense),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_dense, num_dense),
            nn.Dropout(p=dropout_proba),
            nn.Linear(num_dense, 1),
            nn.Sigmoid(),
        )

        self.lossfun = nn.BCEWithLogitsLoss()
        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(
        self,
        question1: Dict[str, torch.Tensor],
        question2: Dict[str, torch.Tensor],
        pos_weight: torch.Tensor = None,
        label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        mask_q1 = get_text_field_mask(question1)
        mask_q2 = get_text_field_mask(question2)

        embed_q1 = self.text_field_embedder(mask_q1)
        embed_q2 = self.text_field_embedder(mask_q2)

        encoded_q1 = self.encoder(embed_q1)
        encoded_q2 = self.encoder(embed_q2)

        h = torch.cat((encoded_q1, encoded_q2), dim=1)
        logits = self.projection(h)

        output_dict = {"logits": logits}
        if label is not None:
            # label = label.float()
            bce_loss = self.lossfun(logits, label)
            weight = label * pos_weight.transpose(0, 1) + (1 - label)
            loss = (bce_loss * weight).mean(dim=1).sum()
            output_dict["loss"] = loss

            one_minus_logits = 1 - logits
            probs = torch.stack((one_minus_logits, logits), dim=-1)
            for metricfun in self.metrics.values():
                metricfun(probs, label)

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
