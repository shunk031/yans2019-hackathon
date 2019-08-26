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


# @Model.register("bert_for_my_classification")
# class BertForClassification(Model):
#     """
#     An AllenNLP Model that runs pretrained BERT,
#     takes the pooled output, and adds a Linear layer on top.
#     If you want an easy way to use BERT for classification, this is it.
#     Note that this is a somewhat non-AllenNLP-ish model architecture,
#     in that it essentially requires you to use the "bert-pretrained"
#     token indexer, rather than configuring whatever indexing scheme you like.
#     See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
#     for an example of what your config might look like.
#     Parameters
#     ----------
#     vocab : ``Vocabulary``
#     bert_model : ``Union[str, BertModel]``
#         The BERT model to be wrapped. If a string is provided, we will call
#         ``BertModel.from_pretrained(bert_model)`` and use the result.
#     num_labels : ``int``, optional (default: None)
#         How many output classes to predict. If not provided, we'll use the
#         vocab_size for the ``label_namespace``.
#     index : ``str``, optional (default: "bert")
#         The index of the token indexer that generates the BERT indices.
#     label_namespace : ``str``, optional (default : "labels")
#         Used to determine the number of classes if ``num_labels`` is not supplied.
#     trainable : ``bool``, optional (default : True)
#         If True, the weights of the pretrained BERT model will be updated during training.
#         Otherwise, they will be frozen and only the final linear layer will be trained.
#     initializer : ``InitializerApplicator``, optional
#         If provided, will be used to initialize the final linear layer *only*.
#     regularizer : ``RegularizerApplicator``, optional (default=``None``)
#         If provided, will be used to calculate the regularization penalty during training.
#     """

#     def __init__(
#         self,
#         vocab: Vocabulary,
#         bert_model: Union[str, BertModel],
#         dropout: float = 0.0,
#         num_labels: int = None,
#         index: str = "bert",
#         label_namespace: str = "labels",
#         trainable: bool = True,
#         initializer: InitializerApplicator = InitializerApplicator(),
#         regularizer: Optional[RegularizerApplicator] = None,
#     ) -> None:
#         super().__init__(vocab, regularizer)

#         if isinstance(bert_model, str):
#             self.bert_model = PretrainedBertModel.load(bert_model)
#         else:
#             self.bert_model = bert_model

#         for param in self.bert_model.parameters():
#             param.requires_grad = trainable

#         in_features = self.bert_model.config.hidden_size

#         self._label_namespace = label_namespace

#         if num_labels:
#             out_features = num_labels
#         else:
#             out_features = vocab.get_vocab_size(namespace=self._label_namespace)

#         self._dropout = torch.nn.Dropout(p=dropout)

#         self._classification_layer = torch.nn.Linear(in_features, out_features)
#         self._accuracy = CategoricalAccuracy()
#         self._loss = torch.nn.CrossEntropyLoss()
#         self._index = index
#         initializer(self._classification_layer)

#     def forward(
#         self,  # type: ignore
#         tokens: Dict[str, torch.LongTensor],
#         label: torch.IntTensor = None,
#     ) -> Dict[str, torch.Tensor]:
#         # pylint: disable=arguments-differ
#         """
#         Parameters
#         ----------
#         tokens : Dict[str, torch.LongTensor]
#             From a ``TextField`` (that has a bert-pretrained token indexer)
#         label : torch.IntTensor, optional (default = None)
#             From a ``LabelField``
#         Returns
#         -------
#         An output dictionary consisting of:
#         logits : torch.FloatTensor
#             A tensor of shape ``(batch_size, num_labels)`` representing
#             unnormalized log probabilities of the label.
#         probs : torch.FloatTensor
#             A tensor of shape ``(batch_size, num_labels)`` representing
#             probabilities of the label.
#         loss : torch.FloatTensor, optional
#             A scalar loss to be optimised.
#         """
#         input_ids = tokens[self._index]
#         token_type_ids = tokens[f"{self._index}-type-ids"]
#         input_mask = (input_ids != 0).long()

#         _, pooled = self.bert_model(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#             attention_mask=input_mask,
#         )

#         pooled = self._dropout(pooled)

#         # apply classification layer
#         logits = self._classification_layer(pooled)

#         probs = torch.nn.functional.softmax(logits, dim=-1)

#         output_dict = {"logits": logits, "probs": probs}

#         if label is not None:
#             loss = self._loss(logits, label.long().view(-1))
#             output_dict["loss"] = loss
#             self._accuracy(logits, label)

#         return output_dict
