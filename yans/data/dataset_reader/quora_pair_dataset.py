import logging
from typing import Dict, Iterator

import numpy as np
import pandas as pd
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register("quora")
class QuoraPairDataset(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy=False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.pos_weight = None

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        logger.info(f"Reading instances from: {file_path}")

        df = pd.read_csv(file_path)
        df = df.dropna()

        y = df.is_duplicate
        self.pos_weight = np.asarray([len(y) / sum(y) - 1])

        logger.info(f"Class weight: {self.pos_weight}")

        for i in range(len(df)):
            dfi = df.iloc[i]
            yield self.text_to_instance(
                dfi.question1, dfi.question2, str(dfi.is_duplicate)
            )

    @overrides
    def text_to_instance(self, q1: str, q2: str, target: str = None) -> Instance:

        tokenized_q1 = self._tokenizer.tokenize(q1)
        tokenized_q2 = self._tokenizer.tokenize(q2)

        fields = {
            "question1": TextField(tokenized_q1, self._token_indexers),
            "question2": TextField(tokenized_q2, self._token_indexers),
        }
        fields["pos_weight"] = ArrayField(self.pos_weight)
        if target is not None:
            fields["label"] = LabelField(target)

        return Instance(fields)
