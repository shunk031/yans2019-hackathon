# YANS2019 Hackathon

```shell
allennlp train config/train_bert_model.json --include-package yans -s output -o '{"trainer": {"cuda_device": 0}}' --force
```
