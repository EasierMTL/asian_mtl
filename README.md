# `novel_tl_models`

This repository contains the code and documentation for the models used for EasierMTL's API.

## Supported Translators

- `ChineseToEnglishTranslator()`
- `EnglishToChineseTranslator()`

## Getting Started

```bash
# https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option

poetry config virtualenvs.in-project true

# shows the name of the current environment
poetry env list

poetry install
```

## Evaluation

See [`scripts`](./scripts) for evaluation scripts.

To run the scripts, simply run:

```bash
# regular
python ./scripts/evaluate.py

# evaluate quantized
python ./scripts/evaluate_quantized.py

# Running with CLI and config
python ./scripts/evaluation/eval.py -c ./scripts/evaluation/configs/helsinki.yaml
```
