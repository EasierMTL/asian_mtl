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
# Running with CLI and config with BERT
python ./scripts/evaluation/eval.py -c ./scripts/evaluation/configs/helsinki.yaml
```

Change the config [`helsinki.yaml`](./scripts/evaluation/configs/helsinki.yaml) to use quantized or your specific use case.
