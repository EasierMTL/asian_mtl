# `asian_mtl`

This repository contains the code and documentation for the machine translation models used for EasierMTL's API.

Improved version of the models in the original repository: [EasierMTL/chinese-translation-app](https://github.com/EasierMTL/chinese-translation-app/tree/main/server/chinese_translation_api)

## Supported Translators

All translators support dynamic quantization! [Our benchmarks](#benchmarks) indicate that they 2x inference speeds, while losing <1% BLEU.

- `ChineseToEnglishTranslator()`
- `EnglishToChineseTranslator()`

## Getting Started

```bash
pip install asian-mtl
```

Here's a simple example:

```python
from asian_mtl.models.base import ChineseToEnglishTranslator

translator = ChineseToEnglishTranslator()
# Quantize for better CPU production performance!
translator.quantize()

prediction = translator.predict("我爱ECSE484.")
print(prediction)
# prediction will be:
# "I love ECSE 484."
```

And you're good to go!

If you are contributing, run:

```bash
# https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option

poetry config virtualenvs.in-project true

# shows the name of the current environment
poetry env list

poetry install
```

## Usage

When you are using quantized models in this repository, make sure to set `torch.set_num_threads(1)`. This is not set under-the-hood because it could interfere with user setups in an invasive way.

Not doing so will make the quantized models slower than their vanilla counterparts.

## Evaluation

See [`scripts`](./scripts) for evaluation scripts.

To run the scripts, simply run:

```bash
# Running with CLI and config with BERT
python ./scripts/evaluation/eval.py -c ./scripts/evaluation/configs/helsinki.yaml
```

Change the config [`helsinki.yaml`](./scripts/evaluation/configs/helsinki.yaml) to use quantized or your specific use case.

### Benchmarks

Here are some basic benchmarks of models in this repository:

| Model                      | Quantized? | N   | BLEU  | Runtime |
| -------------------------- | ---------- | --- | ----- | ------- |
| Helsinki-NLP/opus-mt-zh-en | No         | 100 | 0.319 | 27s     |
|                            | Yes        | 100 | 0.306 | 13.5s   |

The benchmarks described in the [docs](./docs/evaluation/EVALUATION_REG.md) are a little out-of-date.
