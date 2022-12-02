# Machine Translation

Focusing on the best models and available resources for creating Chinese to English Machine Translation models.

## Available Resources

- MarianMT: Microsoft's TL framework
- OpenNMT: Harvard NLP
- Sockeye: seq2seq; Amazon Translate
- Fairseq: Facebook's TL framework
  - https://github.com/facebookresearch/fairseq
- Huggingface

Only Hugggingface has a lot of pretrained models.

## Helsinki NLP

- Large public set of translation models published to Huggingface.
- Trained using MarianMT on the OPUS dataset

## mBART

Seems like a pretty legit option.

https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

https://github.com/hyunwoongko/asian-bart

https://huggingface.co/docs/transformers/model_doc/mbart

## M2M100

https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/m2m_100#transformers.M2M100ForConditionalGeneration.forward.example

Machine translation, focused on translation between non-English languages.

## Multilingual models for inference

https://huggingface.co/docs/transformers/v4.24.0/en/multilingual

M2M and mBART are the main ones.

## DistilBERT

https://huggingface.co/distilbert-base-multilingual-cased

The model has 6 layers, 768 dimension and 12 heads, totalizing 134M parameters (compared to 177M parameters for mBERT-base). On average, this model, referred to as DistilmBERT, is twice as fast as mBERT-base.

**Used for masked language modeling :(**

Would need to retrain it for machine translation usage.

## `dl-translate`

https://github.com/xhluca/dl-translate

Basically the same as Huggingface tbh.
