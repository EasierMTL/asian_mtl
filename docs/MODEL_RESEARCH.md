# Model Research <!-- omit from toc -->

## Table Of Contents <!-- omit from toc -->

- [BERT](#bert)
- [RoBERTa](#roberta)
- [XLNet](#xlnet)
- [DistilBERT](#distilbert)
- [ALBERT](#albert)
- [Resources](#resources)
- [Interesting Questions](#interesting-questions)
  - [Which lightweight BERT model would you recommend with TensorFlow.js on React Native?](#which-lightweight-bert-model-would-you-recommend-with-tensorflowjs-on-react-native)
  - [\[D\] Why does BERT perform so well?](#d-why-does-bert-perform-so-well)

## BERT

- self-supervised training on masked language modeling and next sentence prediction tasks to learn contextual representations of words.
  - **Masked language modeling:** "fill in the blanks" task
    - Using context around blank to guess what the blank token is (use mask to create blanks in samples)
- **Architecture**
  - Builds on encoder-decoder architecture of transformers
  - Stacks 12 encoder blocks (bi-directional encoders)
  - Adds extra linear layers on top of the encoder blocks to make fine-tuning easier
    - Basically the same for fine-tuning image classification models and CNNs
- **Usage (Basic)**

  ```python
  from transformers import BertModel
  class Bert_Model(nn.Module):
     def __init__(self, class):
         super(Bert_Model, self).__init__()
         self.bert = BertModel.from_pretrained('bert-base-uncased')
         self.out = nn.Linear(self.bert.config.hidden_size, classes)
     def forward(self, input):
         _, output = self.bert(**input)
         out = self.out(output)
         return out
  ```

## RoBERTa

- Same architecture as BERT but better trained
  - Pretrained with masked language modeling instead of both MLM and next sentence prediction (BERT)
  - Longer training time and larger training data
  - Larger batch size (256 to 8000)
  - Larger vocab size (30,000 to 50,000)
  - Longer sequences used as input, but has same max 512 tokens cap
  - Dynamic masking, which made masking pattern different everytime a sequence was fed into the model
    - BERT used a static masking pattern
- **Usage:**
  ```python
  from transformers import RobertaModel
  import torch
  import torch.nn as nn
  class RoBERTa_Model(nn.Module):
    def __init__(self, classes):
      super(RoBERTa_Model, self).__init__()
      self.roberta = RobertaModel.from_pretrained('roberta-base')
      self.out = nn.Linear(self.roberta.config.hidden_size, classes)
      self.sigmoid = nn.Sigmoid()
    def forward(self, input, attention_mask):
      _, output = self.roberta(input, attention_mask = attention_mask)
      out = self.sigmoid(self.out(output))
      return out
  ```

## XLNet

- Published around the same time as RoBERTa, but signficantly harder to implement changes than RoBERTa

## DistilBERT

- Reduces size of BERT and increases the speed of BERT, while being as performant as possible
  - 40% smaller, 60% faster, and retrains 97% functionality
- Uses 6 encoder blocks instead of 12
  - Initialized with 1/2 of each **pretrained** encoder block in BERT
- Removed token type and pooling functionalities from BERT
  - Why?
- Trained with only masked language modeling instead of both MLM and next sentence prediction
  - And other same procedures as RoBERTa
- Uses a triple loss:
  - Same language model loss used by BERT
  - Distillation loss measures the similarity of the outputs of DistilBERT and BERT
  - Cosine distance measures how similar the hidden states of DistilBERT and BERT are
- DistilBERT acts as the student learning from BERT
- **Usage:**
  ```python
  from transformers import DistilBertModel
  import torch
  import torch.nn as nn
  class DistilBERT_Model(nn.Module):
   def __init__(self, classes):
     super(DistilBERT_Model, self).__init__()
     self.distilbert = DistilBertModel.from_pretrained('distilbert
                                                       base-uncased')
     self.out = nn.Linear(self.distilbert.config.hidden_size, classes)
     self.sigmoid = nn.Sigmoid()
   def forward(self, input, attention_mask):
     _, output = self.distilbert(input, attention_mask
                                        = attention_mask)
     out = self.sigmoid(self.out(output))
     return out
  ```

## ALBERT

- Reduces model size of BERT by 18x and can be trained 1.7x faster
- No tradeoff in performance
  - ALBERT trained from scratch while DistilBERT relies on learning from BERT
  - ALBERT outperforms all previous models
- Uses these parameter reduction techniques:
  - **Factorized Embedding Parameterization:** By separating hidden layer params from embedding params, we can increase the hidden layer size without also having to increase the embedding matrix size.
    - See more [here](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#albert)
  - **Cross-Layer Parameter Sharing:** Parameters are shared across all 12 encoder blocks
    - Regularizes model
    - Reduces parameters by 12x
    - **Reasoning:** The model often performed similar operations different layers and there has been some research showing that there is redundancy in attention heads.
      - See [this stack exchange post](https://stats.stackexchange.com/questions/446594/cross-layer-parameter-sharing-in-albert-model) for more information.
      - **How did they detect redundant nodes in neural networks?**
        - Pruned --> tested if removing the nodes impacted accuracy
  - Remove dropout (balances regularization caused by cross-layer parameter sharing)
- **Usage:**

  ```python
  from transformers import AlbertModel
  import torch
  import torch.nn as nn
  class ALBERT_Model(nn.Module):
   def __init__(self, classes):
     super(ALBERT_Model, self).__init__()
     self.albert = AlbertModel.from_pretrained('albert-base-v2')
     self.out = nn.Linear(self.albert.config.hidden_size, classes)
     self.sigmoid = nn.Sigmoid()
   def forward(self, input, attention_mask):
     _, output = self.albert(input, attention_mask = attention_mask)
     out = self.sigmoid(self.out(output))
     return out
  ```

## Resources

- [Everything you need to know about ALBERT, RoBERTa, and DistilBERT](https://towardsdatascience.com/everything-you-need-to-know-about-albert-roberta-and-distilbert-11a74334b2da)
  - July 2022
- [A review of pre-trained language models: from BERT, RoBERTa, to ELECTRA, DeBERTa, BigBird, and more](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#albert)
  - December 2021

## Interesting Questions

### Which lightweight BERT model would you recommend with TensorFlow.js on React Native?

https://www.reddit.com/r/deeplearning/comments/pjk32s/which_lightweight_bert_model_would_you_recommend/

General Consensus:

- DistilBERT
- ALBERT
- TinyBERT

### [D] Why does BERT perform so well?

https://www.reddit.com/r/MachineLearning/comments/k4saj5/d_why_does_bert_perform_so_well/
