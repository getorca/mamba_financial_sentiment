# Mamba Financial Sentiment

[Model](https://huggingface.co/winddude/mamba_financial_headline_sentiment) | [Dataset](https://huggingface.co/datasets/winddude/finacial_pharsebank_66agree_split)

This is the example code for [Mamba Financial Headline Sentiment Classifier
](https://huggingface.co/winddude/mamba_financial_headline_sentiment) a sentiment classifier for finacial headlines using mamba 2.8b as the base model. Trained on [our selection of FinacialPhrasebank](https://huggingface.co/datasets/winddude/finacial_pharsebank_66agree_split)

## Usage

Examples for running the model are found in:

-
-
-

The basic prompt format is as follows:

```
prompt = f"""Classify the setiment of the following news headlines as either `positive`, `neutral`, or `negative`.\n
  Headline: {headline}\n
  Classification:"""
```

## Evaluation

Although many models are trained on [FinacialPhrasebank](https://huggingface.co/datasets/financial_phrasebank) the orginal dataset is not split into test and train datasets, and it's highly likely most models have been trained on the entire dataset or parts in our [test split](https://huggingface.co/datasets/winddude/finacial_pharsebank_66agree_split). The basic evaluation metrics are good for compairing models trained on our dataset.

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| [Mamba Financial Headline Sentiment Classifier](https://huggingface.co/winddude/mamba_financial_headline_sentiment) | 0.82 | 0.82 | 0.82 | 0.82 |


## Citations
```
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

```
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```