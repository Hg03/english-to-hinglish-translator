# English to Hinglish Translator 🎚️🎚️

## Brief 🚀

As according to [Assignment 3 📰](https://ansoncareers.notion.site/AI-ML-Challenge-cc150a48f27f487ab81ba8054c9bd5dd), we need to convert **english** sentences to **hinglish (hindi + english)** sentences. As we know, In India 🇮🇳, most people are not communicating 🗣️ in pure english or pure hindi. They are mixing both language and this is even suitable for everybody. Here we are building Machine learning model which takes `english sentences` as an **input** 🔤 and generates output in `hinglish sentences` as an **output** 🔣. 

For e.g. 
|English|Hinglish|
|-------|--------|
|Definitely share your feedback in the comment section|aapka feedback zarur share karna comment section me|
|So even its a big video, I will clearly mention all the products|Bhale hi video badi ho, me saare products mention karuga|
|I was waiting for my bag|Mein mere bag ka wait kar raha tha|

## Approach 🚀

Here, we approach this problem using finetuning very popular **transformer model** named **T5**. T5, which stands for "Text-to-Text Transfer Transformer," is a versatile and powerful 🔋🔋 natural language processing (NLP) model developed by Google AI 😶‍🌫️😶‍🌫️. It was introduced in the paper titled "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" by Colin Raffel et al. The T5 model is part of the Transformer architecture family and is designed to handle various NLP tasks in a unified way, framing them all as text-to-text tasks.

![t5](https://github.com/Hg03/english-to-hinglish-translator/assets/69637720/254650b0-e180-41c9-b54a-4fe7f040f472)

So we have a dataset on huggingface having different rows consisting **english** sentences and its corresponding **hinglish** sentences. You can view it [here](https://huggingface.co/datasets/harish03/english_hinglist_sentences). Then we'll finetune our **T5 transformer** model on this dataset using some parameters like epochs etc.

![Streamlit(1)](https://github.com/Hg03/english-to-hinglish-translator/assets/69637720/67c6fdbe-16b7-434d-9daa-62a43f242c57)

## Code walkthrough 🚀

- To have a detailed look , view the [notebook](https://github.com/Hg03/english-to-hinglish-translator/blob/main/experimentation_notebook/english-to-hinglish-translation.ipynb) in my repo.

## Workflow

### Import and load essential libraries

```python
# Installation
!pip install transformers[sentencepiece] sacrebleu datasets -q
```

```python
# Import 
import os
import sys
import transformers
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import AdamWeightDecay
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
```

### Load the dataset

```python
raw_datasets = load_dataset("harish03/english_hinglist_sentences",split='train') # loading
dataset = raw_datasets.train_test_split(test_size=0.3) # Splitting
```

### Create instance of tokenizer and preprocess the data before passing it to the finetuning process

```python
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # loading the tokenizer
def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True) # Encoding the dataset
```

### Load the model and configure it corresponding to tokenizer using datacollator

```python
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 2
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)

```

### Generate the dataset suitable to pass to the model 

```python
train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)
```

### Initialize the optimizer

```python
optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)
```

### Finally fit the model and save it

```python
model.fit(train_dataset, validation_data=validation_dataset, epochs=num_train_epochs)
model.save_pretrained("model/")
```

### Now test the model

```python
def generate_output(input_text):
    tokenized = tokenizer([input_text], return_tensors='np')
    out = model.generate(**tokenized, max_length=128)
    with tokenizer.as_target_tokenizer():
        return tokenizer.decode(out[0], skip_special_tokens=True)

texts  = ["Definitely share your feedback in the comment section",
          "So even it's a big video, I will clearly mention all the products",
          "I was waiting for my bag"
         ]

for input_text in texts:
    print(generate_output(input_text))
```

### Output for example text
![Screenshot from 2023-09-13 18-47-48](https://github.com/Hg03/english-to-hinglish-translator/assets/69637720/350e8da8-1920-4e51-a593-e4b4c6505a19)

### Cons 

- Output maybe not that user friendly, but data grows , model learns more and becomes accurate to all the english words

### FINISHED 🎉🎉🎉🎉








