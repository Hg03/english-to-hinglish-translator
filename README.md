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

- To have a detailed look , view the [notebook]() in my repo.





