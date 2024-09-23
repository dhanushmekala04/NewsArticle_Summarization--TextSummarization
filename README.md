# *Extractive Text Summarization of News Articles Using the T5-Base Model
Project Overview
This project focuses on developing an extractive text summarization model for news articles using the T5-Base model (Text-to-Text Transfer Transformer). The goal is to condense long-form news articles into concise and meaningful summaries while retaining the key information and context of the original text.

The T5 model is a transformer-based text generation model that reformulates all language tasks into a text-to-text format, making it an excellent candidate for summarization tasks. The model will be trained and fine-tuned to generate coherent summaries from the news dataset, ensuring the summaries are not only short but also accurate and readable.

Dataset
The dataset used in this project is the BBC News Summary dataset, which contains 417 articles from the BBC published between 2004 and 2005. Each article is accompanied by five corresponding summaries that provide a brief overview of the article’s content. The first clause of each article acts as a title, giving further context for summarization.

Dataset Link: gopalkalpande/bbc-news-summary

Key Features of the Dataset:
Articles: Full news articles from the BBC.
Summaries: Five manually written summaries per article.
File Path: File path indicating the class label (e.g., business, sports).
Objectives
The primary objectives of this project are:

Build an extractive summarization model capable of summarizing long news articles.
Use the T5-Base model to generate human-like summaries.
Evaluate the model using ROUGE metrics to assess the quality of the summaries.
Implement the model for real-world inference and summarization tasks.
Methodology
1. Data Preparation
Exploratory Data Analysis (EDA): Analyzed the dataset to understand the distribution of article lengths and summary lengths.
Calculated average, maximum, and minimum lengths of articles and summaries.
Visualized the distribution of article lengths and file paths to get an idea of content variety.
Text Preprocessing: Preprocessed the text data to prepare it for model training:
Added the "summarize:" prefix to each article to guide the model on the task.
Tokenized articles and summaries using the T5 tokenizer.
Dataset Split: The dataset was split into 80% training and 20% validation sets for model development and evaluation.
2. Model Training
Model Choice: The T5-Base model was chosen for its flexibility in text-to-text transformations. It is a state-of-the-art model for summarization, translation, and text generation tasks.

Training Configuration:

Trained using the Hugging Face Trainer API.
Hyperparameters like batch size, learning rate, number of epochs, and evaluation steps were fine-tuned for optimal performance.
Metrics such as ROUGE (ROUGE-1, ROUGE-2, and ROUGE-L) were computed to evaluate the model.
Training Process:

The model was trained using the preprocessed articles as input and their corresponding summaries as the target output.
During training, the model generated summaries for validation data, which were evaluated to monitor progress and prevent overfitting.
3. Inference
After training, the model was used to summarize new articles. Inference involved:

Tokenizing the input article.
Using the trained T5 model to generate a summary.
Decoding the output and returning the summary.
Evaluation
The model was evaluated using ROUGE metrics, a standard evaluation measure for summarization tasks. ROUGE compares the overlap between the generated summaries and the reference summaries in terms of unigrams, bigrams, and longest common subsequence (LCS).

The following metrics were computed:

ROUGE-1: Measures the overlap of unigrams between the generated and reference summaries.
ROUGE-2: Measures the overlap of bigrams.
ROUGE-L: Measures the longest common subsequence.
The model’s performance was periodically evaluated during training using the validation dataset, and the model was saved after each epoch to preserve the best version.
