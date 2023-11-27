## BioEmoDetector

# Overview
The BioEmoDetector is a platform designed for predicting emotions in clinical narrative texts. With a specific structure and customizable options, users can run predictions using biomedical and clinical pre-trained language models. This README provides essential information on how to use the BioEmoDetector and customize the instructions file.

<p align="center">
<img src="https://github.com/201190024/BioEmoDetector/assets/54450055/e7a98e28-f7f9-4613-ab6c-09428b0a65d2" width="700">
</p>

## Table of Contents
1. [Pre-processing](#pre-processing)
2. [Biomedical Pre-trained Language Model Training](#biomedical-pre-trained-language-model-training)
3. [Getting Started](#Getting Started)
5. [Prerequisites](#Prerequisites)
6. [Usage](#usage)
7. [Contributing](#contributing)

## Pre-processing
The BioEmoDetector framework commences with a critical text pre-processing phase, serving as a foundational step to ensure uniformity and consistency in the input data. This process encompasses several key stages:
- **Lowercasing**: Text is systematically converted to lowercase, ensuring uniformity and case-insensitivity across the dataset.
- **Special Character, Number, and Whitespace Removal**: This step involves the removal of special characters, numerical digits, and excessive whitespace. 
- **Tokenization**: Text is tokenized into individual words, enabling more granular analysis of the content.
- **Stopword Removal**: Common stopwords are eliminated to focus on the core and more informative content.
- **Lemmatization**: Text is lemmatized to reduce words to their base form, improving the quality of subsequent analysis.

## Biomedical Pre-trained Language Model Training
This phase configures and fine-tunes Pre-trained Language Models (PLMs) to recognize emotions in clinical text. Key steps include:
- **Data Preparation**: Using the Careopinion dataset (25,000 patient opinions) for training. An additional 3,500 opinions are reserved for validation.
- **Text Representation**: Tokenizing patient opinions for better analysis.
- **Training and Fine-Tuning**: Adapting PLMs (e.g., CODER, Bio_ClinicalBERT) for emotion prediction.
- **Validation**: Assessing model performance using reserved opinions.
- **Model Configuration and Storage**: Saving model configurations and trained models.

## Getting Started
- To run the program, follow these simple steps:

1. Customize the instructions file (instructionss.txt) to specify the models you want to use for predictions.
2. Choose whether to enable majority voting (MajorityVoting=YES) for ensemble results.
3. Provide a file path in the instruction file (e.g., File=path) or enter plaintext sentences (e.g., PlainText=['sentence1', 'sentence2']).

# Important Notes
- If no models are selected, an error message will guide you to check the instructions.
- Priority is given to file predictions. If you provide a file path and exclude PlainText=none, predictions will focus solely on the file content.
- Conversely, if you enter plaintext sentences and exclude File=none, predictions will focus exclusively on those sentences.
- An error message prompts correction if both File=None and PlainText=None are omitted.

# Input Rules
To ensure correct usage:

- For plaintext sentences, enclose each sentence in square brackets and single quotes, like this: PlainText=['sentence1', 'sentence2'].
- Supported file formats include TXT, CSV, and JSON. In JSON and CSV files, the column containing sentences must be labeled "sentences" for accurate predictions.

## Running Predictions
Once the instructions file is customized, initiate the prediction process with the following command:


















































