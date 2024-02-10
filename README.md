# BioEmoDetector

## Overview
The BioEmoDetector is a platform designed for predicting emotions in clinical narrative texts. With a specific structure and customizable options, users can run predictions using biomedical and clinical pre-trained language models. This README provides essential information on how to use the BioEmoDetector and customize the Configuration file.

<p align="center">
<img src="https://github.com/201190024/BioEmoDetector/assets/54450055/e7a98e28-f7f9-4613-ab6c-09428b0a65d2" width="700">
</p>

## Table of Contents
1. [Pre-processing](#pre-processing)
2. [Biomedical Pre-trained Language Model Training](#biomedical-pre-trained-language-model-training)
3. [Getting Started](#Getting-Started)
4. [Important Notes](#Important-Notes)
5. [Input Rules](#Input-Rules)
6. [Running Predictions](#Running-Predictions)
7. [Prerequisites](#Prerequisites)
8. [Usage](#Usage)

## Pre-processing
The BioEmoDetector framework begins with a critical text pre-processing phase, serving as a foundational step to ensure uniformity and consistency in the input data. This process includes several key stages:

- **Lowercasing**: Text is systematically converted to lowercase, ensuring uniformity and case-insensitivity across the dataset.
- **Special Character, Number, and Whitespace Removal**: This step involves the removal of special characters, numerical digits, and excessive whitespace. 
- **Tokenization**: Text is tokenized into individual words, enabling more granular analysis of the content.
- **Stopword Removal**: Common stopwords are eliminated to focus on the core and more informative content.
- **Lemmatization**: Text is lemmatized to reduce words to their base form, improving the quality of subsequent analysis.

## Biomedical Pre-trained Language Model Training
This phase configures and fine-tunes Pre-trained Language Models (PLMs) to recognize emotions in clinical text. Key steps include:
- **Data Preparation**: Using the Careopinion dataset (28,500 patient opinions) for training and validation.
- **Text Representation**: Tokenizing patient opinions for better analysis.
- **Training and Fine-Tuning**: Adapting PLMs (e.g., CODER, Bio_ClinicalBERT) for emotion prediction.
- **Validation**: Assessing model performance using reserved opinions.
- **Model Configuration and Storage**: Saving model configurations and trained models.

## Getting Started
- To run the program, follow these simple steps:

1. Customize the Configuration file (Configuration.txt) to specify the models you want to use for predictions.
2. Choose whether to enable majority voting (MajorityVoting=yes) for ensemble results.
3. Provide a file path in the Configuration file (e.g., File=path) or enter plaintext (e.g., PlainText=['text1', 'text2']).

## Important Notes

- If no models are selected, an error message will guide the user to check the configuration file.
- If the user provides a file path and excludes PlainText=none, predictions will focus only on the file content.
- Conversely, if the user enters PlainText and excludes File=none, predictions will focus exclusively on those texts.
- Priority is given to file predictions. if the user provides a file path and PlainText, predictions will focus only on the file content.
- An error message prompts correction if both File=None and PlainText=None are excluded.

## Input Rules
To ensure correct usage:

- For PlainText, include each text in square brackets and single quotes, like this: PlainText=['text1', 'text2'].
- In the case of using an external file, the program supports three types of formats for the files: TXT, CSV, and JSON. When using a JSON file, the field containing the opinions must be labeled as "texts" and every opinion has to be written in quotation marks. For CSV files, the first row must be also labeled as "texts" and every row represents one opinion.  For the TXT files, the first row represents the first opinion. 

## Running Predictions
Once the Configuration file is customized, initiate the prediction process with the following steps:

### Prerequisites
- Python 3.6 or higher
- Install required libraries using `requirements.txt`

### Usage
1. Clone the repository:
`!git clone https://your_username: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@github.com/201190024/BioEmoDetector.git`

2. changes the current working directory to BioEmoDetector `%cd BioEmoDetector`

3. Install required packages:
`pip install -r requirement.txt`

4. Run the main prediction script:
`python src/Prediction.py`

5. After running the command, you will find the results saved to `Results.txt`, providing predictions for each model selected and the majority voting results
