# BioEmoDetector

BioEmoDetector is an open-source framework for emotion prediction in texts related to medical environments. This tool leverages multiple biomedical and clinical pre-trained language models for emotion classification in clinical texts, providing a flexible ensemble model for accurate predictions.
<p align="center">
<img src="https://github.com/201190024/BioEmoDetector/assets/54450055/7ae9b076-3e25-4ae9-8736-e85b09bb395c" width="700">
</p>

## Table of Contents
1. [Pre-processing](#pre-processing)
2. [Biomedical Pre-trained Language Model Training](#biomedical-pre-trained-language-model-training)
3. [Features](#Features)
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

## Features
- **Ensemble Model:** The framework employs an ensemble of biomedical and clinical pre-trained language models, including CODER, BioBERT, BioClinical BERT, SciBERT, ClinicalLongformer, BlueBERT, and BioMedRoberta.

- **Various Input Options:** Users can choose between two modes:
  1. **Default Input (user_input=False):** Predictions are made based on a default text file provided by the user. The results are saved in a `Results.txt` file, including predictions from all models and majority voting.
  2. **Custom Input (user_input=True):** Users have the flexibility to upload various data file formats (txt, CSV, JSON) or input free text. Multiple prediction options are available, including choosing specific models, pairs of models, all models, or majority voting. Results are saved in a `Results.txt` file.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Install required libraries using `requirements.txt`

`pip install -r requirements.txt`

## Usage
1. Clone the repository:
`!git clone https://201190024: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@github.com/201190024/BioEmoDetector.git`

`%cd BioEmoDetector`

3. Install required packages:
`pip install -r requirement.txt`

4. Run the main prediction script:
`python src/Prediction.py`

5. After running the command, you will find the results saved to `Results.txt`, providing predictions for each model and majority voting results.

- Run the prediction script with custom input:
`!python src/Prediction.py True`

Follow the on-screen instructions to make predictions based on your preferred input mode.
![choices](https://github.com/201190024/BioEmoDetector/assets/54450055/521dee61-0999-4b74-84b4-201045d41307)

**Training Models**
- The models included in BioEmoDetector were trained on a preprocessed dataset. If you wish to retrain or fine-tune the models, refer to the models' script in the `src/ directory`.


