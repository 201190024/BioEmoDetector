# BioEmoDetector
The BioEmoDetector framework is a meticulously designed system for detecting emotions in clinical text data. It employs a systematic approach including three distinct stages. The first of these stages is data pre-processing. This process entails cleaning and organizing the data to eliminate inconsistencies and enhance its quality. After data pre-processing, the framework proceeds to the biomedical and clinical PLM training stage, where biomedical and clinical PLMs (CODER, BlueBERT, SciBERT, BioMed-RoBERTa, Bio_ClinicalBERT, Clinical_Longformer, BioBERT) are comprehensively trained. These models form the foundational elements of the emotion prediction process. In the final stage, the framework utilizes the pre-trained models to predict emotions from clinical text data.

<p align="center">
<img src="https://github.com/201190024/BioEmoDetector/assets/54450055/7ae9b076-3e25-4ae9-8736-e85b09bb395c" width="700">
</p>

## Table of Contents
1. [Pre-processing](#Pre-processing)
2. [Biomedical Pre-trained Language Model Training](#biomedical-pre-trained-language-model-training)
3. [Emotion Prediction](#emotion-prediction)
4. [Usage](#usage)
5. [Contributing](#contributing)
   
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

## Emotion Prediction
Once PLMs are trained, users can predict emotions. 
- Input text: Users can input text via plain text, TXT, CSV, JSON files, or freeform text.
- Model Selection: Users choose a specific model or use multiple models for emotion prediction.
- Majority Voting Ensemble: This method combines the results from all models to give you a single, comprehensive prediction based on the majority vote.
  
The results are a "results.TXT" file of the emotion prediction presented as probabilities assigned to each emotion for every opinion or sentence entered by the user.
## Usage
To get started with the BioEmoDetector framework, follow these steps:
1. Clone this repository.
2. Set the project directory as the working directory `cd <project_directory_path>`.
3. Install the required dependencies by running `pip install -r requirements.txt`.
4. Run the prediction script to perform emotion predictions based on the pre-trained models.
   ![choices](https://github.com/201190024/BioEmoDetector/assets/54450055/521dee61-0999-4b74-84b4-201045d41307)
6. Choose your input method (free text, TXT, CSV, JSON) and model options (Single Model, Specific Models, All Models, Majority Voting).
7. Interpret Results - The prediction results will be generated and saved in a file named "results.txt".
Open the "results.txt" file to view the predicted emotions for the provided input.

## Contributing
If you'd like to contribute to the BioEmoDetector framework, we welcome your contributions. Feel free to open issues, suggest improvements, or submit pull requests. Please adhere to the project's guidelines when contributing.
For more information, contact:
Bashar at bashar.alshouha1993@gmail.com.

