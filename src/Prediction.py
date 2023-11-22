import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from huggingface_hub import hf_hub_download
import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
import pandas as pd
import json
from preprocessing import preprocess_text

BioBERT_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="BioBERT.bin")
BioMedRoBERTa_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="BioMedRoBERTa.bin")
BlueBERT_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="BlueBERT.bin")
ClinicalBERT_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="ClinicalBERT.bin")
CODER_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="CODER.bin")
SciBERT_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="SciBERT.bin")
ClinicalLongFormer_path = hf_hub_download(repo_id="Bashar-Alshouha/BioEmoDetector", filename="ClinicalLongFormer.safetensors")

# Define the paths to the configuration files and the binary model files for each model
# Define the paths to the configuration files and the binary model files for each model
model_paths = {
    1: {
        "config": 'config/BioBERTconfig.json',
        "model": BioBERT_path,
        "tokenizer": "dmis-lab/biobert-base-cased-v1.1"
    },
    2: {
        "config": 'config/BioMedRoBERTaconfig.json',
        "model": BioMedRoBERTa_path,
        "tokenizer": "allenai/biomed_roberta_base"
    },
    3: {
        "config": 'config/BlueBERTconfig.json',
        "model": BlueBERT_path,
        "tokenizer": "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12"
    },
    4: {
        "config": 'config/ClinicalBERTconfig.json',
        "model": ClinicalBERT_path,
        "tokenizer": "emilyalsentzer/Bio_ClinicalBERT"
    },
    5: {
        "config": 'config/CODERconfig.json',
        "model": CODER_path,
        "tokenizer": "GanjinZero/UMLSBert_ENG"
    },
    6: {
        "config": 'config/SciBERTconfig.json',
        "model": SciBERT_path,
        "tokenizer": "allenai/scibert_scivocab_cased"
    },
    7: {
        "config": 'config/ClinicalLongFormerconfig.json',
        "model": ClinicalLongFormer_path,
        "tokenizer": "yikuan8/Clinical-Longformer"
    }
}

# Initialize an empty dictionary to store the models
models = {}

# Load and use each model
for model_number, model_data in model_paths.items():
    # Load the configuration
    config = AutoConfig.from_pretrained(model_data['config'], local_files_only=True)

    # Load the tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(model_data["tokenizer"])

    # Load the model weights
    model = AutoModelForSequenceClassification.from_pretrained(model_data['model'], config=config, local_files_only=True)

    # Store the loaded model in the dictionary
    models[model_number] = {
        "name": model_data["tokenizer"],
        "tokenizer": tokenizer,
        "model": model
    }

# Define emotion labels
emotion_labels = ['anger', 'fear', 'sadness', 'calmness', 'disgust', 'pleasantness', 'eagerness', 'joy']

# Function to make predictions for a list of sentences
def predict_sentences(sentences, model_numbers):
    results = {}
    for model_number in model_numbers:
        model_data = models[model_number]
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        model_results = []
        for input_text in sentences:
            input_text = preprocess_text(input_text)
            # Tokenize the input text
            tokens = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

            # Make predictions
            with torch.no_grad():
                outputs = model(**tokens)

            probabilities = torch.sigmoid(outputs.logits)[0]

            # Store the predicted probabilities with emotion labels
            result = {label: probability.item() for label, probability in zip(emotion_labels, probabilities)}
            model_results.append(result)

        model_name = models[model_number]['name']
        results[model_name] = model_results

    return results

def handle_user_input(user_input=False):
    choice = None  # Define choice outside the try block

    if not user_input:
        # Read sentences from the default file
        input_file_name =  "data/test case (txt).txt"
        with open(input_file_name, 'r') as text_file:
            sentences = text_file.read().splitlines()
        model_numbers = list(models.keys())
    else:
        try:
            choice = input("Choose an option:\n1. Predict with a specific model (1: BioBERT , 2: BioMedRoBERTa, 3: BlueBERT , 4: ClinicalBERT, 5: CODER , 6: SciBERT , 7: ClinicalLongFormer)\n2. Predict with two specific models\n3. Predict with all models\n4. Predict with majority voting ensemble\nEnter your choice (1/2/3/4): ")

            if choice == '1':
                model_input = input("Enter the number of the model (1: BioBERT , 2: BioMedRoBERTa, 3: BlueBERT , 4: ClinicalBERT, 5: CODER , 6: SciBERT, 7: ClinicalLongFormer): ")
                model_numbers = [int(model_input)]
            elif choice == '2':
                model_input = input("Enter model numbers separated by a comma (e.g., 1,2): ")
                model_numbers = [int(x) for x in model_input.split(',') if x]
            elif choice == '3':
                model_numbers = list(models.keys())
            elif choice == '4':
                model_numbers = [1, 2, 3, 4, 5, 6, 7]  # All models for majority voting ensemble
            else:
                print("Invalid choice. Please enter a valid option (1/2/3/4).")
                return

            print("Enter 's' to enter sentences, 'f' to upload a CSV file, 'j' for JSON file, or 't' for a text file:")
            input_type = input()

            sentences = []

            if input_type == 's':
                while True:
                    sentence = input("Enter a sentence (or type 'done' to finish): ")
                    if sentence.lower() == 'done':
                        break
                    sentences.append(sentence)

            elif input_type == 'f':
                input_file_name = input("Enter the name of the file for input (e.g., input.csv): ")

                if input_file_name.lower().endswith('.csv'):
                    try:
                        df = pd.read_csv(input_file_name)
                        sentences = df['sentences'].tolist()
                    except Exception as e:
                        print(f"Error reading CSV file: {str(e)}")
                        return

                elif input_file_name.lower().endswith('.json'):
                    try:
                        with open(input_file_name, 'r') as json_file:
                            input_data = json.load(json_file)
                            sentences = input_data['sentences']
                    except Exception as e:
                        print(f"Error reading JSON file: {str(e)}")
                        return

                elif input_file_name.lower().endswith('.txt'):
                    try:
                        with open(input_file_name, 'r') as text_file:
                            sentences = text_file.read().splitlines()
                    except Exception as e:
                        print(f"Error reading text file: {str(e)}")
                        return

                else:
                    print("Invalid file format. Please enter a valid CSV, JSON, or text file.")
                    return

            elif input_type == 'j':
                input_file_name = input("Enter the name of the file for input (e.g., input.json): ")

                if input_file_name.lower().endswith('.json'):
                    try:
                        with open(input_file_name, 'r') as json_file:
                            input_data = json.load(json_file)
                            sentences = input_data['sentences']
                    except Exception as e:
                        print(f"Error reading JSON file: {str(e)}")
                        return

                else:
                    print("Invalid file format. Please enter a valid JSON file.")
                    return

            elif input_type == 't':
                input_file_name = input("Enter the name of the file for input (e.g., input.txt): ")

                if input_file_name.lower().endswith('.txt'):
                    try:
                        with open(input_file_name, 'r') as text_file:
                            sentences = text_file.read().splitlines()
                    except Exception as e:
                        print(f"Error reading text file: {str(e)}")
                        return

                else:
                    print("Invalid file format. Please enter a valid text file.")
                    return

            else:
                print("Invalid choice. Please enter 's', 'f', 'j', or 't'.")
                return

        except Exception as e:
            print(f"Error: {str(e)}")

    results = predict_sentences(sentences, model_numbers)

    # Output results to a text file
    output_file_name = "Results.txt"
    try:
        with open(output_file_name, 'w') as text_file:
            if not user_input:
                text_file.write("Results for Each Model:\n")
                for model_name, model_results in results.items():
                    text_file.write(f"\nModel: {model_name}\n")
                    for i, result in enumerate(model_results, start=1):
                        text_file.write(f"Prediction for TEXT {i}:\n")
                        for label, probability in result.items():
                            text_file.write(f"{label}: {probability:.4f}\n")

                text_file.write("\nResults for Majority Voting Ensemble:\n")
                majority_results = {i: {label: [] for label in emotion_labels} for i in range(len(sentences))}
                for model_results in results.values():
                    for i, result in enumerate(model_results):
                        for label, probability in result.items():
                            majority_results[i][label].append(probability)

                for i, emotion_dict in majority_results.items():
                    text_file.write(f"Majority Voting for TEXT {i + 1}:\n")
                    for label, probabilities in emotion_dict.items():
                        majority_prob = sum(probabilities) / len(probabilities)
                        text_file.write(f"{label}: {majority_prob:.4f}\n")

            else:
                if choice == '4':
                    text_file.write("Results for Majority Voting Ensemble:\n")
                    majority_results = {i: {label: [] for label in emotion_labels} for i in range(len(sentences))}
                    for model_results in results.values():
                        for i, result in enumerate(model_results):
                            for label, probability in result.items():
                                majority_results[i][label].append(probability)

                    for i, emotion_dict in majority_results.items():
                        text_file.write(f"Majority Voting for TEXT {i + 1}:\n")
                        for label, probabilities in emotion_dict.items():
                            majority_prob = sum(probabilities) / len(probabilities)
                            text_file.write(f"{label}: {majority_prob:.4f}\n")

                else:
                    text_file.write("Results based on User's Choice:\n")
                    for model_name, model_results in results.items():
                        text_file.write(f"\nModel: {model_name}\n")
                        for i, result in enumerate(model_results, start=1):
                            text_file.write(f"Prediction for TEXT {i}:\n")
                            for label, probability in result.items():
                                text_file.write(f"{label}: {probability:.4f}\n")

        print(f"Results saved to {output_file_name}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Call the function with user_input=False to use the default input
user_input=False
if len(sys.argv) > 1:
    user_input=sys.argv[1]
handle_user_input(user_input=user_input)
