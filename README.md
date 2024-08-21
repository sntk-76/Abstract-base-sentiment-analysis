# Slurs Classification Using BERT, DistilBERT, and ALBERT

## Overview
This repository contains a project aimed at classifying slur gender using various pre-trained transformer models, including BERT, DistilBERT, and ALBERT. The project involves data preprocessing, tokenization, model building, training, and evaluation. The goal is to leverage the powerful contextual embeddings provided by these models to achieve accurate classification.

## Table of Contents
- Installation
- Data Preprocessing
- Tokenization and Encoding
- Model Building and Training
  - BERT-Based Model
  - DistilBERT-Based Model
  - ALBERT-Based Model
- Evaluation
- Results
- Conclusion
- Acknowledgements

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- imbalanced-learn
- tensorflow
- transformers
- scikit-learn

You can install the required libraries using the following command:
```bash
pip install pandas numpy matplotlib imbalanced-learn tensorflow transformers scikit-learn
```
## Data Preprocessing
The dataset used in this project contains slurs and their associated attributes. The preprocessing steps include:

Loading the dataset from a CSV file.
Dropping unnecessary columns.
Reordering the columns to focus on relevant features.
Mapping categorical values to numerical values.
Removing rows with missing values.
Resetting the index of the DataFrame.

## Tokenization and Encoding
The text data is tokenized and encoded using the BERT tokenizer. The steps include:

Loading the pre-trained BERT tokenizer.
Defining a function to tokenize and encode sentences.
Splitting the data into training and testing sets.
Tokenizing and encoding the training and testing data.
Converting the encoded data to TensorFlow tensors.

## Model Building and Training
BERT-Based Model
Custom BERT Layer: A custom layer is created to load the pre-trained BERT model and extract the pooler output.
Model Definition: The model is designed to accept input IDs and attention masks, process them through the custom BERT layer, and classify the input using a dense layer with a softmax activation function.
Early Stopping: Early stopping is implemented to monitor the validation loss and prevent overfitting.
Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
Model Training: The model is trained on the training data with early stopping applied.

## DistilBERT-Based Model
Custom DistilBERT Layer: A custom layer is created to load the pre-trained DistilBERT model and extract the [CLS] token’s output.
Model Definition: The model is designed to accept input IDs and attention masks, process them through the custom DistilBERT layer, and classify the input using a dense layer with a softmax activation function.
Early Stopping: Early stopping is implemented to monitor the validation loss and prevent overfitting.
Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
Model Training: The model is trained on the training data with early stopping applied.

## ALBERT-Based Model
Custom ALBERT Layer: A custom layer is created to load the pre-trained ALBERT model and extract the pooler output.
Model Definition: The model is designed to accept input IDs and attention masks, process them through the custom ALBERT layer, and classify the input using a dense layer with a softmax activation function.
Early Stopping: Early stopping is implemented to monitor the validation loss and prevent overfitting.
Model Compilation: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.
Model Training: The model is trained on the training data with early stopping applied.

## Evaluation
The models are evaluated on the testing data using accuracy as the primary metric. Early stopping ensures that the best model weights are restored based on the validation loss.

## Results
The results of the models are compared based on their accuracy and processing time. The BERT-based model provides strong performance with detailed contextual embeddings, while the DistilBERT and ALBERT models offer efficient alternatives with faster training times.

## Conclusion
This project demonstrates the effectiveness of using pre-trained transformer models for text classification tasks. By leveraging BERT, DistilBERT, and ALBERT, we can achieve accurate and efficient classification of slur gender. The choice of model depends on the trade-off between performance and computational efficiency.

## Acknowledgements
Hugging Face for providing the pre-trained transformer models.
TensorFlow and Keras for the deep learning framework.
The authors of the slurs dataset for making the data available.
Feel free to explore the code and experiment with different models and parameters. Contributions and feedback are welcome!
