# AT82.05-NLU-Project
### Contributor  
Jirasak Buranathawornsom st122162  
Saratoon Khantasima st122149  
Pasit Tiwawongrut st122442  

### Introduction 
<b>Survey on improving Few-shot Learning for Intent Detection</b>  
Intent Detection is a text classification task used in chat-bots and intelligent dialogue systems. The model will be used to detect intent from user’s statement. One of the main problem is most system do not have many dataset for training. Thus, in this project we will focus on training models for intent detection with the few-shot dataset.

### Dataset
Banking77: https://huggingface.co/datasets/banking77

## BERT-FP 

### Description
This model is created base on Fine-grained Post-training for Improving Retrieval-based Dialogue
Systems paper.  
For the pretrained model, it can download from https://github.com/hanjanghoon/BERT_FP.

## Infersent

### Description
Infersent is an universal sentence encoder from Meta AI. In this experiment, we use it as an sentences encoder for classification.  
All the data will be encoded by Infersent, then we pass the dataset into classification layer.  
All the required setup is based on https://github.com/facebookresearch/InferSent.

## DeCLUTR

### Description
DeCLUTR is an universal sentence encoder that was trained using self-supervised method which does not require labelled training data
and it has achieve about the same performance as that of the one using supervised  method.
more information can be found on https://github.com/JohnGiorgi/DeCLUTR 

## Code directory
Folder `Experiment 1` contains the following
- `final.ipynb` : IPYNB file to reproduce Experiment 1 (Calibration of few-shot "in-context" learning).
The source code comes from https://github.com/tonyzhaozh/few-shot-learning.
- `data_utils.py`: the file that helps loading the datasets in data folder.
- final_results.xlsx : The excel file containing the accuracy from the experiment.
- saved_results folder contains the results saved in pickle format.

Folder `Experiment 2&3` contains the following
- `Experiment 2&3.ipynb`: IPYNB file to reproduce Experiment 2 (Except Infersent & BERT-FP Model) and Experiment 3
- Folder `Experiment 2 Infersent&BERT-FP`: Contain the IPYNB files to reproduce Infersent & BERT-FP Model in Experiment 2 
- Folder `Experiment 3 Visualiztion`: Contain csv result from `Experiment 2&3.ipynb` (Experiment 3, particularly) and IPYNB to reproduce the plot in the presentation

## Model comparision accuracy
### Banking77 5 shot accuracy
![5shot result](./images/result5shot.PNG)
### Banking77 10 shot accuracy
![10shot result](./images/result10shot.PNG)
