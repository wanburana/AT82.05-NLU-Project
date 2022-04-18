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



## Code directory
Folder `Experiment 2&3` contains the following
- `Experiment 2&3.ipynb`: IPYNB file to reproduce Experiment 2 (Except Infersent & BERT-FP Model) and Experiment 3
- Folder `Experiment 2 Infersent&BERT-FP`: Contain the IPYNB files to reproduce Infersent & BERT-FP Model in Experiment 2 
- Folder `Experiment 3 Visualiztion`: Contain csv result from `Experiment 2&3.ipynb` (Experiment 3, particularly) and IPYNB to reproduce the plot in the presentation
