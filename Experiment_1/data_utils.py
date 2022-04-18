import pandas as pd
import json
import pickle
import numpy as np
# from utils import ROOT_DIR

def load_examples(file_path, do_lower_case=False):
    examples = []
    
    with open('{}/seq.in'.format(file_path),'r',encoding="utf-8") as f_text, open('{}/newlabel'.format(file_path),'r',encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            
            e = Inputexample(text.strip(),label=label.strip())
            examples.append(e)
            
    return examples
    
class Inputexample(object):
    def __init__(self,text_a,label = None):
        self.text = text_a
        self.label = label

def load_banking():
    
    path_5s = f'./BANKING77/train_5/'
    path_test = f'./BANKING77/test/'
    path_valid = f'./BANKING77/valid/'


    all_train_sentences = []
    all_train_labels = []
    all_test_sentences = []
    all_test_labels = []
    train_samples = load_examples(path_5s)
    test_samples = load_examples(path_test)

    for i in range(len(train_samples)):
        all_train_sentences.append(train_samples[i].text)
        all_train_labels.append(train_samples[i].label)
    for i in range(len(test_samples)):
        all_test_sentences.append(test_samples[i].text)
        all_test_labels.append(test_samples[i].label)

    # unique_label = np.unique(np.array(all_train_labels))

    # # map text label to index classes
    # label_maps = {unique_label[i]: i for i in range(len(unique_label))}

    # all_train_labels = [label_maps[stringtoId] for stringtoId in all_train_labels]
    # all_test_labels = [label_maps[stringtoId] for stringtoId in all_test_labels]
    

    return all_train_sentences, all_train_labels, all_test_sentences, all_test_labels

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    # with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
    #     train_lines = f.readlines()
    # with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
    #     test_lines = f.readlines()
    with open(f"./data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"./data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    # train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    # test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')
    train_data = pd.read_csv(f'./data/agnews/train.csv')
    test_data = pd.read_csv(f'./data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels


def load_dbpedia():
    # train_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/train_subset.csv')
    # test_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/test.csv')
    train_data = pd.read_csv(f'./data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'./data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels


def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'banking':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_banking()
        num_class = len(np.unique(np.array(orig_train_labels)))
        # print("inside_data_util",type(orig_train_labels[0]))
        # get text label of uniqure classes
        unique_label = np.unique(np.array(orig_train_labels))

        # map text label to index classes
        inv_label_maps = {unique_label[i]: i for i in range(len(unique_label))}
        label_maps = {i:[(unique_label[i].replace("_",""))] for i in range(len(unique_label))}
        params['prompt_prefix'] = "Classify the Complaint into the categories of Balance, Charge, ERROR, Fail, Hack, Problem, Transfer, Unknown, Virtual, accept, account, activate, age, app, atm, automatic, balance, benefit, block, broken, cancel, card, charge, check, cost, country, credit, currency, delete, double, drop, edit, ended, error, extra, fail, false, fee, fund, get, hack, limit, limits, linger, link, lock, look, loss, lost, method, money, order, password, pay, pend, pause, person, physical, pin, problem, proof, rate, refund, request, return, rule, second, send, sent, special, stop, time, transfer, unknown, virtual, wait, wrong.\n\n"
        # params['prompt_prefix'] = ""
        params["q_prefix"] = "Complaint: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = label_maps
        params['inv_label_dict'] = inv_label_maps
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        orig_train_labels = [inv_label_maps[stringtoId] for stringtoId in orig_train_labels]
        orig_test_labels = [inv_label_maps[stringtoId] for stringtoId in orig_test_labels]
        # print(params['label_dict'])

    elif params['dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        # print("inside_data_util",type(orig_train_labels[0]))
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        # params['prompt_prefix'] = ""
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
        print(params['label_dict'])

    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels