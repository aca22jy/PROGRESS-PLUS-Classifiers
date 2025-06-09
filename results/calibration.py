import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score
from ast import literal_eval
import csv


fileName = 'scibert_500len_8b_20e_binary_results.csv'
df = pd.read_csv(fileName, converters={'Ground truth': pd.eval, 'Prediction': pd.eval, 'Probability': pd.eval})

pred_label = df['Prediction']
target = df['Ground truth']
probability = df['Probability']

pred_array = np.array([np.array(xi) for xi in pred_label])
targets_array = np.array([np.array(xi) for xi in target])
prob_array = np.array([np.array(xi) for xi in probability])


list_of_label = ['Place', 'Race', 'Occupation', 'Gender', 'Religion', 'Education', 'Socioeconomic', 'Social', 'Plus']

# binary_f1_score_micro = f1_score(targets_array, pred_array, average='micro')
# binary_f1_score_macro = f1_score(targets_array, pred_array, average='macro')
# roc = roc_auc_score(targets_array, prob_array)
# print('micro: ', binary_f1_score_micro, 'macro: ', binary_f1_score_macro, 'roc: ', roc)


def one_label_f1(label_index, threshold):
    true_label = targets_array[:, label_index]
    prob = prob_array[:, label_index]
    pred = [1 if x > threshold else 0 for x in prob]
    brier = brier_score_loss(true_label, pred)
    auc = roc_auc_score(true_label,pred)
    return auc, brier

all_brier = []
threshold_list = [10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2,
                  1-10e-2, 1-10e-3, 1-10e-4, 1-10e-5, 1-10e-6, 1-10e-7, 1-10e-8, 1-10e-9, 1-10e-10]
print('---------------------')

all_results = []
for i, label in enumerate(list_of_label):
    label_name = label
    print(label_name)
    all_auc = []
    all_brier = []
    for t in threshold_list:
        auc, brier = one_label_f1(i, t)
        all_auc.append(auc)
        all_brier.append(brier)
    print('auc')
    print(all_auc)
    print('brier')
    print(all_brier)
    all_results.append(all_auc)
    all_results.append(all_brier)

df = pd.DataFrame.from_records(all_results)
df = pd.DataFrame.transpose(df)
print(df)
df.to_csv('all_results.csv')
