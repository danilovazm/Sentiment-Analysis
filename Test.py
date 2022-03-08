import torch
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix

def Test(test_set, rnn, hidden):
    predictions = []
    labels = []
    j = 0
    with torch.no_grad():

        for i, (x, y) in enumerate(test_set):
            if torch.cuda.is_available():
                x = x.cuda()
            predicted, hidden = rnn(x, hidden)
            if torch.round(predicted[-1].type(torch.FloatTensor)) == y:
                j += 1
            predictions.append(torch.round(predicted[-1].type(torch.FloatTensor)))
            labels.append(y)
    cf_matrix = confusion_matrix(labels, predictions)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix\n')
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Labels')
    print("Acurária total de: " + str(100*j/len(test_set)) + "%")
    return ax