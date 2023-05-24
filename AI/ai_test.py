import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

DATAPATH = 'AI/hostinfo_with_interval.csv'

def main():
    # read data
    data = pd.read_csv(DATAPATH)

    # data has lists on each column. in this case, we want only the first element of each list
    # remove collumns after 'apparentips' exclusive
    data = data.iloc[:, :data.columns.get_loc('apparentips') + 1]
    # get headers
    headers = data.columns
    for header in headers:
        if header != 'interval':
            data[header] = data[header].apply(lambda x: json.loads(x)[0])

    host1 = data

    # add true label column
    true_label = [0 if (i+1) % 20 < 10 else 1 for i in range(len(host1))]
    host1['failure'] = true_label

    print(host1.head())

    # count failures
    print("Class distribution:")
    print(host1['failure'].value_counts())

    metrics = [[], [], [], []]  # accuracy, precision, recall, f1
    for _ in range(100):
        # split data into train and test
        train, test = train_test_split(host1, test_size=0.4)

        # train model
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

        # predict
        pred = clf.predict(test.iloc[:, :-1])

        # evaluate
        metrics[0].append(accuracy_score(test.iloc[:, -1], pred))
        metrics[1].append(precision_score(test.iloc[:, -1], pred))
        metrics[2].append(recall_score(test.iloc[:, -1], pred))
        metrics[3].append(2 * (metrics[1][-1] * metrics[2][-1]) / (metrics[1][-1] + metrics[2][-1]))

    # boxplot metrics and save
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig('AI/metrics.png')







if __name__ == '__main__':
    main()     