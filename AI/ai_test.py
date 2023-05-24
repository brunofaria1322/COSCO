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

    # create copies of data
    host1 = data

    host1 = data.copy()
    host2 = data.copy()
    host3 = data.copy()

    for header in headers:
        if header != 'interval':
            host1[header] = host1[header].apply(lambda x: json.loads(x)[0])
            host2[header] = host2[header].apply(lambda x: json.loads(x)[1])
            host3[header] = host3[header].apply(lambda x: json.loads(x)[2])

    # add true label column
    true_label = [0 if (i+1) % 20 < 10 else 1 for i in range(len(host1))]
    host1['failure'] = true_label
    host2['failure'] = true_label
    host3['failure'] = true_label

    # count failures
    print("Class distribution:")
    print(host1['failure'].value_counts())

    # train and evaluate only on host1
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
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig('AI/metrics_1.png')


    # train and evaluate on all hosts together

    # concatenate data
    all_hosts = pd.concat([host1, host2, host3])

    metrics = [[], [], [], []]  # accuracy, precision, recall, f1
    for _ in range(100):
        # split data into train and test
        train, test = train_test_split(all_hosts, test_size=0.4)

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
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig('AI/metrics_all.png')


    # train with host1 and host2, evaluate on host3

    # concatenate data
    host1_2 = pd.concat([host1, host2])

    metrics = [[], [], [], []]  # accuracy, precision, recall, f1
    for _ in range(100):
        train = host1_2
        test = host3

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
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig('AI/metrics_12_3.png')



if __name__ == '__main__':
    main()     