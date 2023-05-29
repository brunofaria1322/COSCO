import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
    print("Host1:\n", host1['failure'].value_counts())
    print("Host2:\n", host2['failure'].value_counts())
    print("Host3:\n", host3['failure'].value_counts())


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

    best_f1 = 0
    best_pred = None

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

        if metrics[3][-1] > best_f1:
            best_f1 = metrics[3][-1]
            best_pred = pred

    # boxplot metrics and save
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig('AI/metrics_12_3.png')

    # plot cpu usage with the color of the confusion label
    classes = []
    for i in range(len(best_pred)):
        if best_pred[i] == test.iloc[:, -1].values[i]:
            if best_pred[i] == 1:
                # True Positive
                classes.append(1)
            else:
                # True Negative
                classes.append(0)
        else:
            if best_pred[i] == 1:
                # False Positive
                classes.append(3)
            else:
                # False Negative
                classes.append(2)

    colors = ListedColormap(['blue', 'green', 'yellow', 'orange'])

    plt.figure(figsize=(15, 5))
    scatter = plt.scatter(range(len(best_pred)), host3['cpu'], c=classes, cmap=colors)
    plt.xlabel('Time')
    plt.ylabel('CPU usage')
    plt.legend(handles=scatter.legend_elements()[0], loc="upper left", labels=['True Negative', 'True Positive', 'False Negative', 'False Positive'])
    
    plt.savefig('AI/cpu_12_3.png')



if __name__ == '__main__':
    main()     