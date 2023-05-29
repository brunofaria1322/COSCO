import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


from cosco import runCOSCO, NUM_SIM_STEPS, FAULTY, FAULT_RATE, FAULT_TIME, FAULT_INCREASE_TIME, RECOVER_TIME, FAULTY_HOSTS, ACCUMULATIVE_FAULTS


hosts_str = ''.join([str(i) for i in FAULTY_HOSTS])
type_str = 'acc' if ACCUMULATIVE_FAULTS else 'rec'

DATAPATH = f'AI/backups/{NUM_SIM_STEPS}i_{FAULT_RATE}fr_{FAULT_TIME}ft_{RECOVER_TIME}rt_{FAULT_INCREASE_TIME}fit_hosts{hosts_str}_{type_str}/'
CSV_PATH = 'logs/MyFog_MyAzure2019Workload_100_6_30_10000_300_1/hostinfo_with_interval.csv'

NUMBER_OF_SIMULATIONS = 10
NUMBER_OF_REPETITIONS = 10

def main():
    # create datapath folder if it doesn't exist

    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)

    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = DATAPATH + f"data{i}.csv"
        # pass if log file already exists
        if os.path.isfile(datapath_i):
            continue

        print(f"Creating DATA {i+1} of {NUMBER_OF_SIMULATIONS}")
        # run simulation
        runCOSCO(prints = False)

        # copy log file to datapath
        os.system(f"cp {CSV_PATH} {DATAPATH}/data{i}.csv")

    # EVALUATING DATA
    metrics_1 = [[], [], [], []]  # accuracy, precision, recall, f1
    metrics_all = [[], [], [], []]  # accuracy, precision, recall, f1
    metrics_12_3 = [[], [], [], []]  # accuracy, precision, recall, f1

    best_f1 = 0
    best_pred = None
    best_cpu = None

    for i in range(NUMBER_OF_SIMULATIONS):
        print(f"Evaluating DATA {i+1} of {NUMBER_OF_SIMULATIONS}")

        datapath_i = DATAPATH + f"data{i}.csv"

    
        # read data
        data = pd.read_csv(datapath_i)

        # data has lists on each column. in this case, we want only the first element of each list
        # remove some columns that are not needed for now

        data = data.drop(columns=['interval','ram', 'ramavailable', 'disk', 'diskavailable'])
        # get headers
        headers = data.columns

        # create copies of data
        host1 = data

        host1 = data.copy()
        host2 = data.copy()
        host3 = data.copy()

        for header in headers:
            host1[header] = host1[header].apply(lambda x: json.loads(x)[0])
            host2[header] = host2[header].apply(lambda x: json.loads(x)[1])
            host3[header] = host3[header].apply(lambda x: json.loads(x)[2])

        # count failures
        #print("Class distribution:")
        #print("Host1:\n", host1['numfailures'].value_counts())
        #print("Host2:\n", host2['numfailures'].value_counts())
        #print("Host3:\n", host3['numfailures'].value_counts())

        # WORK WITH BINARY CLASSIFICATION
        host1['numfailures'] = host1['numfailures'].apply(lambda x: 1 if x > 0 else 0)
        host2['numfailures'] = host2['numfailures'].apply(lambda x: 1 if x > 0 else 0)
        host3['numfailures'] = host3['numfailures'].apply(lambda x: 1 if x > 0 else 0)

        #print("Class distribution after binary classification:")
        #print("Host1:\n", host1['numfailures'].value_counts())
        #print("Host2:\n", host2['numfailures'].value_counts())
        #print("Host3:\n", host3['numfailures'].value_counts())


        # TRAIN AND EVALUATE ONLY ON HOST1
        for _ in range(NUMBER_OF_REPETITIONS):
            # split data into train and test
            train, test = train_test_split(host1, test_size=0.4)

            # train model
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

            # predict
            pred = clf.predict(test.iloc[:, :-1])

            # evaluate
            metrics_1[0].append(accuracy_score(test.iloc[:, -1], pred))
            metrics_1[1].append(precision_score(test.iloc[:, -1], pred))
            metrics_1[2].append(recall_score(test.iloc[:, -1], pred))
            metrics_1[3].append(2 * (metrics_1[1][-1] * metrics_1[2][-1]) / (metrics_1[1][-1] + metrics_1[2][-1]))


        # TRAIN AND EVALUATE ON ALL HOSTS TOGETHER

        # concatenate data
        all_hosts = pd.concat([host1, host2, host3])

        for _ in range(NUMBER_OF_REPETITIONS):
            # split data into train and test
            train, test = train_test_split(all_hosts, test_size=0.4)

            # train model
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

            # predict
            pred = clf.predict(test.iloc[:, :-1])

            # evaluate
            metrics_all[0].append(accuracy_score(test.iloc[:, -1], pred))
            metrics_all[1].append(precision_score(test.iloc[:, -1], pred))
            metrics_all[2].append(recall_score(test.iloc[:, -1], pred))
            metrics_all[3].append(2 * (metrics_all[1][-1] * metrics_all[2][-1]) / (metrics_all[1][-1] + metrics_all[2][-1]))


        # TRAIN ON HOSTS 1 AND 2, EVALUATE ON HOST 3

        # concatenate data
        host1_2 = pd.concat([host1, host2])

        for _ in range(NUMBER_OF_REPETITIONS):
            train = host1_2
            test = host3

            # train model
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

            # predict
            pred = clf.predict(test.iloc[:, :-1])

            # evaluate
            metrics_12_3[0].append(accuracy_score(test.iloc[:, -1], pred))
            metrics_12_3[1].append(precision_score(test.iloc[:, -1], pred))
            metrics_12_3[2].append(recall_score(test.iloc[:, -1], pred))
            metrics_12_3[3].append(2 * (metrics_12_3[1][-1] * metrics_12_3[2][-1]) / (metrics_12_3[1][-1] + metrics_12_3[2][-1]))

            if metrics_12_3[3][-1] > best_f1:
                best_f1 = metrics_12_3[3][-1]
                best_pred = pred
                best_cpu = test['cpu'].values


    # plots for host1
    plt.figure()
    plt.boxplot(metrics_1)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{DATAPATH}/metrics_1.png')

    # plots for all hosts
    plt.figure()
    plt.boxplot(metrics_all)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{DATAPATH}/metrics_all.png')

    # plots for train on host1 and host2, test on host3
    # boxplot metrics and save
    plt.figure()
    plt.boxplot(metrics_12_3)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{DATAPATH}/metrics_12_3.png')

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
    scatter = plt.scatter(range(len(best_pred)), best_cpu, c=classes, cmap=colors)
    plt.xlabel('Time')
    plt.ylabel('CPU usage')
    plt.legend(handles=scatter.legend_elements()[0], loc="upper left", labels=['True Negative', 'True Positive', 'False Negative', 'False Positive'])
    
    plt.savefig(f'{DATAPATH}/cpu_12_3.png')



if __name__ == '__main__':
    main()     