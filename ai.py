import time
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


#from cosco import runCOSCO, NUM_SIM_STEPS, FAULT_RATE, FAULT_TIME, FAULT_INCREASE_TIME, RECOVER_TIME, FAULTY_HOSTS, ACCUMULATIVE_FAULTS
from cosco import runCOSCO, NUM_SIM_STEPS, FAULT_INCREASE_TIME, FAULTY_HOSTS, ACCUMULATIVE_FAULTS

FAULT_RATE = 0.3
FAULT_TIME = 6
RECOVER_TIME = 18



hosts_str = ''.join([str(i) for i in FAULTY_HOSTS])
type_str = 'acc' if ACCUMULATIVE_FAULTS else 'rec'

DATAPATH = f'AI/backups/{NUM_SIM_STEPS}i_{FAULT_RATE}fr_{FAULT_TIME}ft_{RECOVER_TIME}rt_{FAULT_INCREASE_TIME}fit_hosts{hosts_str}_{type_str}/'
FIGURES_PATH = f'{DATAPATH}/figures/'
CSV_PATH = f'logs/MyFog_MyAzure2019Workload_{NUM_SIM_STEPS}_6_30_10000_300_1/hostinfo_with_interval.csv'

NUMBER_OF_SIMULATIONS = 30
NUMBER_OF_REPETITIONS = 50

def main():
    # create datapath folder if it doesn't exist

    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH+'analysis/'), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH+'metrics/'), exist_ok=True)

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
        metrics_path = FIGURES_PATH+'metrics/'

    
        # read data
        data = pd.read_csv(datapath_i)

        # data has lists on each column. in this case, we want only the first element of each list
        # remove some columns that are not needed for now

        data = data.drop(columns=['interval','ram', 'ramavailable', 'disk', 'diskavailable'])
        # get headers
        headers = data.columns

        # create copies of data
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
            metrics_1[3].append(0 if metrics_1[1][-1] * metrics_1[2][-1] == 0 else 2 * (metrics_1[1][-1] * metrics_1[2][-1]) / (metrics_1[1][-1] + metrics_1[2][-1]))


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
            metrics_all[3].append(0 if metrics_all[1][-1] * metrics_all[2][-1] == 0 else 2 * (metrics_all[1][-1] * metrics_all[2][-1]) / (metrics_all[1][-1] + metrics_all[2][-1]))


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
            metrics_12_3[3].append(0 if metrics_12_3[1][-1] * metrics_12_3[2][-1] == 0 else 2 * (metrics_12_3[1][-1] * metrics_12_3[2][-1]) / (metrics_12_3[1][-1] + metrics_12_3[2][-1]))

            if metrics_12_3[3][-1] > best_f1:
                best_f1 = metrics_12_3[3][-1]
                best_pred = pred
                best_cpu = test['cpu'].values


    # plot histograms for f1 scores
    plt.figure()
    plt.hist(metrics_1[3], bins=10, alpha=0.5)
    plt.hist(metrics_all[3], bins=10, alpha=0.5)
    plt.hist(metrics_12_3[3], bins=10, alpha=0.5)
    plt.legend(['Host1', 'All hosts', 'Host1 and Host2'])
    plt.savefig(f'{metrics_path}/f1_scores.png')


    # plots for host1
    plt.figure()
    plt.boxplot(metrics_1)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{metrics_path}/metrics_1.png')

    # plots for all hosts
    plt.figure()
    plt.boxplot(metrics_all)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{metrics_path}/metrics_all.png')

    # plots for train on host1 and host2, test on host3
    plt.figure()
    plt.boxplot(metrics_12_3)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{metrics_path}/metrics_12_3.png')

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
    
    plt.savefig(f'{metrics_path}/cpu_12_3.png')


def train_and_evaluate_big_data():
    data_temp = pd.read_csv(DATAPATH + f"data0.csv")

    num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
    
    # create big data dataframe
    big_data = [pd.DataFrame() for _ in range(num_hosts)]

    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = DATAPATH + f"data{i}.csv"
        data_temp = pd.read_csv(datapath_i)
        #print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=['interval','ram', 'ramavailable', 'disk', 'diskavailable'])
        # get headers
        headers = data_temp.columns

        # create list of copies of data
        data = [data_temp.copy() for _ in range(num_hosts)] 

        for j in range(num_hosts):
            for header in headers:
                data[j][header] = data[j][header].apply(lambda x: json.loads(x)[j])

            # append data to big data
            big_data[j] = big_data[j].append(data[j])

    for i in range(num_hosts):
        big_data[i] = big_data[i].reset_index(drop=True)

    # train and evaluate
    metrics = [[], [], [], []]
    for _ in range(NUMBER_OF_REPETITIONS):
        # split data
        train, test = train_test_split(big_data[0], test_size=0.2)

        # train model
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

        # predict
        pred = clf.predict(test.iloc[:, :-1])

        # evaluate
        metrics[0].append(accuracy_score(test.iloc[:, -1], pred))
        metrics[1].append(precision_score(test.iloc[:, -1], pred, average='macro'))
        metrics[2].append(recall_score(test.iloc[:, -1], pred, average='macro'))
        metrics[3].append(0 if metrics[1][-1] * metrics[2][-1] == 0 else 2 * (metrics[1][-1] * metrics[2][-1]) / (metrics[1][-1] + metrics[2][-1]))

    # plot metrics
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{FIGURES_PATH}/metrics_big_data.png')

    # train in all data
    merged_big_data = pd.DataFrame()
    for i in range(num_hosts):
        merged_big_data = merged_big_data.append(big_data[i])

    merged_big_data = merged_big_data.reset_index(drop=True)
    print(merged_big_data.shape)

    # split data
    train, test = train_test_split(merged_big_data, test_size=0.2)

    # train model
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train.iloc[:, :-1], train.iloc[:, -1])

    # predict
    pred = clf.predict(test.iloc[:, :-1])

    # evaluate
    metrics = []
    metrics.append(accuracy_score(test.iloc[:, -1], pred))
    metrics.append(precision_score(test.iloc[:, -1], pred, average='macro'))
    metrics.append(recall_score(test.iloc[:, -1], pred, average='macro'))
    metrics.append(0 if metrics[1] * metrics[2] == 0 else 2 * (metrics[1] * metrics[2]) / (metrics[1] + metrics[2]))

    print(metrics)

def dataanalysis():
    analysis_path = FIGURES_PATH+'analysis/'

    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = DATAPATH + f"data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
        #print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=['interval','ram', 'ramavailable', 'disk', 'diskavailable'])
        # get headers
        headers = data_temp.columns

        # create list of copies of data
        data = [data_temp.copy() for _ in range(num_hosts)] 

        for j in range(num_hosts):
            for header in headers:
                data[j][header] = data[j][header].apply(lambda x: json.loads(x)[j])
            

        # count number of failures
        counts = [list(host['numfailures'].value_counts()) for host in data]
        
        num_max_labels = max([len(count)] for count in counts)[0]

        for count in counts:
            while len(count) < num_max_labels:
                count.append(0)
        #print(f'Number of labels: {num_max_labels}')

        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        
        x = np.arange(num_max_labels)
        x_labels = [str(label) for label in range(num_max_labels)]

        width = 1 / (num_hosts + 1)
        multiplier = 0

        for h_i in range(num_hosts):
            offset = width * multiplier
            multiplier += 1
            rects = ax.bar(x + offset, counts[h_i], width, label=f'Host {h_i}')
            #ax.bar_label(rects, padding=3)
            for rect in rects:
                height = rect.get_height()

                if height > 0:
                    ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')


        ax.set_xlabel('Failure Intensity')
        ax.set_ylabel('Number of Occurrences')
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper right')
        plt.savefig(f'{analysis_path}/failuresdist{i}.png')

def big_merged_data_eda():
    # Exploratory Data Analysis on the merged data

    # load and merge data
    merged_big_data = pd.DataFrame()
    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = DATAPATH + f"data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
        #print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=['interval','ram', 'ramavailable', 'disk', 'diskavailable'])
        # get headers
        headers = data_temp.columns

        # create list of copies of data
        data = [data_temp.copy() for _ in range(num_hosts)] 

        for j in range(num_hosts):
            for header in headers:
                data[j][header] = data[j][header].apply(lambda x: json.loads(x)[j])

        for j in range(num_hosts):
            data[j]['host_ltype'] = j
            merged_big_data = merged_big_data.append(data[j])

    merged_big_data = merged_big_data.reset_index(drop=True)
    print(merged_big_data.shape)

    # following https://www.digitalocean.com/community/tutorials/exploratory-data-analysis-python

    # 1. Basic Information

    print('INFO')
    print(merged_big_data.info())

    print('DESCRIPTION')
    print(merged_big_data.describe())

    # 2. Duplicate Values

    print(f'DUPLICATES: {merged_big_data.duplicated().sum()}')

    # 5. Missing Values
    print(f'MISSING VALUES:\n{merged_big_data.isnull().sum()}')

    # 10. Correlation Matrix
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    corr = merged_big_data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
    plt.savefig(f'{FIGURES_PATH}correlation_matrix.png')

    # Correlation Matrix shows that there is no strong correlation between numfailures and [numcontainers, baseips, ipsavailable, ipscap, host_ltype]
    # Whith this information, we will try to predict numfailures using all the features and compare it to the results of using only the features that have a correlation with numfailures
    #   wich are [cpu, apparentips]

    # Train and Evaluate with all features
    metrics = train_and_evaluate(merged_big_data, 'numfailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)

    # plot metrics
    plot_metrics(metrics, 'big_merged_data_all_features')

    # Train and Evaluate with only the features that have a correlation with numfailures
    metrics = train_and_evaluate(merged_big_data[['cpu','apparentips', 'numfailures']], 'numfailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)

    # plot metrics
    plot_metrics(metrics, 'big_merged_data_correlated_features')



            
def train_and_evaluate(data, y_col, model, binary=False):

    if binary:
        data[y_col] = data[y_col].apply(lambda x: 1 if x > 0 else 0)

    metrics = [[], [], [], []]
    for _ in range(NUMBER_OF_REPETITIONS):
        # split data
        train, test = train_test_split(data, test_size=0.3, shuffle=True)

        x_train = train.drop(columns=[y_col])
        y_train = train[y_col]

        x_test = test.drop(columns=[y_col])
        y_test = test[y_col]

        # train and predict
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # evaluate
        metrics[0].append(accuracy_score(y_test, y_pred))
        metrics[1].append(precision_score(y_test, y_pred, average='binary' if binary else 'weighted'))
        metrics[2].append(recall_score(y_test, y_pred, average='binary' if binary else 'weighted'))
        metrics[3].append(f1_score(y_test, y_pred, average='binary' if binary else 'weighted'))

    return metrics

def plot_metrics(metrics, name):
    
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{FIGURES_PATH}metrics/{name}.png')


    

if __name__ == '__main__':
    time_start = time.time()
    #main()

    #dataanalysis()

    #train_and_evaluate_big_data()

    big_merged_data_eda()

    print(f'Time taken: {time.time() - time_start}')


