import ast
import time
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from sklearn.svm import SVC

from cosco import runCOSCO, NUM_SIM_STEPS, FAULT_RATE, FAULT_TIME, FAULT_INCREASE_TIME, RECOVER_TIME, FAULTY_HOSTS, FAILURE_TYPES, ACCUMULATIVE_FAULTS
#from cosco import runCOSCO, NUM_SIM_STEPS, FAULT_INCREASE_TIME, FAULTY_HOSTS, ACCUMULATIVE_FAULTS

#FAULT_RATE = 0.3
#FAULT_TIME = 6
#RECOVER_TIME = 18



hosts_str = ''.join([str(i) for i in FAULTY_HOSTS])
type_str = 'acc' if ACCUMULATIVE_FAULTS else 'rec'
fault_type_str = ''.join([str(i[0]).lower() for i in FAILURE_TYPES])

DATAPATH = f'AI/backups/{NUM_SIM_STEPS}i_{FAULT_RATE}fr_{FAULT_TIME}ft_{RECOVER_TIME}rt_{FAULT_INCREASE_TIME}fit_hosts{hosts_str}_{type_str}_{fault_type_str}/'
FIGURES_PATH = f'{DATAPATH}/figures/'
CSV_PATH = f'logs/MyFog_MyAzure2019Workload_{NUM_SIM_STEPS}_6_30_10000_300_1/hostinfo_with_interval.csv'

NUMBER_OF_SIMULATIONS = 30
NUMBER_OF_SIMULATIONS = 18
NUMBER_OF_REPETITIONS = 50

def generate_datasets():
    """
    Generates datasets for the AI by calling the COSCO simulator
    Will generate NUMBER_OF_SIMULATIONS datasets

    """
    # create datapath folder if it doesn't exist

    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)
    os.makedirs(os.path.dirname(DATAPATH+'data/'), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH+'analysis/'), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH+'metrics/'), exist_ok=True)


    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        # skip if log file already exists
        if os.path.isfile(datapath_i):
            continue

        print(f"Creating DATA {i+1} of {NUMBER_OF_SIMULATIONS}")
        # run simulation
        runCOSCO(prints = False)

        # copy log file to datapath
        os.system(f"cp {CSV_PATH} {datapath_i}")

def evaluate_datasets():

    # EVALUATING DATA
    metrics_1 = [[], [], [], []]  # accuracy, precision, recall, f1
    metrics_all = [[], [], [], []]  # accuracy, precision, recall, f1
    metrics_12_3 = [[], [], [], []]  # accuracy, precision, recall, f1

    best_f1 = 0
    best_pred = None
    best_cpu = None

    metrics_path = FIGURES_PATH+'metrics/'


    for i in range(NUMBER_OF_SIMULATIONS):
        print(f"Evaluating DATA {i+1} of {NUMBER_OF_SIMULATIONS}")

        datapath_i = f"{DATAPATH}data/data{i}.csv"

    
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
        #print("Host1:\n", host1['cpufailures'].value_counts())
        #print("Host2:\n", host2['cpufailures'].value_counts())
        #print("Host3:\n", host3['cpufailures'].value_counts())

        # WORK WITH BINARY CLASSIFICATION
        host1['cpufailures'] = host1['cpufailures'].apply(lambda x: 1 if x > 0 else 0)
        host2['cpufailures'] = host2['cpufailures'].apply(lambda x: 1 if x > 0 else 0)
        host3['cpufailures'] = host3['cpufailures'].apply(lambda x: 1 if x > 0 else 0)

        #print("Class distribution after binary classification:")
        #print("Host1:\n", host1['cpufailures'].value_counts())
        #print("Host2:\n", host2['cpufailures'].value_counts())
        #print("Host3:\n", host3['cpufailures'].value_counts())


        # TRAIN AND EVALUATE ONLY ON HOST1
        metrics_temp, _ = train_and_evaluate(host1, 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)
        metrics_1[0].extend(metrics_temp[0])
        metrics_1[1].extend(metrics_temp[1])
        metrics_1[2].extend(metrics_temp[2])
        metrics_1[3].extend(metrics_temp[3])


        # TRAIN AND EVALUATE ON ALL HOSTS TOGETHER

        # concatenate data
        all_hosts = pd.concat([host1, host2, host3])

        metrics_temp, _ = train_and_evaluate(all_hosts, 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)
        metrics_all[0].extend(metrics_temp[0])
        metrics_all[1].extend(metrics_temp[1])
        metrics_all[2].extend(metrics_temp[2])
        metrics_all[3].extend(metrics_temp[3])
        
        # TRAIN ON HOSTS 1 AND 2, EVALUATE ON HOST 3

        # concatenate data
        host1_2 = pd.concat([host1, host2])

        metrics_temp, best_info = train_and_evaluate(host1_2, 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), data_test=host3, binary=True)
        metrics_12_3[0].extend(metrics_temp[0])
        metrics_12_3[1].extend(metrics_temp[1])
        metrics_12_3[2].extend(metrics_temp[2])
        metrics_12_3[3].extend(metrics_temp[3])

    
        if best_info[1] > best_f1:
            best_f1 = best_info[1]
            best_pred = best_info[0]
            best_cpu = host3['cpu'].values


    # plot histograms for f1 scores
    plt.figure()
    plt.hist(metrics_1[3], bins=10, alpha=0.5)
    plt.hist(metrics_all[3], bins=10, alpha=0.5)
    plt.hist(metrics_12_3[3], bins=10, alpha=0.5)
    plt.legend(['Host1', 'All hosts', 'Host1 and Host2'])
    plt.savefig(f'{metrics_path}/f1_scores.png')


    # plots for host1
    plot_metrics(metrics_1, 'metrics_1')

    # plots for all hosts
    plot_metrics(metrics_all, 'metrics_all')

    # plots for train on host1 and host2, test on host3
    plot_metrics(metrics_12_3, 'metrics_12_3')

    # plot cpu usage with the color of the confusion label
    classes = []
    for i in range(len(best_pred)):
        if best_pred[i] == best_cpu.iloc[:, -1].values[i]:
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
    data_temp = pd.read_csv(f"{DATAPATH}data/data0.csv")

    num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
    
    # create big data dataframe
    big_data = [pd.DataFrame() for _ in range(num_hosts)]

    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
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

def failure_distribution():
    analysis_path = FIGURES_PATH+'analysis/'

    os.makedirs(os.path.dirname(analysis_path+'failuresdist/'), exist_ok=True)


    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
        #print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=['interval', 'disk', 'diskavailable'])
        # get headers
        headers = data_temp.columns

        # create list of copies of data
        data = [data_temp.copy() for _ in range(num_hosts)] 

        for j in range(num_hosts):
            for header in headers:
                data[j][header] = data[j][header].apply(lambda x: json.loads(x)[j])
            

        # count number of cpu failures
        counts_cpu = [list(host['cpufailures'].value_counts()) for host in data]
        
        # count number of ram failures
        counts_ram = [list(host['ramfailures'].value_counts()) for host in data]

        num_max_labels = max([max([len(count)] for count in counts_cpu)[0], max([len(count) for count in counts_ram])])
        #print(f'Number of labels: {num_max_labels}')

        for count in counts_cpu:
            while len(count) < num_max_labels:
                count.append(0)

        for count in counts_ram:
            while len(count) < num_max_labels:
                count.append(0)


        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        
        x = np.arange(num_max_labels)
        x_labels = [str(label) for label in range(num_max_labels)]

        width = 1 / ((num_hosts * 2) + 1)
        multiplier = 0

        for h_i in range(num_hosts):
            offset = width * multiplier
            multiplier += 1

            # cpu failures
            rects = ax.bar(x + offset, counts_cpu[h_i], width, label=f'Host {h_i}')
            
            for rect in rects:
                height = rect.get_height()

                if height > 0:
                    ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

            offset = width * (multiplier + num_hosts - 1)

            # ram failures
            rects = ax.bar(x + offset, counts_ram[h_i], width, label=f'Host {h_i}', hatch='//')

            for rect in rects:
                height = rect.get_height()

                if height > 0:
                    ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')


        ax.set_xlabel('Failure Intensity')
        ax.set_ylabel('Number of Occurrences')
        
        ax.set_xticks(x + (num_hosts * 2 - 1) / 2  * width)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='upper right')
        plt.savefig(f'{analysis_path}failuresdist/data{i}.png')

def big_merged_data_eda():
    # Exploratory Data Analysis on the merged data

    big_analysis_path = FIGURES_PATH+'analysis/big_merged_data_eda/'
    os.makedirs(os.path.dirname(big_analysis_path), exist_ok=True)
    os.makedirs(os.path.dirname(big_analysis_path+'pairs/'), exist_ok=True)

    # load and merge data
    merged_big_data = pd.DataFrame()
    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = int(len(json.loads(data_temp['cpu'][0]))/2)
        #print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=['interval', 'disk', 'diskavailable'])
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

    print('\n---- INFO ----')
    print(merged_big_data.info())

    print('\n---- DESCRIPTION ----')
    print(merged_big_data.describe())

    # 2. Duplicate Values

    print(f'\n---- DUPLICATES: {merged_big_data.duplicated().sum()}')

    # 5. Missing Values
    print(f'\n---- MISSING VALUES ----\n{merged_big_data.isnull().sum()}')

    #"""

    # 10. Correlation Matrix
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 9), tight_layout=True)
    corr = merged_big_data.corr()
    sns.heatmap(corr, annot=True, fmt='.3f', ax=ax)
    plt.savefig(f'{big_analysis_path}correlation_matrix.png')

    # Correlation Matrix shows that there is no strong correlation between cpufailures and [numcontainers, baseips, ipsavailable, ipscap, host_ltype]
    # Whith this information, we will try to predict cpufailures using all the features and compare it to the results of using only the features that have a correlation with cpufailures
    #   wich are [cpu, apparentips]

    """

    # Train and Evaluate with all features
    metrics, _ = train_and_evaluate(merged_big_data, 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data, 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)

    print(f'''\t{'METRICS ALL FEATURES':<48}\t{'METRICS ALL FEATURES (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')
    
    #   METRICS ALL FEATURES                                    METRICS ALL FEATURES (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1
    #   mean    0.9535    0.9516    0.9535    0.9523                    0.9645    0.8807    0.8115    0.8446
    #   median  0.9535    0.9515    0.9535    0.9522                    0.9646    0.8801    0.8120    0.8450
    #   std     0.0010    0.0011    0.0010    0.0010                    0.0008    0.0053    0.0058    0.0035


    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)

    print(f'''\t{'METRICS WITHOUT HOST_LTYPE':<48}\t{'METRICS WITHOUT HOST_LTYPE (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')
    
    #   METRICS WITHOUT HOST_LTYPE                              METRICS WITHOUT HOST_LTYPE (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1
    #   mean    0.9644    0.9635    0.9644    0.9637                    0.9647    0.8822    0.8120    0.8456
    #   median  0.9644    0.9635    0.9644    0.9637                    0.9647    0.8838    0.8126    0.8461
    #   std     0.0009    0.0009    0.0009    0.0009                    0.0008    0.0065    0.0072    0.0042
    

    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE AND IPSCAP
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)

    print(f'''\t{'METRICS WITHOUT HOST_LTYPE AND IPSCAP':<48}\t{'METRICS WITHOUT HOST_LTYPE AND IPSCAP (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')
    
    #   METRICS WITHOUT HOST_LTYPE AND IPSCAP                   METRICS WITHOUT HOST_LTYPE AND IPSCAP (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1
    #   mean    0.9640    0.9631    0.9640    0.9634                    0.9642    0.8789    0.8106    0.8433
    #   median  0.9640    0.9632    0.9640    0.9634                    0.9641    0.8800    0.8105    0.8432
    #   std     0.0008    0.0008    0.0008    0.0008                    0.0009    0.0065    0.0061    0.0041


    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)

    print(f'''\t{'METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS':<48}\t{'METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')

    #   METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS          METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1
    #   mean    0.9631    0.9622    0.9631    0.9625                    0.9632    0.8730    0.8096    0.8401
    #   median  0.9631    0.9623    0.9631    0.9626                    0.9632    0.8735    0.8103    0.8398
    #   std     0.0011    0.0011    0.0011    0.0011                    0.0008    0.0055    0.0067    0.0039

    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS BUT WITH SVM
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips']), 'cpufailures', SVC(), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips']), 'cpufailures', SVC(), binary=True)

    print(f'''{'METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (SVM)':<56}\tMETRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (SVM) (binary)
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')
    
    #   METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (SVM)    METRICS WITHOUT HOST_LTYPE, IPSCAP AND BASEIPS (SVM) (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1        
    #   mean    0.8813    0.8031    0.8813    0.8263                    0.8837    0.8343    0.0285    0.0551    
    #   median  0.8817    0.7916    0.8817    0.8267                    0.8835    0.8354    0.0283    0.0548    
    #   std     0.0020    0.0302    0.0020    0.0029                    0.0018    0.0420    0.0023    0.0043 
    

    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips', 'ipsavailable']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips', 'ipsavailable']), 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=True)

    print(f'''\t{'METRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE':<48}\t{'METRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')
    
    #   METRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE    METRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE (binary)
    #           accuracy  precision recall    f1                        accuracy  precision recall    f1        
    #   mean    0.9456    0.9434    0.9456    0.9443                    0.9590    0.8543    0.7901    0.8209    
    #   median  0.9458    0.9435    0.9458    0.9444                    0.9590    0.8554    0.7897    0.8206    
    #   std     0.0013    0.0014    0.0013    0.0013                    0.0009    0.0050    0.0059    0.0038    
    

    # TRAIN AND EVALUATE WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE BUT WITH SVM
    metrics, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips', 'ipsavailable']), 'cpufailures', SVC(), binary=False)
    # binary classification
    metrics_bin, _ = train_and_evaluate(merged_big_data.drop(columns=['host_ltype', 'ipscap', 'baseips', 'ipsavailable']), 'cpufailures', SVC(), binary=True)

    print(f'''{'METRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE (SVM)':<56}\tMETRICS WITHOUT HOST_LTYPE, IPSCAP, BASEIPS AND IPSAVAILABLE (SVM) (binary)
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        ''')

    """

    # Train and Evaluate without host_ltype, ipscap, baseips and numcontainers
    # It was tested but the results were a lot worse

    # plot metrics
    #plot_metrics(metrics, 'big_merged_data_all_features')

    # Train and Evaluate with only the features that have a correlation with cpufailures
    #metrics, _ = train_and_evaluate(merged_big_data[['cpu','apparentips', 'cpufailures']], 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)

    # plot metrics
    #plot_metrics(metrics, 'big_merged_data_correlated_features')

    #"""

    # Remove ltype and ips cap?
    # normalize data?
    # remove outliers? is there any?

    """
    # plot every feature against cpufailures
    for feature in merged_big_data.columns:
        if feature != 'cpufailures':
            # 2 subplots:
                # 1. scatter plot
                # 2. box plot

            plt.figure()
            fig, ax = plt.subplots(1,2, figsize=(10, 5), tight_layout=True)


            # 1. scatter plot
            sns.scatterplot(x='cpufailures', y=feature, data=merged_big_data, ax=ax[0])

            # 2. box plot
            sns.boxplot(x='cpufailures', y=feature, data=merged_big_data, ax=ax[1])

            plt.savefig(f'{big_analysis_path}pairs/{feature}_vs_numfailures.png')
    
    # Pairplot
    plt.figure()
    sns.pairplot(merged_big_data, hue='cpufailures')
    plt.savefig(f'{big_analysis_path}pairs/pairplot.png')

    """


    # select k best features
    # https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning
    # the aforementioned tutorial mentions that, for numerical input and categorical output, we should use ANOVA Correlation Coefficient (linear) or Kendall's rank coefficient (non-linear)

    print('\n---- SELECT K BEST FEATURES - RAM ----')

    print('\n\t-- ANOVA --')

    best_features = SelectKBest(score_func=f_classif, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))

    print('\n\t-- CHI2 --')

    best_features = SelectKBest(score_func=chi2, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))


    # feature importance
    print('\n---- FEATURE IMPORTANCE ----')

    model = ExtraTreesClassifier()
    model.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)
    print(feat_importances.sort_values(ascending=False))

    print('\n---- SELECT K BEST FEATURES - CPU ----')

    print('\n\t-- ANOVA --')

    best_features = SelectKBest(score_func=f_classif, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))

    """
    print('\n\t-- KENDALL --')

    best_features = SelectKBest(score_func=kendalltau, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))
    """


    print('\n\t-- CHI2 --')

    best_features = SelectKBest(score_func=chi2, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))


    # feature importance
    print('\n---- FEATURE IMPORTANCE ----')

    model = ExtraTreesClassifier()
    model.fit(merged_big_data.drop(columns=['cpufailures', 'ramfailures']), merged_big_data['ramfailures'])

    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=merged_big_data.drop(columns=['cpufailures', 'ramfailures']).columns)
    print(feat_importances.sort_values(ascending=False))



def test():
    #plot ram from datasets

    # read data 1
    datapath = "logs/MyFog_MyAzure2019Workload_100_6_30_10000_300_1/hostinfo_with_interval.csv"
    data = pd.read_csv(datapath)

    num_hosts = 3

    headers = ['ram_s','ram_r','ram_w','ramavailable_s','ramavailable_r','ramavailable_w']

    data = data[['interval'] + headers]

    hosts = [data.copy() for _ in range(num_hosts)]

    for i in range(num_hosts):
        for h in headers:
            hosts[i][h] = hosts[i][h].apply(lambda x: ast.literal_eval(x)[i])

        print(hosts[i].head())
        print(hosts[i].describe())

    
    # plot with 'interval' as x axis
    
    # polot horizontal line in each plot
    list_ram = [[4295, 17180, 34360], [372., 360., 376.54], [200., 305., 266.75]]

    _, ax = plt.subplots(3, 3, figsize=(15, 10))
    for i, host in enumerate(hosts):
        for r in ['ram', 'ramavailable']:
            for j, hh in enumerate(['s', 'r', 'w']):
                ax[i][j].axhline(y=list_ram[j][i], color='r', linestyle='-')
                sns.lineplot(x='interval', y=f'{r}_{hh}', data=host, ax=ax[i][j])
                

    plt.savefig('ram.png')

            
def train_and_evaluate(data, y_col, model, data_test = None, binary=False):
    """
    Train and evaluate a model using the given data and model
    It will run NUMBER_OF_REPETITIONS times

    Parameters
    ----------
    data : pandas.DataFrame
        Data to train and evaluate the model
    y_col : str
        Name of the column to predict
    model : sklearn.model
        Model to train and evaluate
    data_test : pandas.DataFrame, optional
        Test data to evaluate the model, by default None
    binary : bool, optional
        If True, the data will be converted to binary, by default False

    Returns
    -------
    list
        List of metrics [accuracy, precision, recall, f1]
    tuple
        Tuple with the best predicted values and respective f1 score
    """

    if binary:
        data[y_col] = data[y_col].apply(lambda x: 1 if x > 0 else 0)

    metrics = [[], [], [], []]
    best_f1 = 0
    y_pred_best = None
    for _ in range(NUMBER_OF_REPETITIONS):
        # split data if test data is not provided
        if data_test is None:
            train, test = train_test_split(data, test_size=0.3, shuffle=True)
        else:
            train = data
            test = data_test

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

        if metrics[3][-1] > best_f1:
            best_f1 = metrics[3][-1]
            y_pred_best = y_pred

    return metrics, (y_pred_best, best_f1)

def plot_metrics(metrics, name):
    """
    Plot the metrics

    Parameters
    ----------
    metrics : list
        List of metrics to plot [accuracy, precision, recall, f1]
    name : str
        Name of the file to save the plot
    """
    
    plt.figure()
    plt.boxplot(metrics)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{FIGURES_PATH}metrics/{name}.png')


    

if __name__ == '__main__':
    time_start = time.time()
    
    #generate_datasets()

    ##failure_distribution()

    #train_and_evaluate_big_data()

    big_merged_data_eda()

    #test()

    print(f'Time taken: {time.time() - time_start}')


