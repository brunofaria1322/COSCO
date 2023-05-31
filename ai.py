import pandas as pd
import numpy as np
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
FIGURES_PATH = f'{DATAPATH}/figures/'
CSV_PATH = f'logs/MyFog_MyAzure2019Workload_{NUM_SIM_STEPS}_6_30_10000_300_1/hostinfo_with_interval.csv'

NUMBER_OF_SIMULATIONS = 10
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
    plt.savefig(f'{metrics_path}/metrics_1.png')

    # plots for all hosts
    plt.figure()
    plt.boxplot(metrics_all)
    plt.xticks([1, 2, 3, 4], ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.savefig(f'{metrics_path}/metrics_all.png')

    # plots for train on host1 and host2, test on host3
    # boxplot metrics and save
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
        #print(f'Number of labels: {num_max_labels}')

        plt.figure()
        fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
        
        x = np.arange(num_max_labels)
        x_labels = [str(i) for i in range(num_max_labels)]

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



if __name__ == '__main__':
    #main()

    dataanalysis()



"""

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()

"""