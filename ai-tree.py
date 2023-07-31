import ast
import time
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# sns color palette
COLOR_PALETTE = "hls"
sns.set_palette(COLOR_PALETTE)


from matplotlib.colors import ListedColormap


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from cosco import (
    runCOSCO,
    NUM_SIM_STEPS,
    HOSTS,
    CONTAINERS,
    ROUTER_BW,
    INTERVAL_TIME,
    NEW_CONTAINERS,
    FAULT_RATE,
    FAULT_TIME,
    FAULT_INCREASE_TIME,
    RECOVER_TIME,
    FAULTY_HOSTS,
    FAILURE_TYPES,
    ACCUMULATIVE_FAULTS,
)

# from cosco import runCOSCO, NUM_SIM_STEPS, FAULT_INCREASE_TIME, FAULTY_HOSTS, ACCUMULATIVE_FAULTS

# FAULT_RATE = 0.3
# FAULT_TIME = 6
# RECOVER_TIME = 18


hosts_str = "".join([str(i) for i in FAULTY_HOSTS])
type_str = "acc" if ACCUMULATIVE_FAULTS else "rec"
fault_type_str = "".join([str(i[0]).lower() for i in FAILURE_TYPES])

DATAPATH = f"AI/tree/{NUM_SIM_STEPS}i_{FAULT_RATE}fr_{FAULT_TIME}ft_{RECOVER_TIME}rt_{FAULT_INCREASE_TIME}fit_hosts{hosts_str}_{type_str}_{fault_type_str}/"
FIGURES_PATH = f"{DATAPATH}/figures/"
CSV_PATH = f"logs/MyFog_MyAzure2019Workload_{NUM_SIM_STEPS}_{HOSTS}_{CONTAINERS}_{ROUTER_BW}_{INTERVAL_TIME}_{NEW_CONTAINERS}/hostinfo_with_interval.csv"

NUMBER_OF_SIMULATIONS = 30

NUMBER_OF_REPETITIONS = 50

SAVE_SVG = False


def generate_datasets():
    """
    Generates datasets for the AI by calling the COSCO simulator
    Will generate NUMBER_OF_SIMULATIONS datasets

    """
    # create datapath folder if it doesn't exist

    os.makedirs(os.path.dirname(DATAPATH), exist_ok=True)
    os.makedirs(os.path.dirname(DATAPATH + "data/"), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH + "analysis/"), exist_ok=True)
    os.makedirs(os.path.dirname(FIGURES_PATH + "metrics/"), exist_ok=True)

    # old version (without multiprocessing)
    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        # skip if log file already exists
        if os.path.isfile(datapath_i):
            continue

        print(f"Creating DATA {i+1} of {NUMBER_OF_SIMULATIONS}")
        # run simulation
        runCOSCO(prints=False)

        # copy log file to datapath
        os.system(f"cp {CSV_PATH} {datapath_i}")


def plot_distribution(data, dataset_index):
    num_hosts = len(data)

    # limit color palette to number of hosts
    colors = sns.color_palette(COLOR_PALETTE, num_hosts)

    # count number of cpu failures
    counts_cpu = [list(host["cpufailures"].value_counts()) for host in data]

    # count number of ram failures
    counts_ram = [list(host["ramfailures"].value_counts()) for host in data]

    num_max_labels = max(
        [
            max([len(count)] for count in counts_cpu)[0],
            max([len(count) for count in counts_ram]),
        ]
    )
    # print(f'Number of labels: {num_max_labels}')

    for count in counts_cpu:
        while len(count) < num_max_labels:
            count.append(0)

    for count in counts_ram:
        while len(count) < num_max_labels:
            count.append(0)

    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)

    x = np.arange(num_max_labels)
    x_labels = [str(label) for label in range(num_max_labels)]

    width = 1 / ((num_hosts * 2) + 1)
    multiplier = 0

    for h_i in range(num_hosts):
        offset = width * multiplier
        multiplier += 1

        # cpu failures
        rects = ax.bar(
            x + offset,
            counts_cpu[h_i],
            width,
            label=f"Host {h_i}",
            color=colors[h_i],
        )

        for rect in rects:
            height = rect.get_height()

            if height > 0:
                ax.annotate(
                    f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        offset = width * (multiplier + num_hosts - 1)

        # ram failures
        rects = ax.bar(
            x + offset, counts_ram[h_i], width, hatch="///", color=colors[h_i]
        )

        for rect in rects:
            height = rect.get_height()

            if height > 0:
                ax.annotate(
                    f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    ax.set_xlabel("Stress Intensity")
    ax.set_ylabel("Number of Occurrences")

    ax.set_xticks(x + (num_hosts * 2 - 1) / 2 * width)
    ax.set_xticklabels(x_labels)

    # add invisible data to add second legend
    ax.bar(1, 0, color="gray", label="CPU")
    ax.bar(1, 0, color="gray", hatch="///", label="RAM")

    ax.legend(loc="best")

    plt.savefig(
        f"{FIGURES_PATH}analysis/individuals/data{dataset_index}/png/failure_distribution.png"
    )
    if SAVE_SVG:
        plt.savefig(
            f"{FIGURES_PATH}analysis/individuals/data{dataset_index}/svg/failure_distribution.svg"
        )


def plot_cpu_ram(data, dataset_index):
    num_hosts = len(data)
    individual_data_path = f"{FIGURES_PATH}analysis/individuals/data{dataset_index}/"

    def plot_usage_and_failures(component):
        component_failures = f"{component}failures"

        fig_hos, ax_hos = plt.subplots(
            nrows=num_hosts // 2,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(10, num_hosts),
        )

        fig_all, ax_all = plt.subplots(
            nrows=num_hosts,
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(10, 2 * num_hosts),
        )

        # each row represents a different host
        # x = ["interval"]
        # y = [component]
        # color intervals according to failure intensity [component_failures]

        most_failures = 0
        most_failures_index = 0

        scs_hos = [None for _ in range(num_hosts // 2)]

        for h_i in range(num_hosts):
            # print(f'Host {h_i}')
            # print(f'{component.upper()}: {data[h_i][component]}')
            # print(f'{component.upper()} Failures: {data[h_i][component_failures]}')

            ### INDIVIDUAL ###
            fig_ind, ax_ind = plt.subplots(figsize=(10, 5))
            ax_ind.plot(
                data[h_i]["interval"],
                data[h_i][component],
                color="black",
                label=f"{component.upper()} Usage",
            )

            if h_i in FAULTY_HOSTS:
                # component failures
                sc = ax_ind.scatter(
                    data[h_i]["interval"],
                    data[h_i][component],
                    c=data[h_i][component_failures],
                    cmap="magma_r",
                )

                fig_ind.legend(
                    *sc.legend_elements(),
                    bbox_to_anchor=(1.13, 0.62),
                    title="Stress Intensity",
                )

            ax_ind.set_xlabel("Interval")
            ax_ind.set_ylabel(f"{component.upper()} Usage (%)")

            ax_ind.set_xlim([0, len(data[h_i]["interval"])])
            # ax_ind.set_ylim([0, 100])

            fig_ind.legend(
                *ax_ind.get_legend_handles_labels(), bbox_to_anchor=(1.13, 0.72)
            )

            fig_ind.tight_layout()
            fig_ind.savefig(f"{individual_data_path}png/indiv/{component}_{h_i}.png")
            if SAVE_SVG:
                fig_ind.savefig(
                    f"{individual_data_path}svg/indiv/{component}_{h_i}.svg"
                )

            ### HOST PLOTS ###
            if h_i in FAULTY_HOSTS:
                if max(data[h_i][component_failures]) > most_failures:
                    most_failures = max(data[h_i][component_failures])
                    most_failures_index = h_i

                # component usage
                ax_hos[h_i // 2].plot(
                    data[h_i]["interval"],
                    data[h_i][component],
                    color="black",
                    label=f"{component.upper()} Usage",
                )

                # component failures
                scs_hos[h_i // 2] = ax_hos[h_i // 2].scatter(
                    data[h_i]["interval"],
                    data[h_i][component],
                    c=data[h_i][component_failures],
                    cmap="magma_r",
                )

                # ax_hos[h_i//2].set_ylim([0, 100])
                ax_hos[h_i // 2].set_xlim([0, len(data[h_i]["interval"])])
                ax_hos[h_i // 2].set_title(f"Host {h_i}")

            ### ALL PLOTS (with replicas) ###
            # component usage
            ax_all[h_i].plot(
                data[h_i]["interval"],
                data[h_i][component],
                color="black",
                label=f"{component.upper()} Usage",
            )

            if h_i in FAULTY_HOSTS:
                # component failures
                ax_all[h_i].scatter(
                    data[h_i]["interval"],
                    data[h_i][component],
                    c=data[h_i][component_failures],
                    cmap="magma_r",
                )

            # ax_all[h_i].set_ylim([0, 100])
            ax_all[h_i].set_xlim([0, len(data[h_i]["interval"])])

        # HOST PLOTS
        # xlabel and ylabel in the middle
        fig_hos.supylabel(f"{component.upper()} Usage (%)")
        fig_hos.supxlabel("Interval")

        fig_hos.legend(
            *ax_hos[0].get_legend_handles_labels(), bbox_to_anchor=(1.13, 0.72)
        )
        fig_hos.legend(
            *scs_hos[most_failures_index].legend_elements(),
            bbox_to_anchor=(1.13, 0.62),
            title="Stress Intensity",
        )

        fig_hos.tight_layout()
        fig_hos.savefig(f"{individual_data_path}png/{component}.png")
        if SAVE_SVG:
            fig_hos.savefig(f"{individual_data_path}png/{component}.svg")

        # ALL PLOTS
        # xlabel and ylabel in the middle
        fig_all.supylabel(f"{component.upper()} Usage (%)")
        fig_all.supxlabel("Interval")

        fig_all.legend(
            *ax_all[0].get_legend_handles_labels(), bbox_to_anchor=(1.13, 0.72)
        )
        fig_all.legend(
            *scs_hos[most_failures_index].legend_elements(),
            bbox_to_anchor=(1.13, 0.62),
            title="Stress Intensity",
        )

        fig_all.tight_layout()
        fig_all.savefig(f"{individual_data_path}png/{component}_all.png")
        if SAVE_SVG:
            fig_all.savefig(f"{individual_data_path}png/{component}_all.svg")

    plot_usage_and_failures("cpu")
    plot_usage_and_failures("ram")


def plot_data():
    os.makedirs(os.path.dirname(FIGURES_PATH + "analysis/"), exist_ok=True)
    individual_path = FIGURES_PATH + "analysis/individuals/"
    os.makedirs(os.path.dirname(individual_path), exist_ok=True)

    for i in range(NUMBER_OF_SIMULATIONS):
        print("plotting data for simulation", i)
        os.makedirs(os.path.dirname(f"{individual_path}data{i}/"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{individual_path}data{i}/png/"), exist_ok=True)
        os.makedirs(
            os.path.dirname(f"{individual_path}data{i}/png/indiv/"), exist_ok=True
        )
        os.makedirs(os.path.dirname(f"{individual_path}data{i}/svg/"), exist_ok=True)
        os.makedirs(
            os.path.dirname(f"{individual_path}data{i}/svg/indiv/"), exist_ok=True
        )

        datapath_i = f"{DATAPATH}data/data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = len(json.loads(data_temp["cpu"][0]))
        # print(f'Number of hosts: {num_hosts}')

        data_temp = data_temp.drop(columns=["disk", "diskavailable"])
        # get headers
        headers = data_temp.columns

        # create list of copies of data
        data = [data_temp.copy() for _ in range(num_hosts)]

        for j in range(len(data)):
            for header in headers:
                if header != "interval":
                    data[j][header] = data[j][header].apply(lambda x: json.loads(x)[j])

        plot_distribution(
            data[::2],  # only hosts, not replicas
            i,
        )

        plot_cpu_ram(data, i)

        return ()

        # plot number pf containers
        fig, ax = plt.subplots(
            nrows=num_hosts // 2, ncols=1, sharex=True, sharey=True, figsize=(10, 5)
        )

        for h_i in range(num_hosts):
            # INDIVIDUAL PLOT
            fig_in, ax_in = plt.subplots(figsize=(10, 5))
            ax_in.plot(data[h_i]["interval"], data[h_i]["numcontainers"])

            # SUBPLOT

            # component usage
            ax[h_i].plot(data[h_i]["interval"], data[h_i]["numcontainers"])

            # ax[h_i].set_ylim([0, 100])
            ax[h_i].set_xlim([0, len(data[h_i]["interval"])])
            ax[h_i].set_title(f"Host {h_i}")

        # ylabel in the middle
        ax[np.floor(num_hosts / 2).astype(int)].set_ylabel(
            f"Number of Containers", loc="center"
        )
        ax[-1].set_xlabel("Interval")
        plt.tight_layout()
        plt.savefig(f"{individual_path}data{i}/numcontainers.png")
        if SAVE_SVG:
            plt.savefig(f"{individual_path}data{i}/numcontainers.svg")


def merge_and_create_datasets():
    """
    Merge the data from all simulations into a single big dataset
    Merge the data from all simulations into a 2 datasets, one for each layer

    Saves the datasets in the datasets folder

    """

    os.makedirs(os.path.dirname(DATAPATH + "datasets/"), exist_ok=True)
    os.makedirs(os.path.dirname(DATAPATH + "datasets/cpu/"), exist_ok=True)
    os.makedirs(os.path.dirname(DATAPATH + "datasets/ram/"), exist_ok=True)

    IPS_CAP_DIVISOR = (
        10 * 2054
    )  # If the ips is above this value, it is a fog node, otherwise it is an edge node
    # This value was chosen by looking at the ips values of the nodes in the simulation
    # and choosing a value that would separate the fog nodes from the edge nodes

    # load and merge data
    fog_cpu_data = [None] * NUMBER_OF_SIMULATIONS
    fog_ram_data = [None] * NUMBER_OF_SIMULATIONS

    edge_cpu_data = [None] * NUMBER_OF_SIMULATIONS
    edge_ram_data = [None] * NUMBER_OF_SIMULATIONS

    cpu_headers = {
        "cpu": float,
        "numcontainers": int,
        "baseips": float,
        "ipsavailable": float,
        "ipscap": float,
        "apparentips": float,
        "cpufailures": int,
    }

    ram_headers = {
        "ram": float,
        "numcontainers": int,
        "ram_s": float,
        "ram_r": float,
        "ram_w": float,
        "ramavailable_s": float,
        "ramavailable_r": float,
        "ramavailable_w": float,
        "ramfailures": int,
        "ipscap": float,  # this is only in the ram data, so we can separate fog and edge nodes
    }

    for i in range(NUMBER_OF_SIMULATIONS):
        print("loading data for simulation", i)
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        data_temp_cpu = data_temp[cpu_headers.keys()]
        data_temp_ram = data_temp[ram_headers.keys()]

        data_temp_cpu = data_temp_cpu.applymap(lambda x: json.loads(x)[::2]).apply(
            pd.Series.explode
        )
        data_temp_ram = data_temp_ram.applymap(lambda x: json.loads(x)[::2]).apply(
            pd.Series.explode
        )

        # Divide by 2 the number of failures (each failure level corresponds to 2 containers)
        data_temp_cpu["cpufailures"] = data_temp_cpu["cpufailures"] // 2
        data_temp_ram["ramfailures"] = data_temp_ram["ramfailures"] // 2

        # separate fog and edge nodes based on ipscap
        fog_cpu_data[i] = data_temp_cpu[data_temp_cpu["ipscap"] > IPS_CAP_DIVISOR]
        edge_cpu_data[i] = data_temp_cpu[data_temp_cpu["ipscap"] < IPS_CAP_DIVISOR]

        fog_ram_data[i] = data_temp_ram[data_temp_ram["ipscap"] > IPS_CAP_DIVISOR].drop(
            columns=["ipscap"]
        )
        edge_ram_data[i] = data_temp_ram[
            data_temp_ram["ipscap"] < IPS_CAP_DIVISOR
        ].drop(columns=["ipscap"])

    # merge data
    fog_cpu_data = pd.concat(fog_cpu_data)
    edge_cpu_data = pd.concat(edge_cpu_data)
    # print(fog_cpu_data.head())
    # print(edge_cpu_data.head())

    fog_ram_data = pd.concat(fog_ram_data)
    edge_ram_data = pd.concat(edge_ram_data)
    # print(fog_ram_data.head())
    # print(edge_ram_data.head())

    all_cpu_data = pd.concat([fog_cpu_data, edge_cpu_data])
    all_ram_data = pd.concat([fog_ram_data, edge_ram_data])
    # print(all_cpu_data.head())
    # print(all_ram_data.head())

    # save data
    all_cpu_data.to_csv(f"{DATAPATH}datasets/cpu/all_cpu_data.csv", index=False)
    all_ram_data.to_csv(f"{DATAPATH}datasets/ram/all_ram_data.csv", index=False)

    fog_cpu_data.to_csv(f"{DATAPATH}datasets/cpu/fog_cpu_data.csv", index=False)
    fog_ram_data.to_csv(f"{DATAPATH}datasets/ram/fog_ram_data.csv", index=False)

    edge_cpu_data.to_csv(f"{DATAPATH}datasets/cpu/edge_cpu_data.csv", index=False)
    edge_ram_data.to_csv(f"{DATAPATH}datasets/ram/edge_ram_data.csv", index=False)


def eda(dataset_name, type_failure):
    """
    Performs exploratory data analysis on the dataset

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to perform EDA on
    type_failure : str
        Type of failure to perform EDA on (cpu or ram)
    """

    print("Performing EDA on", dataset_name)

    headers = None
    failure_str = None
    if type_failure == "cpu":
        headers = {
            "cpu": float,
            "numcontainers": int,
            "baseips": float,
            "ipsavailable": float,
            "ipscap": float,
            "apparentips": float,
            "cpufailures": int,
        }
        failure_str = "cpufailures"
    elif type_failure == "ram":
        headers = {
            "ram": float,
            "numcontainers": int,
            "ram_s": float,
            "ram_r": float,
            "ram_w": float,
            "ramavailable_s": float,
            "ramavailable_r": float,
            "ramavailable_w": float,
            "ramfailures": int,
        }
        failure_str = "ramfailures"
    else:
        raise ValueError("type_failure must be either cpu or ram")

    eda_path = FIGURES_PATH + "eda/"
    os.makedirs(os.path.dirname(eda_path), exist_ok=True)

    eda_path = eda_path + f"{type_failure}/"
    os.makedirs(os.path.dirname(eda_path), exist_ok=True)

    eda_path = eda_path + f"{dataset_name}/"
    os.makedirs(os.path.dirname(eda_path), exist_ok=True)

    os.makedirs(os.path.dirname(eda_path + "pairs/"), exist_ok=True)
    os.makedirs(os.path.dirname(eda_path + "dim_red/"), exist_ok=True)

    # load data
    data = pd.read_csv(f"{DATAPATH}datasets/{type_failure}/{dataset_name}.csv")

    print(data.shape)

    # attribute types
    data = data.astype(headers)

    # following https://www.digitalocean.com/community/tutorials/exploratory-data-analysis-python

    # 1. Basic Information
    print("\n---- INFO ----")
    print(data.info())

    print("\n---- DESCRIPTION ----")
    print(data.describe())

    # 2. Duplicate Values
    print(f"\n---- DUPLICATES: {data.duplicated().sum()}")

    # 2.1. Drop duplicates
    data.drop_duplicates(inplace=True)

    # 5. Missing Values
    print(f"\n---- MISSING VALUES:\n{data.isnull().sum()}")

    # Count number of failures
    print(f"\n---- FAILURES ----\n{data[failure_str].value_counts()}")

    # 10. Correlation Matrix
    plt.figure()
    _, ax = plt.subplots(figsize=(10, 9), tight_layout=True)
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt=".3f", ax=ax)
    plt.savefig(f"{eda_path}correlation_matrix.png")
    if SAVE_SVG:
        plt.savefig(f"{eda_path}correlation_matrix.svg")

    # plot every feature against failure_str
    for feature in data.columns:
        if feature != failure_str:
            # 2 subplots:
            # 1. scatter plot
            # 2. box plot

            plt.figure()
            fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

            # 1. scatter plot
            sns.scatterplot(x=failure_str, y=feature, data=data, ax=ax[0])

            # 2. box plot
            sns.boxplot(x=failure_str, y=feature, data=data, ax=ax[1])

            plt.savefig(f"{eda_path}pairs/{feature}_vs_numfailures.png")
            if SAVE_SVG:
                plt.savefig(f"{eda_path}pairs/{feature}_vs_numfailures.svg")

    plt.close("all")

    # Pairplot
    plt.figure()
    sns.pairplot(data, hue=failure_str)
    plt.savefig(f"{eda_path}pairs/pairplot.png")
    if SAVE_SVG:
        plt.savefig(f"{eda}pairs/pairplot.svg")

    plt.close("all")

    # FEATURE SELECTION

    # select k best features
    # https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning
    # the aforementioned tutorial mentions that, for numerical input and categorical output, we should use ANOVA Correlation Coefficient (linear) or Kendall's rank coefficient (non-linear)

    print("\n---- SELECT K BEST FEATURES ----")

    print("\n\t-- ANOVA --")

    best_features = SelectKBest(score_func=f_classif, k="all")

    fit = best_features.fit(data.drop(columns=[failure_str]), data[failure_str])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.drop(columns=[failure_str]).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]

    print(featureScores.sort_values(by="Score", ascending=False))

    """
    print('\n\t-- KENDALL --')

    best_features = SelectKBest(score_func=kendalltau, k='all')

    fit = best_features.fit(merged_big_data.drop(columns=[failure_str, 'ramfailures']), merged_big_data['ramfailures'])

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(merged_big_data.drop(columns=[failure_str, 'ramfailures']).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']

    print(featureScores.sort_values(by='Score', ascending=False))
    """

    """
    print("\n\t-- CHI2 --")

    best_features = SelectKBest(score_func=chi2, k="all")

    fit = best_features.fit(
        data.drop(columns=[failure_str]), data[failure_str].astype("category")
    )

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.drop(columns=[failure_str]).columns)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]

    print(featureScores.sort_values(by="Score", ascending=False))

    """
    
    # FEATURE IMPORTANCE
    print("\n---- FEATURE IMPORTANCE ----")

    model = ExtraTreesClassifier()
    model.fit(
        data.drop(columns=[failure_str]),
        data[failure_str],
    )

    print(model.feature_importances_)

    feat_importances = pd.Series(
        model.feature_importances_, data.drop(columns=[failure_str]).columns
    )
    print(feat_importances.sort_values(ascending=False))

    # FEATURE REDUCTION
    # PCA
    print("\n---- PCA ----")

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data.drop(columns=[failure_str]))
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    finalDf = pd.concat([principalDf, data[[failure_str]]], axis=1)

    print(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1",
        y="principal component 2",
        hue=failure_str,
        data=finalDf,
        s=100,
        alpha=0.75,
    )

    plt.savefig(f"{eda_path}dim_red/pca.png")
    if SAVE_SVG:
        plt.savefig(f"{eda_path}dim_red/pca.svg")

    # TSNE
    # it takes a lot of time to run (~650s)
    print("\n---- TSNE ----")

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(data.drop(columns=[failure_str]))

    tsneDf = pd.DataFrame(
        data=tsne_results, columns=["tsne component 1", "tsne component 2"]
    )

    finalDf = pd.concat([tsneDf, data[[failure_str]]], axis=1)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="tsne component 1",
        y="tsne component 2",
        hue=failure_str,
        data=finalDf,
        s=100,
        alpha=0.75,
    )

    plt.savefig(f"{eda_path}dim_red/tsne.png")
    if SAVE_SVG:
        plt.savefig(f"{eda_path}dim_red/tsne.svg")


def train_and_evaluate(
    data, y_col, model, data_test=None, binary=False, grid_search=False, param_grid=None
):
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
    grid_search : bool, optional
        If True, the model will be trained using grid search, by default False
    param_grid : dict, optional
        Dictionary with the parameters to use in grid search, by default None
        When grid_search is True, param_grid must be provided

    Returns
    -------
    list
        List of metrics [accuracy, precision, recall, f1]
    tuple
        Tuple with the best predicted values and respective f1 score
    list
        List of metrics [accuracy, precision, recall, f1] for the test data
    """

    # create a copy of the data
    data = data.copy()

    if binary:
        data[y_col] = data[y_col].apply(lambda x: 1 if x > 0 else 0)

    if grid_search:
        if not param_grid:
            print("param_grid must be provided for grid search")
            return

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

        ## NORMAL GRID SEARCH

        t = time.time()

        grid = GridSearchCV(
            model,
            param_grid,
            n_jobs=-1,
            cv=5,
            refit=True,
            return_train_score=True,
        )

        grid.fit(x_train, y_train)

        print(
            f"Grid search time: {time.time() - t:.2f} seconds for {len(grid.cv_results_['params'])} candidates parameter settings."
        )

        print(grid.best_params_)
        print(grid.best_estimator_)

        # save results
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(
            f"AI/balanced_tree/1000i_1.0fr_15ft_5rt_5fit_hosts024681012_acc_cr/grid_search_{y_col}_{'bin' if binary else 'multi'}_acc.csv",
            index=False,
        )

        # evaluate train data and test data

        print("Train data")
        y_pred = grid.predict(x_train)

        print(classification_report(y_train, y_pred))

        print("Test data")

        y_pred = grid.predict(x_test)

        print(classification_report(y_test, y_pred))

        ## GRID SEARCH WITH F1 SCORING

        t = time.time()

        grid = GridSearchCV(
            model,
            param_grid,
            scoring=make_scorer(f1_score, average="binary" if binary else "macro"),
            n_jobs=-1,
            cv=5,
            refit=True,
            return_train_score=True,
        )

        grid.fit(x_train, y_train)

        print(
            f"Grid search f1 time: {time.time() - t:.2f} seconds for {len(grid.cv_results_['params'])} candidates parameter settings."
        )

        print(grid.best_params_)
        print(grid.best_estimator_)

        # save results
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(
            f"AI/balanced_tree/1000i_1.0fr_15ft_5rt_5fit_hosts024681012_acc_cr/grid_search_{y_col}_{'bin' if binary else 'multi'}_f1.csv",
            index=False,
        )

        # evaluate train data and test data

        print("Train data")
        y_pred = grid.predict(x_train)

        print(classification_report(y_train, y_pred))

        print("Test data")

        y_pred = grid.predict(x_test)

        print(classification_report(y_test, y_pred))

        return None, None

    metrics = [[], [], [], []]
    metrics_train = [[], [], [], []]

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
        y_pred_train = model.predict(x_train)

        # evaluate
        metrics[0].append(accuracy_score(y_test, y_pred))
        metrics_train[0].append(accuracy_score(y_train, y_pred_train))
        
        metrics[1].append(
            precision_score(y_test, y_pred, average="binary" if binary else "macro")
        )
        metrics_train[1].append(
            precision_score(y_train, y_pred_train, average="binary" if binary else "macro")
        )

        metrics[2].append(
            recall_score(y_test, y_pred, average="binary" if binary else "macro")
        )
        metrics_train[2].append(
            recall_score(y_train, y_pred_train, average="binary" if binary else "macro")
        )

        metrics[3].append(
            f1_score(y_test, y_pred, average="binary" if binary else "macro")
        )
        metrics_train[3].append(
            f1_score(y_train, y_pred_train, average="binary" if binary else "macro")
        )

        # Why macro? https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
        # Because the classes are imbalanced, so we want to give the same importance to each class. average='weighted' would give more importance to the majority class

        # copilot sugested the following link: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

        if metrics[3][-1] > best_f1:
            best_f1 = metrics[3][-1]
            y_pred_best = y_pred

    return metrics, (y_pred_best, best_f1), metrics_train


def ai_rf(dataset_name, type_failure):
    """
    Run the Random Forest Classifier for the given dataset and type of failure

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    type_failure : str
        Type of failure to predict (cpu or ram)
    """

    print(f"AI for {dataset_name}")

    headers = None
    failure_str = None
    if type_failure == "cpu":
        headers = {
            "cpu": float,
            "numcontainers": int,
            "baseips": float,
            "ipsavailable": float,
            "ipscap": float,
            "apparentips": float,
            "cpufailures": int,
        }
        failure_str = "cpufailures"
    elif type_failure == "ram":
        headers = {
            "ram": float,
            "numcontainers": int,
            "ram_s": float,
            "ram_r": float,
            "ram_w": float,
            "ramavailable_s": float,
            "ramavailable_r": float,
            "ramavailable_w": float,
            "ramfailures": int,
        }
        failure_str = "ramfailures"
    else:
        raise ValueError("type_failure must be either cpu or ram")

    # load data
    data = pd.read_csv(f"{DATAPATH}datasets/{type_failure}/{dataset_name}.csv")

    # attribute types
    data = data.astype(headers)

    metrics, _ = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=False,
    )
    # binary classification
    metrics_bin, _ = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=True,
    )

    print(
        f"""\t{'METRICS ALL FEATURES':<48}\t{'METRICS ALL FEATURES (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        """
    )


def ai_rf_norm(dataset_name, type_failure):
    """
    Run the Random Forest Classifier for the given dataset and type of failure with normalised data

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    type_failure : str
        Type of failure to predict: only implemented for cpu
    """

    print(f"AI for {dataset_name}")

    headers = None
    failure_str = None
    if type_failure == "cpu":
        headers = {
            "cpu": float,
            "numcontainers": int,
            "baseips": float,
            "ipsavailable": float,
            "ipscap": float,
            "apparentips": float,
            "cpufailures": int,
        }
        failure_str = "cpufailures"
    else:
        raise ValueError("type_failure must only be cpu")

    # load data
    data = pd.read_csv(f"{DATAPATH}datasets/{type_failure}/{dataset_name}.csv")

    # attribute types
    data = data.astype(headers)

    # Normal AI
    print("AI for normal data")
    print(data.columns)

    time_start = time.time()
    metrics, _, metrics_train = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=False,
    )
    print(f"Time taken multiclasse: {time.time() - time_start:.2f} seconds")

    time_start = time.time()
    # binary classification
    metrics_bin, _, metrics_train_bin = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=True,
    )
    print(f"Time taken binary: {time.time() - time_start:.2f} seconds")

    print(
        f"""\t{'METRICS ALL FEATURES - TRAIN':<48}\t{'METRICS ALL FEATURES (binary) - TRAIN':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics_train[0]):<10.4f}{np.mean(metrics_train[1]):<10.4f}{np.mean(metrics_train[2]):<10.4f}{np.mean(metrics_train[3]):<10.4f}\t\t{np.mean(metrics_train_bin[0]):<10.4f}{np.mean(metrics_train_bin[1]):<10.4f}{np.mean(metrics_train_bin[2]):<10.4f}{np.mean(metrics_train_bin[3]):<10.4f}
        median\t{np.median(metrics_train[0]):<10.4f}{np.median(metrics_train[1]):<10.4f}{np.median(metrics_train[2]):<10.4f}{np.median(metrics_train[3]):<10.4f}\t\t{np.median(metrics_train_bin[0]):<10.4f}{np.median(metrics_train_bin[1]):<10.4f}{np.median(metrics_train_bin[2]):<10.4f}{np.median(metrics_train_bin[3]):<10.4f}
        std\t{np.std(metrics_train[0]):<10.4f}{np.std(metrics_train[1]):<10.4f}{np.std(metrics_train[2]):<10.4f}{np.std(metrics_train[3]):<10.4f}\t\t{np.std(metrics_train_bin[0]):<10.4f}{np.std(metrics_train_bin[1]):<10.4f}{np.std(metrics_train_bin[2]):<10.4f}{np.std(metrics_train_bin[3]):<10.4f}
        """
    )
    print(
        f"""\t{'METRICS ALL FEATURES - TEST':<48}\t{'METRICS ALL FEATURES (binary) - TEST':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        """
    )

    print(max(data["numcontainers"]))

    # normalise data
    print(data.head())
    data["cpu"] = data["cpu"] / 100
    data["numcontainers"] = data["numcontainers"] / max(data["numcontainers"])  # max(data["numcontainers"]) -> manually, but should change
    data["baseips"] = data["baseips"] / data["ipscap"]
    data["ipsavailable"] = data["ipsavailable"] / data["ipscap"]
    data["apparentips"] = data["apparentips"] / data["ipscap"]
    data["ipscap"] = data["ipscap"] / data["ipscap"]
    print(data.head())


    print("AI for normalised data")
    print(data.columns)

    time_start = time.time()
    metrics, _, metrics_train  = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=False,
    )
    print(f"Time taken multiclasse: {time.time() - time_start:.2f} seconds")

    time_start = time.time()
    # binary classification
    metrics_bin, _, metrics_train_bin = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=True,
    )
    print(f"Time taken binary: {time.time() - time_start:.2f} seconds")

    print(
        f"""\t{'METRICS ALL FEATURES - TRAIN':<48}\t{'METRICS ALL FEATURES (binary) - TRAIN':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics_train[0]):<10.4f}{np.mean(metrics_train[1]):<10.4f}{np.mean(metrics_train[2]):<10.4f}{np.mean(metrics_train[3]):<10.4f}\t\t{np.mean(metrics_train_bin[0]):<10.4f}{np.mean(metrics_train_bin[1]):<10.4f}{np.mean(metrics_train_bin[2]):<10.4f}{np.mean(metrics_train_bin[3]):<10.4f}
        median\t{np.median(metrics_train[0]):<10.4f}{np.median(metrics_train[1]):<10.4f}{np.median(metrics_train[2]):<10.4f}{np.median(metrics_train[3]):<10.4f}\t\t{np.median(metrics_train_bin[0]):<10.4f}{np.median(metrics_train_bin[1]):<10.4f}{np.median(metrics_train_bin[2]):<10.4f}{np.median(metrics_train_bin[3]):<10.4f}
        std\t{np.std(metrics_train[0]):<10.4f}{np.std(metrics_train[1]):<10.4f}{np.std(metrics_train[2]):<10.4f}{np.std(metrics_train[3]):<10.4f}\t\t{np.std(metrics_train_bin[0]):<10.4f}{np.std(metrics_train_bin[1]):<10.4f}{np.std(metrics_train_bin[2]):<10.4f}{np.std(metrics_train_bin[3]):<10.4f}
        """
    )
    print(
        f"""\t{'METRICS ALL FEATURES - TEST':<48}\t{'METRICS ALL FEATURES (binary) - TEST':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        """
    )


    # normalised data without redundant features
    print("AI for normalised data without redundant features")

    data = data.drop(columns=["cpu", "ipscap"])
    print(data.columns)

    time_start = time.time()
    metrics, _, metrics_train  = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=False,
    )
    print(f"Time taken multiclasse: {time.time() - time_start:.2f} seconds")

    time_start = time.time()
    # binary classification
    metrics_bin, _, metrics_train_bin = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=True,
    )
    print(f"Time taken binary: {time.time() - time_start:.2f} seconds")

    print(
        f"""\t{'METRICS ALL FEATURES - TRAIN':<48}\t{'METRICS ALL FEATURES (binary) - TRAIN':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics_train[0]):<10.4f}{np.mean(metrics_train[1]):<10.4f}{np.mean(metrics_train[2]):<10.4f}{np.mean(metrics_train[3]):<10.4f}\t\t{np.mean(metrics_train_bin[0]):<10.4f}{np.mean(metrics_train_bin[1]):<10.4f}{np.mean(metrics_train_bin[2]):<10.4f}{np.mean(metrics_train_bin[3]):<10.4f}
        median\t{np.median(metrics_train[0]):<10.4f}{np.median(metrics_train[1]):<10.4f}{np.median(metrics_train[2]):<10.4f}{np.median(metrics_train[3]):<10.4f}\t\t{np.median(metrics_train_bin[0]):<10.4f}{np.median(metrics_train_bin[1]):<10.4f}{np.median(metrics_train_bin[2]):<10.4f}{np.median(metrics_train_bin[3]):<10.4f}
        std\t{np.std(metrics_train[0]):<10.4f}{np.std(metrics_train[1]):<10.4f}{np.std(metrics_train[2]):<10.4f}{np.std(metrics_train[3]):<10.4f}\t\t{np.std(metrics_train_bin[0]):<10.4f}{np.std(metrics_train_bin[1]):<10.4f}{np.std(metrics_train_bin[2]):<10.4f}{np.std(metrics_train_bin[3]):<10.4f}
        """
    )
    print(
        f"""\t{'METRICS ALL FEATURES - TEST':<48}\t{'METRICS ALL FEATURES (binary) - TEST':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        """
    )

    # normalised data without redundant features and baseips

    print("AI for normalised data without redundant features and baseips")

    data = data.drop(columns=["baseips"])

    time_start = time.time()
    metrics, _, metrics_train  = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=False,
    )
    print(f"Time taken multiclasse: {time.time() - time_start:.2f} seconds")

    time_start = time.time()
    # binary classification
    metrics_bin, _, metrics_train_bin = train_and_evaluate(
        data,
        failure_str,
        RandomForestClassifier(n_jobs=-1),
        binary=True,
    )
    print(f"Time taken binary: {time.time() - time_start:.2f} seconds")

    print(
        f"""\t{'METRICS ALL FEATURES - TRAIN':<48}\t{'METRICS ALL FEATURES (binary) - TRAIN':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics_train[0]):<10.4f}{np.mean(metrics_train[1]):<10.4f}{np.mean(metrics_train[2]):<10.4f}{np.mean(metrics_train[3]):<10.4f}\t\t{np.mean(metrics_train_bin[0]):<10.4f}{np.mean(metrics_train_bin[1]):<10.4f}{np.mean(metrics_train_bin[2]):<10.4f}{np.mean(metrics_train_bin[3]):<10.4f}
        median\t{np.median(metrics_train[0]):<10.4f}{np.median(metrics_train[1]):<10.4f}{np.median(metrics_train[2]):<10.4f}{np.median(metrics_train[3]):<10.4f}\t\t{np.median(metrics_train_bin[0]):<10.4f}{np.median(metrics_train_bin[1]):<10.4f}{np.median(metrics_train_bin[2]):<10.4f}{np.median(metrics_train_bin[3]):<10.4f}
        std\t{np.std(metrics_train[0]):<10.4f}{np.std(metrics_train[1]):<10.4f}{np.std(metrics_train[2]):<10.4f}{np.std(metrics_train[3]):<10.4f}\t\t{np.std(metrics_train_bin[0]):<10.4f}{np.std(metrics_train_bin[1]):<10.4f}{np.std(metrics_train_bin[2]):<10.4f}{np.std(metrics_train_bin[3]):<10.4f}
        """
    )
    print(
        f"""\t{'METRICS ALL FEATURES - TEST':<48}\t{'METRICS ALL FEATURES (binary) - TEST':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        """
    )


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
    plt.xticks([1, 2, 3, 4], ["Accuracy", "Precision", "Recall", "F1"])
    plt.savefig(f"{FIGURES_PATH}metrics/{name}.png")
    if SAVE_SVG:
        plt.savefig(f"{FIGURES_PATH}metrics/{name}.svg")


if __name__ == "__main__":
    time_start = time.time()

    # generate_datasets()
    # print(f"Time taken with generation: {time.time() - time_start}")

    # plot_data()

    # split into datasets
    # merge_and_create_datasets()

    failures = ["cpu", "ram"]
    failures = ["cpu"]
    layers = ["edge", "fog", "all"]

    # loop
    for failure in failures:
        for layer in layers:
            # eda(f'{layer}_{failure}_data', failure)
            # ai_rf(f"{layer}_{failure}_data", failure)
            ai_rf_norm(f"{layer}_{failure}_data", failure)

    print(f"Total time taken: {time.time() - time_start}")
