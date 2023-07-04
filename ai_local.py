# Python 3.11.4

import time
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# sns color palette
COLOR_PALETTE = "hls"
sns.set_palette(COLOR_PALETTE)


from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.svm import SVC


DATAPATH = f"AI/backups/1000i_0.3fr_6ft_18rt_2fit_hosts012_acc_cr/"
FIGURES_PATH = f"{DATAPATH}/figures/"

NUMBER_OF_SIMULATIONS = 30

NUMBER_OF_REPETITIONS = 50


def big_merged_data_eda():
    # Exploratory Data Analysis on the merged data

    big_analysis_path = FIGURES_PATH + "analysis/big_merged_data_eda/"
    os.makedirs(os.path.dirname(big_analysis_path), exist_ok=True)

    os.makedirs(os.path.dirname(big_analysis_path + "pairs/"), exist_ok=True)
    os.makedirs(os.path.dirname(big_analysis_path + "pairs/cpu/"), exist_ok=True)
    os.makedirs(os.path.dirname(big_analysis_path + "pairs/ram/"), exist_ok=True)

    os.makedirs(os.path.dirname(big_analysis_path + "dim_red/"), exist_ok=True)


    # load and merge data

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
    }

    # create list for all dataframes
    data_cpu = [None] * NUMBER_OF_SIMULATIONS
    data_ram = [None] * NUMBER_OF_SIMULATIONS

    for i in range(NUMBER_OF_SIMULATIONS):
        datapath_i = f"{DATAPATH}data/data{i}.csv"
        data_temp = pd.read_csv(datapath_i)

        num_hosts = int(len(json.loads(data_temp["cpu"][0])) / 2)

        data_temp_cpu = data_temp[cpu_headers.keys()]
        data_temp_ram = data_temp[ram_headers.keys()]

        data_temp_cpu = data_temp_cpu.applymap(
            lambda x: json.loads(x)[:num_hosts]
        ).apply(pd.Series.explode)
        data_temp_ram = data_temp_ram.applymap(
            lambda x: json.loads(x)[:num_hosts]
        ).apply(pd.Series.explode)

        data_cpu[i] = data_temp_cpu
        data_ram[i] = data_temp_ram

    merged_big_data_cpu = pd.concat(data_cpu, ignore_index=True)
    merged_big_data_ram = pd.concat(data_ram, ignore_index=True)

    print(merged_big_data_cpu.shape, merged_big_data_ram.shape)
    # (90090, 7) (90090, 9)

    # 0. Divide by 2 the number of failures (each failure level corresponds to 2 containers)
    merged_big_data_cpu["cpufailures"] = merged_big_data_cpu["cpufailures"] // 2
    merged_big_data_ram["ramfailures"] = merged_big_data_ram["ramfailures"] // 2

    # attribute types
    merged_big_data_cpu = merged_big_data_cpu.astype(cpu_headers)
    merged_big_data_ram = merged_big_data_ram.astype(ram_headers)

    # following https://www.digitalocean.com/community/tutorials/exploratory-data-analysis-python

    # 1. Basic Information

    print("\n---- INFO ----")
    print("CPU:")
    print(merged_big_data_cpu.info())
    print("\nRAM:")
    print(merged_big_data_ram.info())

    print("\n---- DESCRIPTION ----")
    print("CPU:\n", merged_big_data_cpu.describe())
    print("\nRAM:\n", merged_big_data_ram.describe())

    #   ---- DESCRIPTION ----
    #   CPU:
    #                 cpu  numcontainers      baseips  ipsavailable       ipscap  apparentips  cpufailures
    #   count    90090.0        90090.0      90090.0       90090.0      90090.0      90090.0      90090.0
    #   mean   17.614404       7.298912   755.227653   8658.772347       9414.0  1430.991764     0.144655
    #   std     8.116511       2.488022   406.037958   4828.979276  5018.971237   686.680416     0.425868
    #   min          0.0            0.0          0.0   2249.415044       4029.0          0.0          0.0
    #   25%    11.764706            6.0   458.937091   3675.184072       4029.0        924.0          0.0
    #   50%    16.207496            7.0   682.406998   7397.433305       8102.0       1304.0          0.0
    #   75%    22.154409            9.0   979.996389  14807.924211      16111.0       1825.0          0.0
    #   max    65.053363           20.0  3439.983182       16111.0      16111.0       5678.0          3.0
    #   
    #   RAM:
    #                 ram  numcontainers       ram_s      ram_r      ram_w  ramavailable_s  ramavailable_r  ramavailable_w  ramfailures
    #   count    90090.0        90090.0     90090.0    90090.0    90090.0         90090.0         90090.0         90090.0      90090.0
    #   mean      4.0001       7.298912  506.462041   1.391615   1.139732    18105.204626      368.121718      256.110268     0.144367
    #   std     3.672401       2.488022  502.913795   4.792341   4.561932    12171.185337        8.151337       43.073732     0.424373
    #   min          0.0            0.0         0.0        0.0        0.0       2852.3406      290.007467      197.101333          0.0
    #   25%     1.217631            6.0  186.818967     0.0152     0.0092     4129.343833        359.9776      199.991867          0.0
    #   50%     2.901928            7.0    333.4441      0.038   0.030467    16799.478733        371.9444        266.7152          0.0
    #   75%     5.695342            9.0  604.677533   0.229067   0.196667    33291.646067      375.535067      304.739167          0.0
    #   max    33.589276           20.0   4966.5762  74.025333  69.648667         34360.0          376.54           305.0          3.0

    # 2. Duplicate Values
    print(
        f"\n---- DUPLICATES:\tCPU: {merged_big_data_cpu.duplicated().sum()}\tRAM: {merged_big_data_ram.duplicated().sum()}"
    )
    #   ---- DUPLICATES:        CPU: 279        RAM: 274

    # 2.1. Drop duplicates
    merged_big_data_cpu.drop_duplicates(inplace=True)
    merged_big_data_ram.drop_duplicates(inplace=True)

    # 5. Missing Values
    print(
        f"\n---- MISSING VALUES ----\nCPU:\n{merged_big_data_cpu.isnull().sum()}\nRAM:\n{merged_big_data_ram.isnull().sum()}"
    )
    #   None

    """

    # 10. Correlation Matrix
    plt.figure()
    _, ax = plt.subplots(figsize=(10, 9), tight_layout=True)
    corr = merged_big_data_cpu.corr()
    sns.heatmap(corr, annot=True, fmt=".3f", ax=ax)
    plt.savefig(f"{big_analysis_path}correlation_matrix_cpu.png")
    plt.savefig(f"{big_analysis_path}correlation_matrix_cpu.svg")

    # Correlation Matrix shows that there is no strong correlation between cpufailures and [numcontainers, baseips, ipsavailable, ipscap, host_ltype]
    # With this information, we will try to predict cpufailures using all the features and compare it to the results of using only the features that have a correlation with cpufailures
    #   wich are [cpu, apparentips]

    plt.figure()
    _, ax = plt.subplots(figsize=(10, 9), tight_layout=True)
    corr = merged_big_data_ram.corr()
    sns.heatmap(corr, annot=True, fmt=".3f", ax=ax)
    plt.savefig(f"{big_analysis_path}correlation_matrix_ram.png")
    plt.savefig(f"{big_analysis_path}correlation_matrix_ram.svg")

    # Correlation Matrix shows no strong correlation between ramfailures and others
    plt.close('all')

    # plot every feature against cpufailures
    for feature in merged_big_data_cpu.columns:
        if feature != 'cpufailures':
            # 2 subplots:
                # 1. scatter plot
                # 2. box plot

            plt.figure()
            fig, ax = plt.subplots(1,2, figsize=(10, 5), tight_layout=True)


            # 1. scatter plot
            sns.scatterplot(x='cpufailures', y=feature, data=merged_big_data_cpu, ax=ax[0])

            # 2. box plot
            sns.boxplot(x='cpufailures', y=feature, data=merged_big_data_cpu, ax=ax[1])

            plt.savefig(f'{big_analysis_path}pairs/cpu/{feature}_vs_numfailures.png')
            plt.savefig(f'{big_analysis_path}pairs/cpu/{feature}_vs_numfailures.svg')

    for feature in merged_big_data_ram.columns:
        if feature != 'ramfailures':
            # 2 subplots:
                # 1. scatter plot
                # 2. box plot

            plt.figure()
            fig, ax = plt.subplots(1,2, figsize=(10, 5), tight_layout=True)


            # 1. scatter plot
            sns.scatterplot(x='ramfailures', y=feature, data=merged_big_data_ram, ax=ax[0])

            # 2. box plot
            sns.boxplot(x='ramfailures', y=feature, data=merged_big_data_ram, ax=ax[1])

            plt.savefig(f'{big_analysis_path}pairs/ram/{feature}_vs_numfailures.png')
            plt.savefig(f'{big_analysis_path}pairs/ram/{feature}_vs_numfailures.svg')

    plt.close('all')

    
    # Pairplot
    plt.figure()
    sns.pairplot(merged_big_data_cpu, hue='cpufailures')
    plt.savefig(f'{big_analysis_path}pairs/pairplot_cpu.png')
    plt.savefig(f'{big_analysis_path}pairs/pairplot_cpu.svg')

    plt.figure()
    sns.pairplot(merged_big_data_ram, hue='ramfailures')
    plt.savefig(f'{big_analysis_path}pairs/pairplot_ram.png')
    plt.savefig(f'{big_analysis_path}pairs/pairplot_ram.svg')
    plt.close('all')

    """
    # FEATURE SELECTION - CPU

    # select k best features
    # https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning
    # the aforementioned tutorial mentions that, for numerical input and categorical output, we should use ANOVA Correlation Coefficient (linear) or Kendall's rank coefficient (non-linear)

    '''
    print("\n---- SELECT K BEST FEATURES - CPU ----")

    print("\n\t-- ANOVA --")

    best_features = SelectKBest(score_func=f_classif, k="all")

    fit = best_features.fit(
        merged_big_data_cpu.drop(columns=["cpufailures"]),
        merged_big_data_cpu["cpufailures"]
    )

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(
        merged_big_data_cpu.drop(columns=["cpufailures"]).columns
    )

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]

    print(featureScores.sort_values(by="Score", ascending=False))

    #              Specs        Score
    #   0            cpu  1657.570152
    #   5    apparentips  1639.296489
    #   1  numcontainers     4.107048
    #   2        baseips     2.212147
    #   3   ipsavailable     0.710469
    #   4         ipscap     0.541987

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

    print("\n\t-- CHI2 --")

    best_features = SelectKBest(score_func=chi2, k="all")

    fit = best_features.fit(
        merged_big_data_cpu.drop(columns=["cpufailures"]),
        merged_big_data_cpu["cpufailures"].astype("category")
    )

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(
        merged_big_data_cpu.drop(columns=["cpufailures"]).columns
    )

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ["Specs", "Score"]

    print(featureScores.sort_values(by="Score", ascending=False))

    #              Specs         Score
    #   5    apparentips  1.516488e+06
    #   0            cpu  1.737074e+04
    #   3   ipsavailable  5.738748e+03
    #   4         ipscap  4.353412e+03
    #   2        baseips  1.433564e+03
    #   1  numcontainers  1.017751e+01


    # FEATURE IMPORTANCE - CPU
    print("\n---- FEATURE IMPORTANCE ----")

    model = ExtraTreesClassifier()
    model.fit(
        merged_big_data_cpu.drop(columns=["cpufailures"]),
        merged_big_data_cpu["cpufailures"],
    )

    print(model.feature_importances_)

    feat_importances = pd.Series(
        model.feature_importances_,
        merged_big_data_cpu.drop(columns=["cpufailures"]).columns
    )
    print(feat_importances.sort_values(ascending=False))

    #   numcontainers    0.296458
    #   cpu              0.210491
    #   apparentips      0.207965
    #   ipsavailable     0.139228
    #   baseips          0.136735
    #   ipscap           0.009123

    '''

    """
    # PCA
    print("\n---- PCA ----")

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(
        merged_big_data_cpu.drop(columns=["cpufailures"])
    )
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    finalDf = pd.concat(
        [principalDf, merged_big_data_cpu[["cpufailures"]]], axis=1
    )

    print(pca.explained_variance_ratio_)
    # [0.99000531 0.00887902]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1",
        y="principal component 2",
        hue="cpufailures",
        data=finalDf,
        s=100,
        alpha=0.75,
    )

    plt.savefig(f"{big_analysis_path}dim_red/pca_cpu.png")
    plt.savefig(f"{big_analysis_path}dim_red/pca_cpu.svg")

    # TSNE
    # it takes a lot of time to run (~650s)
    print("\n---- TSNE ----")

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(
        merged_big_data_cpu.drop(columns=["cpufailures"])
    )
    
    tsneDf = pd.DataFrame(
        data=tsne_results, columns=["tsne component 1", "tsne component 2"]
    )

    finalDf = pd.concat(
        [tsneDf, merged_big_data_cpu[["cpufailures"]]], axis=1
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="tsne component 1",
        y="tsne component 2",
        hue="cpufailures",
        data=finalDf,
        s=100,
        alpha=0.75,
    )

    plt.savefig(f"{big_analysis_path}dim_red/tsne_cpu.png")
    plt.savefig(f"{big_analysis_path}dim_red/tsne_cpu.svg")

    """

    """
    # AI

    # Train and Evaluate with all features - CPU
    metrics, _ = train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        binary=False,
    )
    # binary classification
    metrics_bin, _ = train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        binary=True,
    )

    print(
        f'''\t{'METRICS ALL FEATURES':<48}\t{'METRICS ALL FEATURES (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        '''
    )

    #   METRICS ALL FEATURES                                METRICS ALL FEATURES (binary)
    #           accuracy  precision recall    f1        		accuracy  precision recall    f1
    #   mean	0.9318    0.7297    0.5922    0.6384    		0.9459    0.8306    0.6780    0.7466
    #   median	0.9317    0.7359    0.5920    0.6388    		0.9460    0.8302    0.6773    0.7462
    #   std	    0.0012    0.0240    0.0119    0.0145    		0.0010    0.0074    0.0069    0.0051

    # Train and Evaluate with all features - RAM
    metrics, _ = train_and_evaluate(
        merged_big_data_ram,
        "ramfailures",
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        binary=False,
    )
    # binary classification
    metrics_bin, _ = train_and_evaluate(
        merged_big_data_ram,
        "ramfailures",
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        binary=True,
    )

    print(
        f'''t{'METRICS ALL FEATURES':<48}\t{'METRICS ALL FEATURES (binary)':<48}
        \t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}\t\t{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}
        mean\t{np.mean(metrics[0]):<10.4f}{np.mean(metrics[1]):<10.4f}{np.mean(metrics[2]):<10.4f}{np.mean(metrics[3]):<10.4f}\t\t{np.mean(metrics_bin[0]):<10.4f}{np.mean(metrics_bin[1]):<10.4f}{np.mean(metrics_bin[2]):<10.4f}{np.mean(metrics_bin[3]):<10.4f}
        median\t{np.median(metrics[0]):<10.4f}{np.median(metrics[1]):<10.4f}{np.median(metrics[2]):<10.4f}{np.median(metrics[3]):<10.4f}\t\t{np.median(metrics_bin[0]):<10.4f}{np.median(metrics_bin[1]):<10.4f}{np.median(metrics_bin[2]):<10.4f}{np.median(metrics_bin[3]):<10.4f}
        std\t{np.std(metrics[0]):<10.4f}{np.std(metrics[1]):<10.4f}{np.std(metrics[2]):<10.4f}{np.std(metrics[3]):<10.4f}\t\t{np.std(metrics_bin[0]):<10.4f}{np.std(metrics_bin[1]):<10.4f}{np.std(metrics_bin[2]):<10.4f}{np.std(metrics_bin[3]):<10.4f}
        '''
    )

    #   METRICS ALL FEATURES                            	METRICS ALL FEATURES (binary)
    #   	    accuracy  precision recall    f1        		accuracy  precision recall    f1
    #   mean	0.8801    0.2511    0.2500    0.2355    		0.8793    0.1341    0.0044    0.0085
    #   median	0.8801    0.2457    0.2500    0.2353    		0.8794    0.1299    0.0042    0.0082
    #   std	    0.0014    0.0180    0.0003    0.0006    		0.0015    0.0316    0.0011    0.0022



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
    # plot_metrics(metrics, 'big_merged_data_all_features')

    # Train and Evaluate with only the features that have a correlation with cpufailures
    # metrics, _ = train_and_evaluate(merged_big_data[['cpu','apparentips', 'cpufailures']], 'cpufailures', RandomForestClassifier(n_estimators=100, n_jobs=-1), binary=False)

    # plot metrics
    # plot_metrics(metrics, 'big_merged_data_correlated_features')

    # """

    # Remove ltype and ips cap?
    # normalize data?
    # remove outliers? is there any?

    """
    # CPU SVM with grid search
    params = {
        "C": [0.1, 1, 10, 100, 1000],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
    }

    # Train and Evaluate with all features
    print("CPU")
    print("all features")
    train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        SVC(),
        binary=False,
        grid_search=True,
        param_grid=params,
    )
    # binary classification
    print("binary classification")
    train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        SVC(),
        binary=True,
        grid_search=True,
        param_grid=params,
    )

    # RAM
    print("RAM")
    # Train and Evaluate with all features
    print("all features")
    train_and_evaluate(
        merged_big_data_ram,
        "ramfailures",
        SVC(),
        binary=False,
        grid_search=True,
        param_grid=params,
    )
    # binary classification
    print("binary classification")
    train_and_evaluate(
        merged_big_data_ram,
        "ramfailures",
        SVC(),
        binary=True,
        grid_search=True,
        param_grid=params,
    )
    """

    # Tunning Random Forest
    params = {
        "n_estimators": [10, 100, 1000],
        "max_features": ["auto", "sqrt", "log2"],
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "criterion": ["gini", "entropy"],
    }

    # Train and Evaluate with all features
    print("CPU")
    print("all features")
    train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        RandomForestClassifier(),
        binary=False,
        grid_search=True,
        param_grid=params,
    )
    # binary classification
    print("binary classification")
    train_and_evaluate(
        merged_big_data_cpu,
        "cpufailures",
        RandomForestClassifier(),
        binary=True,
        grid_search=True,
        param_grid=params,
    )

    




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

    Returns
    -------
    list
        List of metrics [accuracy, precision, recall, f1]
    tuple
        Tuple with the best predicted values and respective f1 score
    """

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

        grid = GridSearchCV(model, param_grid, refit=True, verbose=3, n_jobs=-1)

        grid.fit(x_train, y_train)

        print(grid.best_params_)
        print(grid.best_estimator_)

        y_pred = grid.predict(x_test)

        print(classification_report(y_test, y_pred))

        return

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
        metrics[1].append(
            precision_score(y_test, y_pred, average="binary" if binary else "macro")
        )
        metrics[2].append(
            recall_score(y_test, y_pred, average="binary" if binary else "macro")
        )
        metrics[3].append(
            f1_score(y_test, y_pred, average="binary" if binary else "macro")
        )

        # Why macro? https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
        # Because the classes are imbalanced, so we want to give the same importance to each class. average='weighted' would give more importance to the majority class

        # copilot sugested the following link: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin

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
    plt.xticks([1, 2, 3, 4], ["Accuracy", "Precision", "Recall", "F1"])
    plt.savefig(f"{FIGURES_PATH}metrics/{name}.png")
    plt.savefig(f"{FIGURES_PATH}metrics/{name}.svg")


def test():
    return


if __name__ == "__main__":
    time_start = time.time()

    big_merged_data_eda()

    # test()

    # multiprocessing_test()

    print(f"Time taken: {time.time() - time_start}")
