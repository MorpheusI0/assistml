import os
import pandas as pd
from flask import current_app
from sklearn.cluster import DBSCAN
from typing import List, Dict, Union


def cluster_models(selected_models: List[Dict[str, Union[str, float]]], preferences: Dict[str, float]) -> Dict[str, Union[List[str], str, Dict[str, int]]]:
    """
    Cluster models with DBSCAN and selects acceptable and nearly acceptable models based on the stated user performance preferences.

    Parameters:
    selected_models (List[Dict[str, Union[str, float]]]): List containing selected models.
    preferences (Dict[str, float]): Dict with performance preferences.

    Returns:
    Dict[str, Union[pd.DataFrame, str, Dict[str, int]]]: List with the two groups of models "acceptable" and "nearly acceptable".
    """
    verbose = current_app.config['VERBOSE']

    if verbose:
        current_app.logger.info("Selected models include")
        current_app.logger.info(selected_models[:5])

    # Convert the list of dictionaries to a pandas DataFrame
    selected_models_df = pd.DataFrame(selected_models)

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    cluster_txt_path = os.path.join(working_dir, "cluster.txt")
    cluster_dbscan_path = os.path.join(working_dir, "cluster_dbscan.csv")
    cluster_nacc_path = os.path.join(working_dir, "cluster_nacc.csv")
    cluster_acc_path = os.path.join(working_dir, "cluster_acc.csv")

    if os.path.exists(cluster_txt_path):
        os.remove(cluster_txt_path)

    with open(cluster_txt_path, 'w') as f:
        f.write("Cluster_models() results \n\n")

    cluster_exp = DBSCAN(eps=0.05, min_samples=3, algorithm='kd_tree').fit(selected_models_df[['accuracy', 'precision', 'recall', 'training_time_std']])
    if not isinstance(cluster_exp, DBSCAN):
        raise ValueError("Error: Clustering failed")
    selected_models_df['dbscan'] = cluster_exp.labels_

    selected_models_df.to_csv(cluster_dbscan_path, index=False)

    majority_ratio = 0.51

    maxacc = selected_models_df['accuracy'].max()
    maxrec = selected_models_df['recall'].max()
    maxpre = selected_models_df['precision'].max()
    maxtra = selected_models_df['training_time_std'].max()

    minacc = maxacc * (1 - preferences['acc_width'])
    minrec = maxrec * (1 - preferences['rec_width'])
    minpre = maxpre * (1 - preferences['pre_width'])
    mintra = maxtra * (1 - preferences['tra_width'])

    current_app.logger.info("Checking fit inside the acceptable region")
    cluster_fit = []
    # Loop to compute for all cluster groups found how many models actually fall inside the acceptable range in each of the dimensions.
    for i in range(selected_models_df['dbscan'].max() + 1):
        a = (selected_models_df.loc[selected_models_df['dbscan'] == i, 'accuracy'] > minacc).mean()
        b = (selected_models_df.loc[selected_models_df['dbscan'] == i, 'precision'] > minpre).mean()
        c = (selected_models_df.loc[selected_models_df['dbscan'] == i, 'recall'] > minrec).mean()
        d = (selected_models_df.loc[selected_models_df['dbscan'] == i, 'training_time_std'] > mintra).mean()

        # Saves an average value for acceptable models from the three perspectives for all cluster groups
        fit = (a + b + c + d) / 4
        current_app.logger.info(f"Cluster {i} is {fit * 100:.2f}% inside the Acceptable region")
        cluster_fit.append(fit)

    cluster_fit = pd.Series(cluster_fit, index=range(selected_models_df['dbscan'].max() + 1))
    # So the selected cluster is...
    clusters_acc = cluster_fit[cluster_fit > majority_ratio]
    current_app.logger.info("So the selected cluster(s) are...")
    current_app.logger.info(clusters_acc)

    # Analysis files for cluster discussion ####
    with open(cluster_txt_path, 'a') as f:
        f.write("Acceptable models \n")
        f.write(str(cluster_fit.index.tolist()))
        f.write(str(cluster_fit.tolist()))


    # Calculate how many clusters are completely inside the region

    # How many clusters are completely inside the region over the number of clusters in ACC
    inside_acc = (clusters_acc == 1).sum() / len(clusters_acc)
    acc_distrustpts = 0 if inside_acc == 1 else 1 if 0.5 <= inside_acc < 1 else 2 if 0 < inside_acc < 0.5 else 3

    current_app.logger.info(f"Distrust points in ACC region {acc_distrustpts}")

    # Section :Checking NACC models####
    subacc = maxacc * (1 - (preferences['acc_width'] * 2))
    subrec = maxrec * (1 - (preferences['rec_width'] * 2))
    subpre = maxpre * (1 - (preferences['pre_width'] * 2))
    subtra = maxtra * (1 - (preferences['tra_width'] * 2))

    current_app.logger.info("Checking fit inside the nearly acceptable region")
    cluster_misfit = []
    for i in range(selected_models_df['dbscan'].max() + 1):
        l = ((selected_models_df.loc[selected_models_df['dbscan'] == i, 'accuracy'] > subacc) &
             (selected_models_df.loc[selected_models_df['dbscan'] == i, 'accuracy'] < minacc)).mean()
        m = ((selected_models_df.loc[selected_models_df['dbscan'] == i, 'precision'] > subpre) &
             (selected_models_df.loc[selected_models_df['dbscan'] == i, 'precision'] < minpre)).mean()
        n = ((selected_models_df.loc[selected_models_df['dbscan'] == i, 'recall'] > subrec) &
             (selected_models_df.loc[selected_models_df['dbscan'] == i, 'recall'] < minrec)).mean()
        o = ((selected_models_df.loc[selected_models_df['dbscan'] == i, 'training_time_std'] > subtra) &
             (selected_models_df.loc[selected_models_df['dbscan'] == i, 'training_time_std'] < mintra)).mean()

        misfit = (l + m + n + o) / 4
        current_app.logger.info(f"Cluster {i} is {misfit * 100:.2f}% inside the Nearly-Acceptable region")
        cluster_misfit.append(misfit)

    cluster_misfit = pd.Series(cluster_misfit, index=range(selected_models_df['dbscan'].max() + 1))
    # So the selected cluster is...
    clusters_nacc = cluster_misfit[cluster_misfit > majority_ratio]
    current_app.logger.info("So the selected cluster(s) are...")
    current_app.logger.info(clusters_nacc)

    # Files for analysis of cluster distribution####
    with open(cluster_txt_path, 'a') as f:
        f.write("\n Nearly acceptable models \n")
        f.write(str(cluster_misfit.index.tolist()))
        f.write(str(cluster_misfit.tolist()))

    selected_cluster = clusters_acc.index.tolist()
    selected_anticluster = clusters_nacc.index.tolist()

    if len(clusters_nacc) > 0:
        # If at least one cluster is chosen for NACC group

        # Calculate how many clusters are completely inside the region

        # How many clusters are completely inside the region over the number of clusters in ACC
        inside_nacc = (clusters_nacc == 1).sum() / len(clusters_nacc)
        nacc_distrustpts = 0 if inside_nacc == 1 else 1 if 0.5 <= inside_nacc < 1 else 2 if 0 < inside_nacc < 0.5 else 3

        current_app.logger.info(f"Distrust points in NACC region {nacc_distrustpts}")

        selected_models_df[selected_models_df['dbscan'].isin(selected_anticluster)][['model_name']].to_csv(cluster_nacc_path, index=False)

        return {
            "acceptable_models": selected_models_df[selected_models_df['dbscan'].isin(selected_cluster)]['model_name'].tolist(),
            "nearly_acceptable_models": selected_models_df[selected_models_df['dbscan'].isin(selected_anticluster)]['model_name'].tolist(),
            "distrust": {
                "acceptable_models": acc_distrustpts,
                "nearly_acceptable_models": nacc_distrustpts
            }
        }
    else:
        if verbose:
            current_app.logger.info("No cluster could be added to the NACC group")

        return {
            "acceptable_models": selected_models_df[selected_models_df['dbscan'].isin(selected_cluster)]['model_name'].tolist(),
            "nearly_acceptable_models": "none",
            "distrust": {
                "acceptable_models": acc_distrustpts,
                "nearly_acceptable_models": 0
            }
        }

    #selected_models_df[selected_models_df['dbscan'].isin(selected_cluster)][['model_name']].to_csv(cluster_acc_path, index=False)
