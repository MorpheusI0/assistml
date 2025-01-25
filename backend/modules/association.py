"""
@author: Dinesh Subhuraaj
"""

import glob
import json
import os
from collections import OrderedDict

import pandas as pd
from quart import current_app
# Imports for mlxtend module
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def _ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def _get_next_experiment_number(directory) -> int:
    max_count = 0

    for filename in glob.glob(os.path.join(directory, '*.json')):
        base_name = os.path.basename(filename).split('.')[0]
        number = ''.join(filter(str.isdigit, base_name))
        max_count = max(max_count, int(number) if number.isdigit() else 0)
    return max_count + 1

def _create_save_json_output(rules, ranking_metric, metric_min_score, min_support, total_nr_models, output_dir) -> None:
    _ensure_directory_exists(output_dir)
    experiment_number = _get_next_experiment_number(output_dir)
    no_of_rules_generated = len(rules.index)

    # two sub dicts are stored in the super dict
    current_app.logger.info("Found " + str(no_of_rules_generated) + " rules !!")
    dict_super = OrderedDict()  # TODO: Check if order is really needed
    dict_experiment = OrderedDict({
        "experiment_nr": experiment_number,
        "total_models": total_nr_models,
        "rules_found": no_of_rules_generated,
        "ranking_by": ranking_metric,
        "metric_min_score_of_a_rule": metric_min_score,
        "min_support": min_support
    })

    # No rules found
    if no_of_rules_generated == 0:
        current_app.logger.info("No Rules Found!")
        dict_super["Rules"] = {}
    # Found rules
    else:
        dict_rules = OrderedDict()
        for index, row in rules.iterrows():
            antecedent = list(row['antecedents'])
            consequent = list(row['consequents'])
            rule_details = OrderedDict({
                "full_rule": str(antecedent) + ' : ' + str(row['antecedent support']) + ' => ' + str(
                consequent) + ' : ' + str(row['consequent support']),
                "antecedents": antecedent,
                "consequents": consequent,
                "antecedent_sup": float(row['antecedent support']),
                "consequent_sup": float(row['consequent support']),
                "confidence": float(row['confidence']),
                "lift": float(row['lift']),
                "leverage": float(row['leverage']),
                "conviction": row['conviction']
            })
            dict_rules[f"Rule_Number_{str(index + 1).zfill(4)}"] = rule_details
        dict_super["Rules"] = dict_rules
    dict_super["Experiment"] = dict_experiment

    outfile_name = "EXP_" + str(experiment_number) + ".json"
    output_file = os.path.join(output_dir, outfile_name)
    with open(output_file, "w") as f:
        json.dump(dict_super, f, indent=4)

def _preprocess_transaction_data(input_file: str) -> (pd.DataFrame, int):
    df = pd.read_csv(input_file)
    column_names = df.columns
    for index, row in df.iterrows():
        for col in column_names:
            df.loc[index, col] = f"{col}_{row[col]}"
    transaction_list = df.values.astype(str).tolist()
    te = TransactionEncoder()
    transaction_bool_array = te.fit(transaction_list).transform(transaction_list)
    transaction_df = pd.DataFrame(transaction_bool_array, columns=te.columns_)
    return transaction_df, len(df)


def generate_rules(input_file: str, ranking_metric: str, metric_min_score: float, min_support: float, output_dir: str) -> pd.DataFrame:
    """
    Generates association rules based on the specified parameters and saves the results in a JSON file.

    Parameters
    ----------
    input_file : str
        Path to the input file containing transaction data in CSV format.
    ranking_metric : str
        Metric to evaluate the rules. Possible values are 'confidence', 'lift', 'leverage', and 'conviction'.
    metric_min_score : float
        Minimum threshold for the evaluation metric to determine if a rule is of interest.
    min_support : float
        Minimum support value to identify frequent itemsets.
    output_dir : str
        Directory where the generated rules will be saved as a JSON file.

    Returns
    ----------
    pandas.DataFrame
        DataFrame containing the generated association rules. If no rules are found, an empty DataFrame is returned.
    """
    df, total_nr_models = _preprocess_transaction_data(input_file)
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric=ranking_metric, min_threshold=metric_min_score)
        current_app.logger.debug(rules)
    else:
        rules = pd.DataFrame()
        current_app.logger.debug("FPGrowth algorithm did not yield frequently occuring itemsets. Please retry with different column names")
    _create_save_json_output(rules, ranking_metric, metric_min_score, min_support, total_nr_models, output_dir)
    return rules
