"""
@author: Dinesh Subhuraaj
"""

import glob
import json
import os
import sys
from collections import OrderedDict

import pandas as pd
from flask import current_app
# Imports for mlxtend module
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def main():

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # creates json file using the rules generated from the FPGrowth algorithm and store in output/FPGROWTH folder
    def create_save_json_output(rules,ranking_metric,metric_min_score,min_support,total_nr_models):
        flag =0
        rule_found=1
        max_count = 0
        fpgrowth_dir = os.path.join(working_dir, "FPGROWTH")
        if not os.path.exists(fpgrowth_dir):
            os.makedirs(fpgrowth_dir)
        for filename in glob.glob(os.path.join(fpgrowth_dir, '*.json')):
            val=""
            name=""
            indb= filename.rfind("\\")
            indb=indb+1
            name=filename[indb:]
            indd=name.rfind(".")
            name=name[:indd]
            for ch in name:
                if(ch.isdigit()):
                    val=val+ch
            yi=int(val)
            if(yi>max_count):
                max_count=yi
                flag=1
        no_of_rules_generated = len(rules.index)
        
        
        #two sub dicts are stored in the super dict
        current_app.logger.info("Found " + str(no_of_rules_generated) + " rules !!")
        dict_super=OrderedDict()
        dict_sub1=OrderedDict()
        dict_sub2=OrderedDict()
        
        dict_sub1["experiment_nr"]=max_count+1
        dict_sub1["total_models"]= total_nr_models
        dict_sub1["rules_found"]= no_of_rules_generated
        dict_sub1["ranking_by"]= ranking_metric
        dict_sub1["metric_min_score_of_a_rule"]= metric_min_score     
        dict_sub1["min_support"]= min_support
        # No rules found
        if(no_of_rules_generated==0):
            current_app.logger.info("No Rules Found!")
            dict_super["Rules"] = {}
            max_count=max_count+1
            outfile_name = "EXP_"+str(max_count)+".json"
            outfile_path = os.path.join(working_dir, outfile_name)
            with open(outfile_path, "a") as myfile:
                myfile.write("{}")
            myfile.close()
            rule_found=0
        # Found rules
        else:
            rule_count=1
            for index, row in rules.iterrows():
                dict_create=OrderedDict()
                antecedent = list(row['antecedents'])
                consequent = list(row['consequents'])
                dict_create["full_rule"] = str(antecedent) + ' : ' + str(row['antecedent support']) + ' => ' + str(consequent) + ' : ' + str(row['consequent support']) 
                dict_create["antecedents"] = list(row['antecedents'])
                dict_create["consequents"] = list(row['consequents'])
                dict_create["antecedent_sup"]=float(row['antecedent support'])
                dict_create["consequent_sup"]=float(row['consequent support'])
                dict_create["confidence"]=float(row['confidence'])
                dict_create["lift"]=float(row['lift'])
                dict_create["leverage"]=float(row['leverage'])
                dict_create["conviction"]=row['conviction']
                
                if((len(str(rule_count)))==1):
                    k="Rule_Number"+"_"+str(000)+str(rule_count)
                elif((len(str(rule_count)))==2):
                    k="Rule_Number"+"_"+str(00)+str(rule_count)
                elif((len(str(rule_count)))==3):
                    k="Rule_Number"+"_"+str(0)+str(rule_count)
                else:
                    k="Rule_Number"+"_"+str(rule_count)
                dict_sub2[k]= dict_create
                rule_count=rule_count+1
            dict_super["Rules"]=dict_sub2

        dict_super["Experiment"]=dict_sub1
        json_format=json.dumps(dict_super, indent=4)
        max_count=max_count+1
        outfile_name = "EXP_"+str(max_count)+".json"
        outfile_path = os.path.join(fpgrowth_dir, outfile_name)
        with open(outfile_path, "a") as myfile:
            myfile.write(json_format)
        myfile.close()
          
     ####### End of function 'create_save_json_output' #######   
    
    # Database details
    '''current_app.logger.info("Connected to database")
    myclient = pymongo.MongoClient("mongodb://admin:admin@mongodb/")
    dbname = myclient["assistml"]
    collection_enriched = dbname["enriched_models"]
    total_no_models = collection_enriched.count()
    '''

    #Select DataSet - Uncomment if needed in future
    '''
    current_app.logger.info("You want to apply association on individual or merged data?")
    x=input("Enter number: 1. Merged, 2. Individual")
    if (x==2):
        current_app.logger.info("Presently kick, bank and Amazon food review dataset or merged data is available")
        dataset = input("Enter Dataset Name ")
        Input_File="quantile_" + str(dataset) +"_selectedcols_binarized.arff"
    elif(x==1):
        Input_File="merged_datasets_kickbankafr_selectedcols_binarized_exp36.arff"
    '''

    # Cmd line arguments
    # Ranking metric :: 0=confidence | 1=lift | 2=leverage | 3=Conviction
    ranking_metric=str(sys.argv[1])
    metric_min_score=str(sys.argv[2])
    min_support=str(sys.argv[3])
    
    ranker=int(ranking_metric)
    rankedby = ""
    if(ranker==0):
        rankedby="confidence"
    if(ranker==1):
        rankedby="lift"
    if(ranker==2):
        rankedby="leverage"
    if(ranker==3):
        rankedby="conviction"

    Input_File = os.path.join(working_dir, "merged_data_selectedcols.csv")
    df = pd.read_csv(Input_File)
    total_no_models = df.shape[0]
    column_names = df.columns
    # Append column names to every element in df
    for index, row in df.iterrows():
        for col in column_names:
            df.loc[index, col] = col + '_' + str(df.loc[index, col])

    # Convert dataframe to list
    transaction_list = df.values.astype(str).tolist()


    # Pre-process transaction list into dataframe of boolean values 
    te = TransactionEncoder()
    transaction_bool_array = te.fit(transaction_list).transform(transaction_list)
    df = pd.DataFrame(transaction_bool_array, columns = te.columns_)

    # Find frequently occuring items using FPGrowth
    frequent_itemsets_fp = fpgrowth(df, min_support=float(min_support), use_colnames=True)

    # Find association rules
    if not (frequent_itemsets_fp.empty == True):
        rules_fp = association_rules(frequent_itemsets_fp, metric=rankedby, min_threshold=float(metric_min_score))
        current_app.logger.info(rules_fp)
        create_save_json_output(rules_fp,rankedby,metric_min_score,min_support,total_no_models)
    else:
        rules_fp = pd.DataFrame()
        create_save_json_output(rules_fp,rankedby,metric_min_score,min_support,total_no_models)
        current_app.logger.info("FPGrowth algorithm did not yield frequently occuring itemsets. Please retry with different column names")
    
    # Association rules using pyfpgrowth module
    '''
    import pyfpgrowth
    patterns = pyfpgrowth.find_frequent_patterns(df, 50)
    rules = pyfpgrowth.generate_association_rules(patterns,0.7)
    '''
   

if __name__ == "__main__": 
    main()

