# -*- coding: utf-8 -*-
"""
Created on Sat Dec 8 18:52:59 2019

@author: Adrika
"""


import os
import glob
import json
from collections import OrderedDict
import sys
import pymongo
from datetime import datetime

from flask import current_app


def main():

    working_dir = os.path.expanduser(current_app.config['WORKING_DIR'])
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    file_list=[]
    file_dict_menu=OrderedDict()
    k=1
    fpgrowth_dir = os.path.join(working_dir, "FPGROWTH")
    max_count=0
    li=[]
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

    fn="EXP_"+str(max_count)+".json"
    filepath = os.path.join(fpgrowth_dir, fn)
    current_app.logger.info(filepath)
    #filepath="C:/Users/Javed/Desktop/INFOTECH/Infotech Semester 3/Study Project/git/mlm-perf/mlm-perf-modules/Output/FPGROWTH/EXP_26.json"

    '''
    file_dict_menu[ke]=new_name
    file_dict_menu=dict((sorted(file_dict_menu.items())))
    current_app.logger.info(file_dict_menu)
    
    
    
    x=input("Enter the name of output file from the menu for which you want to apply the analysis")
    filename=file_dict_menu[int(x)]
    '''
    # Open the latest file generated in the output/FPGROWTH folder.
    with open(filepath) as json_doc:
        raw_data = json.load(json_doc)
    json_doc.close()

    '''if raw_data == {}:
        current_app.logger.info("Exiting as no rules were found")
        exit(1)
    '''
    '''
    Filter out the twin rules and store the raw_data in data
    
    '''

    twin_dict=OrderedDict()

    for iter1 in raw_data:
        if(iter1=="Rules"):
            for iter2 in raw_data[iter1]:
                full_rule=raw_data[iter1][iter2]["full_rule"]
                if "antecedents" in raw_data[iter1][iter2] and "consequents" in raw_data[iter1][iter2]:
                    antecedent1 = raw_data[iter1][iter2]['antecedents']
                    consequent1 = raw_data[iter1][iter2]['consequents']
                else:
                    index_start_a=full_rule.find("[")
                    index_end_a=full_rule.find("]")+5
                    antecedent1=full_rule[index_start_a:index_end_a]
                    antecedent1=antecedent1.strip()
                    index_start_c=full_rule.rfind("[")
                    index_end_c=full_rule.rfind("]")+5
                    consequent1=full_rule[index_start_c:index_end_c]
                    consequent1=consequent1.strip()
                for iter3 in raw_data[iter1]:
                    if(iter2!=iter3):
                        full_rule2=raw_data[iter1][iter3]["full_rule"]
                        if "antecedents" in raw_data[iter1][iter3] and "consequents" in raw_data[iter1][iter3]:
                            antecedent2 = raw_data[iter1][iter3]['antecedents']
                            consequent2 = raw_data[iter1][iter3]['consequents']
                        else:
                            index_start_a_t=full_rule2.find("[")
                            index_end_a_t=full_rule2.find("]")+5
                            antecedent2=full_rule2[index_start_a_t:index_end_a_t]
                            antecedent2=antecedent2.strip()
                            index_start_c_t=full_rule2.rfind("[")
                            index_end_c_t=full_rule2.rfind("]")+5
                            consequent2=full_rule2[index_start_c_t:index_end_c_t]
                            consequent2=consequent2.strip()
                        if(antecedent1==consequent2 and antecedent2==consequent1):
                           flag=0
                           for (k,v) in twin_dict.items():
                               if(v==iter2 and k==iter3):
                                   flag=1
                               else:
                                   continue
                           if(flag==0):
                               twin_dict[iter2]=iter3


    for k in twin_dict.keys():
        current_app.logger.info("Removing a twin rule!!")
        del raw_data["Rules"][k]


    '''
    Eliminate rules that have the same items, either as part of the consequent or the antecedent,
     also if the item sets are different (e.g. A=>B,C and C=>A,B).
    '''
    redundant_dict=OrderedDict()

    for iter1 in raw_data:
        if(iter1=="Rule"):
            for iter2 in raw_data[iter1]:
                attribute_list_rule=[]
                full_rule=raw_data[iter1][iter2]["full_rule"]
                if "antecedents" in raw_data[iter1][iter2] and "consequents" in raw_data[iter1][iter2]:
                    antecedent1 = raw_data[iter1][iter2]['antecedents']
                    consequent1 = raw_data[iter1][iter2]['consequents']
                else:
                    index_start_a=full_rule.find("[")+1
                    index_end_a=full_rule.find("]")
                    antecedent1=full_rule[index_start_a:index_end_a]
                    antecedent1.strip()
                    index_start_c=full_rule.rfind("[")+1
                    index_end_c=full_rule.rfind("]")
                    consequent1=full_rule[index_start_c:index_end_c]
                    consequent1.strip()
                antecedent1_list= antecedent1.split(",")
                consequent1_list= consequent1.split(",")
                for at in antecedent1_list:

                    at=at.encode('ascii','ignore')
                    at=at.strip()
                    attribute_list_rule.append(at)
                for co in consequent1_list:

                    co=co.encode('ascii','ignore')
                    co=co.strip()
                    attribute_list_rule.append(co)
                conf1=raw_data[iter1][iter2]["confidence"]
                lift1=raw_data[iter1][iter2]["lift"]
                lev1=raw_data[iter1][iter2]["leverage"]
                conv1=raw_data[iter1][iter2]["conviction"]
                a_c_1=raw_data[iter1][iter2]["antecedent_sup"]
                c_c_1=raw_data[iter1][iter2]["consequent_sup"]

                for iter3 in raw_data[iter1]:
                    if(iter2!=iter3):
                        attribute_list_rule_2=[]
                        full_rule2=raw_data[iter1][iter3]["full_rule"]
                        if "antecedents" in raw_data[iter1][iter3] and "consequents" in raw_data[iter1][iter3]:
                            antecedent2 = raw_data[iter1][iter3]['antecedents']
                            consequent2 = raw_data[iter1][iter3]['consequents']
                        else:
                            index_start_a_t=full_rule2.find("[")+1
                            index_end_a_t=full_rule2.find("]")
                            antecedent2=full_rule2[index_start_a_t:index_end_a_t]
                            antecedent2.strip()
                            index_start_c_t=full_rule2.rfind("[")+1
                            index_end_c_t=full_rule2.rfind("]")
                            consequent2=full_rule2[index_start_c_t:index_end_c_t]
                            consequent2.strip()
                        antecedent2_list= antecedent2.split(",")
                        consequent2_list= consequent2.split(",")
                        for at1 in antecedent2_list:

                            at1=at1.encode('ascii','ignore')
                            at1=at1.strip()
                            attribute_list_rule_2.append(at1)
                        for co1 in consequent2_list:

                            co1=co1.encode('ascii','ignore')
                            co1=co1.strip()
                            attribute_list_rule_2.append(co1)
                        conf2=raw_data[iter1][iter3]["confidence"]
                        lift2=raw_data[iter1][iter3]["lift"]
                        lev2=raw_data[iter1][iter3]["leverage"]
                        conv2=raw_data[iter1][iter3]["conviction"]
                        a_c_2=raw_data[iter1][iter3]["antecedent_sup"]
                        c_c_2=raw_data[iter1][iter3]["consequent_sup"]
                        if(set(attribute_list_rule)==set(attribute_list_rule_2) and conf1==conf2 and lift1==lift2 and lev1==lev2 and conv1==conv2 and a_c_1==a_c_2 and c_c_1==c_c_2):
                           flag=0
                           for (k,v) in redundant_dict.items():
                               if(v==iter2 and k==iter3):
                                   flag=1
                               else:
                                   continue
                           if(flag==0):
                               redundant_dict[iter2]=iter3


    for k in redundant_dict.keys():
        current_app.logger.info("Removing a redundant rules!!")
        del raw_data["Rules"][k]


    data=raw_data

    '''
    Enter filter attributes
    
    conf=input("Enter confidence threshold value below which the rule will be filtered out; hint: default 0.5, values between 1 and 0.5 " )
    lev=input("Enter the leverage threshold value below which the rule will be filtered out; hint: default 0.05, values greater than 0 ")
    lift=input("Enter the lift threshold value below which the rule will be filtered out; hint: default 1.2, values greater than 1 ")
    
    '''
    '''
    These system arguments should be passed by the executer, these parameters are used to filter out the rules
    Rules below the given threshold values will be filtered out and only the ones that satisfy the threshold will
    be selected and stored in json format.
    
    '''
    conf=float(sys.argv[1])
    lev=float(sys.argv[2])
    lif=float(sys.argv[3])

    new_data=OrderedDict()
    new_data["Selection_Criteria"]=OrderedDict()

    new_data["Selection_Criteria"]["confidence_threshold"]=conf
    new_data["Selection_Criteria"]["leverage_threshold"]=lev
    new_data["Selection_Criteria"]["lift_threshold"]=lif
    for item in data:
        #item=item.encode('ascii','ignore')
        if("Experiment" in item):
            new_data["Experiment"]=data["Experiment"]
            new_data["Experiment"]["created"]=datetime.now().strftime("%Y%m%d-%H%M")
        elif("Rules" in item):
            new_data["Rules"]=OrderedDict()
            for rules in data[item]:
                #rules=rules.encode('ascii','ignore')
                confidence=data[item][rules]['confidence']
                conviction=data[item][rules]['conviction']
                lift=data[item][rules]['lift']
                leverage=data[item][rules]['leverage']
                if(confidence<1 and confidence>=conf and lift>=lif and leverage>=lev and lift>=1 and leverage>0):
                    new_data[item][rules]=data[item][rules]
        else:
            current_app.logger.info(item)

    count=0
    try:
        for rules in  new_data["Rules"]:
            count +=1
    except KeyError:
        current_app.logger.info("No rules received from association.")
    new_data["Selection_Criteria"]["rules_selected"]=count
    new_data["Selection_Criteria"]["exclude_confidence"]=1
    write_ready=json.dumps(new_data,sort_keys=True)
    filedotindex=filename.find(".")

    filename=filename[:filedotindex]
    fpgrowth_filtered_dir = os.path.join(working_dir, "FPGROWTH_FILTERED")
    if not os.path.exists(fpgrowth_filtered_dir):
        os.makedirs(fpgrowth_filtered_dir)
    selected_filename = "EXP_"+str(max_count)+"_selected.json"
    filepath = os.path.join(fpgrowth_filtered_dir, selected_filename)

    with open(filepath, "a") as myfile:
        myfile.write(write_ready)
    myfile.close()

    '''
    database details
    '''
    mylist = []
    current_app.logger.info("Connected to database")
    myclient = pymongo.MongoClient(
            host=current_app.config['MONGO_HOST'],
            port=int(current_app.config['MONGO_PORT']),
            username=current_app.config['MONGO_USER'],
            password=current_app.config['MONGO_PASS']
        )
    dbname = myclient["assistml"]
    collection_rules = dbname["rules"]
    mylist.append(json.loads(write_ready))
    collection_rules.insert_many(mylist)


if __name__ == "__main__":
    main()
