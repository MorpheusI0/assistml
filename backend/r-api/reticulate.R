
python_rules<-function(model_codes){
  verbose<-T

  if(verbose){print("Entered python_rules() with the following chosen models")}
  if(missing(model_codes)) model_codes=""

  if(verbose){print(model_codes)
  print("")}

  # install.packages("reticulate")

  # reticulate::virtualenv_list()
  # reticulate::virtualenv_create("assistml")
  # reticulate::py_install("pymongo",envname = "assistml")
  # reticulate::py_install("mlxtend",envname = "assistml")
  # reticulate::use_virtualenv("assistml")
  # reticulate::use_virtualenv("guacamole")
  # reticulate::py_install("pymongo",envname = "guacamole")
  # reticulate::py_install("mlxtend",envname = "guacamole")

  # reticulate::py_install("pandas")
  # reticulate::py_install("pymongo")



  # Sys.which("python")

  # py_config()

  print("Calling python data_encoder.py...")
  # Triggers execution of python modules with mlxtend to generate rules

  data_enc<-"import os\n"

  data_enc<-paste0(data_enc,"os.system(\'python3 /app/python-modules/data_encoder.py \"[fam_name,nr_hyperparams_label,performance_gap,quantile_accuracy,quantile_recall,quantile_precision,platform,quantile_training_time,nr_dependencies]\" \"[",paste0(model_codes,collapse = ","),"]\"\')")

  # paste0(c("SVM_kick_003","DTR_kick_012","RFR_kick_022","DTR_kick_011","NBY_bank_002"),collapse = "\",\"")

  working_dir <- "/app/working"
  if (!dir.exists(working_dir)) {
    dir.create(working_dir)
  }

  data_rules_gen_py_path <- file.path(working_dir, "data_rules_gen.py")
  rules_py_path <- file.path(working_dir, "rules.py")
  push_rules_py_path <- file.path(working_dir, "push_rules.py")
  python_rules_json_path <- file.path(working_dir, "python_rules.json")

  write(data_enc,file = data_rules_gen_py_path,append = F)
  reticulate::py_run_file(data_rules_gen_py_path)


  # print("Generating rules ...")
  rules_py<-"import os\n"


  # Ranking metric :: 0=confidence | 1=lift | 2=leverage | 3=Conviction
  # ranking metric, metric min score, min support
  rules_py<-paste0(rules_py,"os.system(\'python3 /app/python-modules/association_python.py 0 0.7 0.25 \')")



  write(rules_py,file = rules_py_path,append = F)
  print("Calling python association_python.py")
  reticulate::py_run_file(rules_py_path)


  analysis_py<-"import os\n"


  analysis_py<-paste0(analysis_py,"os.system(\'python3 /app/python-modules/analysis.py 0.5 0.01 1.2 \')")


  write(analysis_py,file = push_rules_py_path,append = F)
  print("Calling python analysis.py")
  reticulate::py_run_file(push_rules_py_path)

  print("Retrieving last added rules summary from Mongo")

  rulestamp <- format(Sys.time(), "%Y%m%d-%H%M")

  if(verbose){
    print("Retrieving rules for experiment inserted at:")
    print(rulestamp)
  }
  



  rules<-mongolite::mongo("rules","assistml","mongodb://admin:admin@mongodb")
  current_setofrules<-rules$find(query = paste0('{ "Rules":{"$exists":true}, "Experiment.created":"',rulestamp,'"}'),
                              fields = '{"Rules":true}'
                              )$Rules


  if(length(current_setofrules)>0){
    print("Storing rules as json")
    # Saves the found rules as Json for backup
    write(rjson::toJSON(current_setofrules[1:length(current_setofrules)],indent = 3),file = python_rules_json_path,append = F)
    return(current_setofrules[1:length(current_setofrules)])
  }else{
    print("No rules were found nor filtered")
    return(current_setofrules)
  }




  }


