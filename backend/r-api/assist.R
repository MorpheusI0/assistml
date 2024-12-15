# setwd("/home/ubuntu/assistmlr")
# Needed when not deployed as package and still launched from RStudio
source("cluster.R")
source("distance.R")
source("query.R")
source("rank.R")
source("results.R")
source("select.R")
source("reticulate.R")
source("explain.R")

# setwd("C:/Users/alexv/Documents/PhD/1prod/1kod/assistML/assistml/R")

#* Enables data upload of CSV files
#* @post /upload
upload_data<-function(req){

  # TEST WITH curl -v -F Content-Disposition=@steel-plates-fault.csv http://localhost:4321/upload


  files<-list(Rook::Multipart$parse(req))
  # print("Name of uploaded file")
  # print(names(files))
  print("Structure of uploaded file")
  print(str(files))

  file_info <- files[[1]]
  file_name <- names(file_info)[1]
  temp_file_path <- file_info[[file_name]]$tempfile
  print(paste("Temporary file path:", temp_file_path))


  # print(paste("Writing",files[[1]]$upload$filename[[1]]))
  newcsv<-read.csv(temp_file_path,header = T,as.is = T,dec = ".")
  print("Sample of uploaded data")
  print(head(newcsv))

  upload_dir <- "/uploads"
  if (!dir.exists(upload_dir)) {
    dir.create(upload_dir)
  }

  write.csv(x=newcsv,
        file = file.path(upload_dir, file_info[[file_name]]$filename),
        row.names = F)

  print(paste("Just uploaded",file_info[[file_name]]$filename))
  # close.connection(data_upload)
}




#* @param classif_type String to say if it is binary or multiclass
#* @param classif_output String to say if the result should be a single prediction or class probabilities
#* @param deployment String to say if the MLS should be deployed in a single host or cluster
#* @param implementation String to say how the implementation should be: single language or multi language
#* @param dataset_name CSV file of the new use case with sample data to analyze
#* @param usecase Name of the new use case for which the query is issued
#* @param sem_types JSON(?) Array with annotations for the semantic types of the data features of dataset
#* @param lang String to state the preferred programming language
#* @param algofam String with 3-char code to specify preferred algorithm family
#* @param platform String to state preferred execution platform
#* @param tuning_limit Int to state an acceptable number of hyper parameters to tune.
#* @param accuracy_range Float to state the range of top models in the accuracy dimension to be considered acceptable (0 to 1)
#* @param precision_range Float to state the range of top models in the precision dimension to be considered acceptable (0 to 1)
#* @param recall_range Float to state the range of top models in the recall dimension to be considered acceptable (0 to 1)
#* @param trtime_range Float to state the range of top models in the training time dimension to be considered acceptable (0 to 1)
#* @post /assistml
#' @param classif_type String to say if it is binary or multiclass
#' @param classif_output String to say if the result should be a single prediction or class probabilities
#' @param dataset CSV file of the new use case with sample data to analyze
#' @param sem_types JSON(?) Array with annotations for the semantic types of the data features of dataset
#' @param accuracy_range Float to state the range of top models in the accuracy dimension to be considered acceptable (0 to 1)
#' @param precision_range Float to state the range of top models in the precision dimension to be considered acceptable (0 to 1)
#' @param recall_range Float to state the range of top models in the recall dimension to be considered acceptable (0 to 1)
#' @param trtime_range Float to state the range of top models in the training time dimension to be considered acceptable (0 to 1)
#'
#' @title AssistML analysis for new usecase/data
#' @description Recommends ML models for a given query based on a base of known trained model. Stitches up all functions from the package.
#'
#' @return Ranked list \code{placeholder} of relevant models as well as insights to consider when using them.
#' @export
#'
#' @examples
assistml<-function(classif_type, # For query usecase
                   classif_output,
                   sem_types,
                   accuracy_range, # For query preferences
                   precision_range,
                   recall_range,
                   trtime_range,
                   dataset_name,
                   usecase, # For query data
                   ...,
                   deployment, # For query data. Optional at the moment
                   lang, # For query settings. All optional.
                   algofam,
                   platform,
                   tuning_limit,
                   implementation # For query usecase. Optional
                   ){

  # assistml_log<-file("assistml_log.log") # File name of output log
  # sink(assistml_log, append = TRUE, type = "output") # Writing console output to log file
  # sink(assistml_log, append = TRUE, type = "message")

verbose=T
start.time <- Sys.time()
  print("Connecting to mongo to get base models")
  base_models<-mongolite::mongo(collection = "base_models",
                                db = "assistml",
                                url = "mongodb://admin:admin@mongodb")

  # Obtaining default values to complete the query
  defaults<-base_models$find('{}',
                             fields='{"Model.Training_Characteristics.Dependencies.Platforms":1,
                             "Model.Training_Characteristics.Dependencies.Libraries":1,
                 "Model.Training_Characteristics.Hyper_Parameters.nr_hyperparams":1,
                             "Model.Training_Characteristics.implementation":1}')


  print("Connecting to mongo to get enriched models")
  enriched_models<-mongolite::mongo(collection = "enriched_models",
                                    db="assistml",
                                    url="mongodb://admin:admin@mongodb")


  # more_defaults<-read.csv("quantile_binary.csv",header = T)
  more_defaults<-enriched_models$find(query = '{}') #Picking data from Mongo

  print(" ")
  # print(dataset)
  print(" ")

  # Making sure that bundled_data always has a value, even if no dataset is provided or the name doesnt match the file name
  # if(missing(dataset)){
  #   print("value of dataset is missing")
  #   # print(dataset)
  #   newdata<-read.csv("dataset.csv",header = T,dec = ".")
  #   bundled_data<-"dataset.csv"
  #   print(paste("Reading csv uploaded with upload endpoint"))
  # }else{
  #   if(is.character(dataset) && file.exists(dataset)){
  #     newdata<-read.csv(dataset,header = T)
  #     bundled_data<-dataset
  #     print(paste("Read from local filesystem the specified file",dataset))
  #   }else{
  #     print("We dont know which dataset to use guy, the info provided is invalid.")
  #   }
  # }



  # Starting connection to the queries database to store this one.
  querydb<-mongolite::mongo(db="assistml",collection = "queries",url="mongodb://admin:admin@mongodb")
  queryId<-querydb$count()+1

  # Inserting query record before doing analysis
  print("Forming query record with fields...")
  print(paste("Query NR",queryId,"Issued at",as.character(Sys.time()),"For classification",classif_type))
  # print(classif_output)
  # print("")
  query_record<-list(
    "number"=queryId,
    "madeat"=paste0(gsub("-","",unlist(strsplit(as.character(Sys.time())," "))[1]),"-",substr(gsub(":","",unlist(strsplit(as.character(Sys.time())," "))[2]),1,4) ), #Gets timestamp in format 20211231-1457
    "classif_type"=classif_type,
    "classif_output"=classif_output,
    "dataset"=dataset_name, # For query data
    "semantic_types"=sem_types,
    "accuracy_range"=accuracy_range, # For query preferences
    "precision_range"=precision_range,
    "recall_range"=recall_range,
    "traintime_range"=trtime_range
    # "pref_language"=lang, # For query settings. All optional.
    # "pref_algofam"=algofam,
    # "pref_platform"=platform,
    # "tuning_limit"=tuning_limit,
    # "pref_implementation"=implementation # For query usecase. Optional
  )

  # Data structures to compute the overal distrust score
  warnings<-list()
  distrust_pts<-0

  ## Call query.R functions####

  # if(missing(lang)){ # Assigns the most popular language found in DB
  #   lang<-names(tapply(more_defaults$language,more_defaults$language,length)[tapply(more_defaults$language,more_defaults$language,length)==max(tapply(more_defaults$language,more_defaults$language,length))])[1]
  #   distrust_pts<-distrust_pts+1
  #   warnings[[length(warnings)+1]]<-paste0("Prog Language preferences not specified. Using ",lang," as default. DistrustPts increased by 1")
  #
  # }else{
  #   query_record$pref_language=lang
  # }
  #
  # if(missing(algofam)){
  #   algofam<-names(tapply(as.factor(substr(more_defaults$model_name,1,3)), as.factor(substr(more_defaults$model_name,1,3)), length)[tapply(as.factor(substr(more_defaults$model_name,1,3)), as.factor(substr(more_defaults$model_name,1,3)), length)==max(tapply(as.factor(substr(more_defaults$model_name,1,3)), as.factor(substr(more_defaults$model_name,1,3)), length))])[1]
  #   distrust_pts<-distrust_pts+1
  #   warnings[[length(warnings)+1]]<-paste0("Algorithm preferences not specified. Using ",algofam," as default. DistrustPts increased by 1")
  #
  # }else{
  #   query_record$pref_algofam=algofam
  # }
  # if(missing(platform)){
  #   platform<-"scikit-learn"
  #   distrust_pts<-distrust_pts+1
  #   warnings[[length(warnings)+1]]<-paste0("Platform preferences not specified. Using ",platform," as default. DistrustPts increased by 1")
  #
  # }else{
  #   query_record$pref_platform=platform
  # }
  # if(missing(tuning_limit)){
  #   tuning_limit=ceiling(quantile(defaults$Model$Training_Characteristics$Hyper_Parameters$nr_hyperparams)[2])
  #   distrust_pts<-distrust_pts+1
  #   warnings[[length(warnings)+1]]<-paste0("Max Number of Hyperparameters willing to tune not specified. Setting value to ",tuning_limit,". DistrustPts increased by 1")
  #
  # }else{
  #   query_record$tuning_limit=tuning_limit
  # }
  # if(missing(implementation)){
  #   implementation<-names(tapply(defaults$Model$Training_Characteristics$implementation,defaults$Model$Training_Characteristics$implementation,length)[tapply(defaults$Model$Training_Characteristics$implementation,defaults$Model$Training_Characteristics$implementation,length)==max(tapply(defaults$Model$Training_Characteristics$implementation,defaults$Model$Training_Characteristics$implementation,length))])[1]
  #   distrust_pts<-distrust_pts+1
  #   warnings[[length(warnings)+1]]<-paste0("Preferred way of implementing the MLS not specified. Choosing ",implementation," as default. DistrustPts increased by 1")
  #
  # }else{
  #   query_record$pref_implementation=implementation
  # }




  querydb$insert(data =
                   rjson::toJSON(query_record))

  print("#### QUERY FUNCTIONS ####")

  if(F){ # Change for verbose
    print("API call values")
    print(classif_type)
    print(classif_output)
    print(" ")

  }


  usecase_info<-query_usecase(classif_type,classif_output)

  if(F){ # Change to verbose
    print(paste(timestamp(prefix = "",suffix = ""),": Created use case info"))
    print(usecase_info)
    print(paste(" "))
  }




  data_feats<-query_data(dataset = newdata,semantic_types = sem_types,dataset_name=dataset_name,use_case=usecase)

  if(verbose){
    writeLines(paste("Retrieved descriptive data features for new dataset : \ncols",data_feats$features,"and \nrows",data_feats$observations))
  }


  # usecase_settings<-query_settings(lang, algofam, platform, tuning_limit)
  # print("Created settings list:")
  # print(usecase_settings)
  print(" ")

  usecase_preferences<-query_preferences(accuracy_range, precision_range, recall_range, trtime_range)
  if(F){ # Change to verbose
    print("Created performance preferences list:")
    print(usecase_preferences)
    print(" ")
  }

  # Dummies to build test query
  # demo_usecase<-fromJSON('{
  # "tasktype": ["binary"],
  # "output": ["single prediction"],
  # "deployment": ["single_host"],
  # "implementation": ["single_language"]
  # }')
  # demo_settings<-fromJSON('{
  # "language": ["python"],
  # "algorithm": ["DTR"],
  # "platform": ["scikit"],
  # "hparams": [5]
  # }')
  # demo_preferences<-fromJSON('{"acc_width": [0.2],
  # "pre_width": [0.1],
  # "rec_width": [0.12],
  # "tra_width": [0.3]
  #  }')




  ## Call select.R functions   ####
  print(" ")
  print("#### SELECT FUNCTIONS ####")

  usecase_models<-choose_models(task_type = usecase_info$tasktype,
                output_type =  usecase_info$output,
                data_features = data_feats)

  if(is.null(usecase_models[[2]]) ){
    print("No similar models could be found")
    return("No similar models could be found")
  }else{
    distrust_pts<-distrust_pts+ (3-usecase_models[[2]])

    warnings[[length(warnings)+1]]<-switch (usecase_models[[2]]+1,
                                            "Dataset similarity level 0. Only the type of task and output match. Distrust Pts increased by 3",
                                            "Dataset similarity level 1. Datasets used shared data types. Distrust Pts increased by 2",
                                            "Dataset similarity level 2. Datasets used have similar ratios of data types. Distrust Pts increased by 1",
                                            "Dataset similarity level 3. Datasets used have features with similar meta feature values. Distrust Pts increased by 0"
    )
  }



  print(paste("assist(): Selected models:",nrow(usecase_models[[1]]),"found with similarity level",usecase_models[[2]]," ."))


  ## Call cluster.R function ####
  print(" ")
  print("#### CLUSTER FUNCTIONS ####")

  usecase_mgroups<-cluster_models(selected_models = usecase_models[[1]],preferences = usecase_preferences)
  print(paste("Acc Models:",length(usecase_mgroups$acceptable_models) ))

  # Checking if there are any NACC models
  if(usecase_mgroups$nearly_acceptable_models[1] %in% c("none")){
    print( paste("Nearly Acc Models:",usecase_mgroups$nearly_acceptable_models[1] ))
  }else{
    print( paste("Nearly Acc Models:",length(usecase_mgroups$nearly_acceptable_models) ))
  }


  # Adding warnings based on the cutoff of the clusters in each ACC and NACC region
  if(usecase_mgroups$distrust$acceptable_models>0){
    distrust_pts<-distrust_pts+usecase_mgroups$distrust$acceptable_models
    print(paste("Added",usecase_mgroups$distrust$acceptable_models," distrust points from ACC cut"))
    warnings[[length(warnings)+1]]<-paste0("The selection of ACC solutions was not as clean as possible. Distrust Pts increased by ",usecase_mgroups$distrust$acceptable_models)
  }

  if(usecase_mgroups$distrust$nearly_acceptable_models>0){
    distrust_pts<-distrust_pts+usecase_mgroups$distrust$nearly_acceptable_models
    print(paste("Added",usecase_mgroups$distrust$nearly_acceptable_models," distrust points from NACC cut"))
    if(verbose==T){print(paste0("The selection of NACC solutions was not as clean as possible. Distrust Pts+",usecase_mgroups$distrust$nearly_acceptable_models))}
    warnings[[length(warnings)+1]]<-paste0("The selection of NACC solutions was not as clean as possible. Distrust Pts increased by ",usecase_mgroups$distrust$nearly_acceptable_models)
  }


  # print(paste("ACC Models sample:",usecase_mgroups$acceptable_models[1:3]))

  ## Call distance.R functions ####
  # print(" ")
  #
  # print("#### DISTANCE FUNCTIONS ####")
  #
  # mgroups_for_distance<-retrieve_settings(selected_models = usecase_mgroups)
  # print(" ")
  # print(paste("Usable cols to measure distances for Acc Models:",ncol(mgroups_for_distance$accmodels)))
  # # writeLines(names(mgroups_for_distance$accmodels))
  # print(" ")
  # print(paste("Usable cols to measure distances for Nearly Acc Models:",ncol(mgroups_for_distance$naccmodels)))
  # # writeLines(names(mgroups_for_distance$naccmodels))
  # print(" ")
  #
  #
  # print("Building query object for distance calculations...")
  #
  # ## Forming query dataframe for Hamming by parsing actual usecase info to form query_df for call compute_distances()
  #
  # bins_rows<-c("less_than_50k_rows","from_50k_to_100k_rows","from_100k_to_1M_rows","more_than_1M_rows")
  #
  #
  #
  # nrows_nominal<-""
  # if(data_feats$observations<50000){
  #   nrows_nominal<-bins_rows[1]
  # }else if (data_feats$observations>=50000 & nrow(data_feats$observations)<100000){
  #   nrows_nominal<-bins_rows[2]
  # }else if(data_feats$observations>=100000 & nrow(data_feats$observations)<1000000){
  #   nrows_nominal<-bins_rows[3]
  # }else if(data_feats$observations>=1000000){
  #   nrows_nominal<-bins_rows[4]
  # }
  #
  #
  # # Getting first data type. Change 1 to 2 to get second data type
  # # names(sort(unlist(demo_data_features[2:5]),decreasing = T)[1])
  #
  # first_ratio<-""
  # switch (names(sort(unlist(data_feats[3:6]),decreasing = T)[1]),
  #         "numeric_ratio" = first_ratio<-"numeric_ratio",
  #         "categorical_ratio"=first_ratio<-"categorical_ratio",
  #         "datetime_ratio"=first_ratio<-"date",
  #         "unstructured_ratio"=first_ratio<-"text")
  #
  # second_ratio<-""
  # switch (names(sort(unlist(data_feats[3:6]),decreasing = T)[2]),
  #         "numeric_ratio" = second_ratio<-"numeric_ratio",
  #         "categorical_ratio"=second_ratio<-"categorical_ratio",
  #         "datetime_ratio"=second_ratio<-"date",
  #         "unstructured_ratio"=second_ratio<-"text")
  #
  # bins_params<-c("less than 5 params", "from 5 to 10 params", "from 10 to 15 params", "more than 15 params" )
  #
  # usecase_hparams_nominal=""
  # if(usecase_settings$hparams<5){
  #   usecase_hparams_nominal<-bins_params[1]
  # }else if(usecase_settings$hparams>=5 & usecase_settings$hparams<10){
  #   usecase_hparams_nominal<-bins_params[2]
  # }else if(usecase_settings$hparams>=10 & usecase_settings$hparams<15){
  #   usecase_hparams_nominal<-bins_params[3]
  # }else if(usecase_settings$hparams>=15){
  #   usecase_hparams_nominal<-bins_params[4]
  # }
  #
  #
  #
  # query_df<-data.frame(
  #   "model_name"="query",
  #   "fam_name"=usecase_settings$algorithm,
  #   "rows"=nrows_nominal, # Computed based on input
  #   "first_datatype"=first_ratio, # Computed based on input. Needs to be parsed properly.
  #   "second_datatype"=second_ratio, # Computed based on input
  #   "nr_hyperparams_label"=usecase_hparams_nominal, # Computed based on input. May make sense to ask for the nominal value directly
  #   "deployment"=usecase_info$deployment, # comes from query_usecase
  #   "language"= usecase_settings$language, # Comes from query_settings
  #   "implementation"=usecase_info$implementation # comes from query_usecase
  # )
  #
  # print(" ")
  # print(paste("Fields of query df created:"))
  # writeLines(names(query_df))
  # print("")
  #
  # usecase_distances<-compute_distances(accmodels_performance= mgroups_for_distance$accmodels,
  #                   naccmodels_performance= mgroups_for_distance$naccmodels,
  #                   query_df )
  # print(" ")
  # print(paste("Finished computing the following kinds of distances:",paste(names(usecase_distances),collapse = " AND ")))




  ## Call rules.R functions####
  print(" ")

  print("#### RULES FUNCTIONS #### \n\n\n")

  # weka_transdata<-build_transdata(selected_models = usecase_mgroups)

  # usecase_rules<-find_rules(weka_transdata)

  if(usecase_mgroups$nearly_acceptable_models[1] %in% c("none")){
    usecase_rules<-python_rules(usecase_mgroups$acceptable_models)
  }else{
    usecase_rules<-python_rules(c(usecase_mgroups$acceptable_models,usecase_mgroups$nearly_acceptable_models))
  }


  print(paste("Finished rules generation.",length(usecase_rules),"rules generated."))

  # write.csv(usecase_distances$query_accmodels,file = "distances_query_accms.csv")
  # write.csv(usecase_distances$query_naccmodels,file = "distances_query_naccms.csv")
  # write.csv(usecase_distances$between,file = "border_distances.csv") # TODO!: Needs to be used to decide whether to present NACC models
  # write.csv(usecase_distances$accmodels,file="distances_inaccms.csv")
  # write.csv(usecase_distances$naccmodels,file="distances_innacms.csv")
  # write.csv(model_data[as.chara<cter(model_data$model.name)  %in% selected_models$acceptable_models |
  #              as.character(model_data$model.name)  %in%   selected_models$nearly_acceptable_models,],file = "rankeable_models.csv")


  ## Call rank.R functions####
  print(" ")

  print("#### RANK FUNCTIONS ####")

  ranked_models<-rank_models(usecase_mgroups,usecase_preferences)

  models_choice<-shortlist_models(ranked_models)


  ## Call results.R functions####
  print(" ")

  print("#### RESULTS FUNCTIONS ####")


  distrust_basis<-3+3+3 # 3 points for dataset similarity, 3 for ACC cut and 3 for NACC cut. ALWAYS
  print("Distrust score calculation")
  print(paste(distrust_pts,"over",distrust_basis))


  models_report<-generate_results(models_choice,usecase_rules,warnings,distrust_pts,query_record,distrust_basis)


  working_dir <- "/app/working"
  if (!dir.exists(working_dir)) {
    dir.create(working_dir)
  }
  models_report_path <- file.path(working_dir, "models_report.json")
  write(rjson::toJSON(models_report,indent = 3),models_report_path)

  querydb$update(
    query=eval(parse(text = paste0("'{\"number\":",queryId,"}'"))),
                 update= eval(parse(text = paste0(
                   "'{\"$set\":{\"report\":",rjson::toJSON(models_report),"}}'" ))),
                 upsert=F,
                 multiple=F)

  print("ASSISTML TERMINATED")
  # close.connection(assistml_log)

  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(paste("Time taken for end to end execution",time.taken))

  return(rjson::toJSON(models_report))


}

#* Echo the parameter that was sent in
#* @param msg The message to echo back.
#* @get /test
function(msg){
  if(missing(msg)) msg="pal"
  # assist_test<-file("assist_test.log")
  # sink(assist_test, append = TRUE, type = "output") # Writing console output to log file
  # sink(assist_test, append = TRUE, type = "message")


  print(paste0(timestamp(),": AssistML working, ",msg))
  print("")
  print(sessionInfo())
  print("")
  print(osVersion)

  # close.connection(assist_test)

  return(paste0(timestamp(),": AssistML working, ",msg))
}
