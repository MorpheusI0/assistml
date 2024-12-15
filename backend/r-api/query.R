
#' Generate list describing the use case task
#'
#' @param classif_type String to say if its binary or multiclass
#' @param classif_output String to say if a \code{single} prediction or \code{probabilities} are expected
#' @return List \code{usecase} with strings to describe the type of task and output in a standard string.
#' @export
#'
#' @examples
query_usecase<-function(classif_type,classif_output){
  verbose<-F

# TODO At the moment, only does input checks. Should parse AD4ML json to extract task properties.####
  usecase<-list("tasktype"="","output"="")
  if(classif_type %in% c("binary","binary classification","BINARY","BINARY CLASSIFICATION")){
    usecase$tasktype="Binary"
  }else if(classif_type %in% c("multiclass", "multiclass classification", "MULTICLASS", "MULTICLASS CLASSIFICATION", "MULTI-CLASS", "multi-class","categorical","CATEGORICAL","Categorical")){
    usecase$tasktype="Multi-Class"
  }

  if(classif_output %in% c("single","SINGLE","SINGLE PREDICTION","single prediction","single_prediction")){
    usecase$output="single"
  }else if(classif_output %in% c("probs","class probabilities","probabilities","PROBABILITIES","multiple")){
    usecase$output="probs"
  }
  # usecase$deployment<-deployment # "single_host" or "cluster"
  # usecase$implementation<-implementation # "single_language" or "multi_language"


  if(verbose){
    print(paste(timestamp(prefix = "",suffix = ""),": Inside query_usecase():"))
    print(usecase)
  }


  return(usecase)

}


#' @title Generates meta data of the new dataset
#' @description Generates a summary of all metadata and also gives details of each feature according to four semantic types: numeric, categorical, datetime or unstructured text.
#'
#' @param dataset the csv file containing the use case data sample
#' @param semantic_types char array with annotations of each feature semantic type (N, C, D, U, T)
#' @param use_case string identifying the use case (to be used with Mongo repository)
#'
#' @return List \code{data_features} describing all data features based on semantic types (numeric, categorical, datetime, unstructured text)
#' @export
#'
#' @examples
query_data<-function(dataset,semantic_types,dataset_name,use_case){

  verbose=F

  if(verbose){
    print("Entered query_data")
    print(" ")
  }

  # Fix to read csv files. Left here for unit tests
  # newdata <- if(is.character(dataset) && file.exists(dataset)){
    # read.csv(dataset)
  # } else {
    # as.data.frame(dataset)
  # }
  # dataset<-newdata
  # print(head(dataset))

  # demofeats<-c("N","N","N","N","N", "N", "N", "N", "N", "N", "N", "C", "C", "N","N", "N", "N", "N", "N", "N", "C", "N", "N", "N", "N", "N", "N","T")
  # if(ncol(dataset)!=length(semantic_types)){
  #    targetcol<-ncol(dataset)-length(semantic_types)
  # }else{
  #   targetcol<-0
  # }
  # featbase<-ncol(dataset)-targetcol



  ## Section R Mode query data:Computing data features directly in this function####

  # # Fix to read csv files. Left here for unit tests
  # newdata <- if(is.character(dataset) && file.exists(dataset)){
  #   read.csv(dataset)
  # } else {
  #   as.data.frame(dataset)
  # }
  # dataset<-newdata
  # # print(head(dataset))
  #
  # # demofeats<-c("N","N","N","N","N", "N", "N", "N", "N", "N", "N", "C", "C", "N","N", "N", "N", "N", "N", "N", "C", "N", "N", "N", "N", "N", "N","T")
  # # Getting total number of features minus the target
  # if(ncol(dataset)!=length(semantic_types)){
  #    targetcol<-ncol(dataset)-length(semantic_types)
  # }else{
  #   targetcol<-0
  # }
  # featbase<-ncol(dataset)-targetcol
  #
  #
  # # Simple way to try to fix malformed feature annotation vectors
  # for(i in 1:length(semantic_types)){
  #   # print(semantic_types)
  #     if(!(semantic_types[i] %in% c("N","C","D","U","T"))==TRUE){
  #
  #       if( verbose==T ){ print(paste0("ERROR in feature ",i,". Type Unknown. Now guessing...")) }
  #       switch (class(dataset[,i]),
  #         "factor" = semantic_types[i]<-"C",
  #         "integer" = semantic_types[i]<-"N"
  #       )
  #     }
  # }
  #
  #
  # # Calculating heuristics according to the semantic data type ####
  # ## Section :Computing metafeatures for numeric data####
  # if(sum(semantic_types=="N")>0){
  #   print("Analyzing numeric features")
  #   dataset.num<-dataset[,semantic_types=="N"]
  #   # print(head(dataset.num))
  #   numfeats<-list()
  #   for (j in 1:ncol(dataset.num)) {
  #     # eval(parse(text = paste0("numfeats$",names(dataset.num)[j])))
  #     numfeats[[j]]<-list(
  #         "missing_values"=sum(is.na(dataset.num[,j])), #Missing values
  #         # "outliers_qty"=sum(dataset.num[,j]>boxplot.stats(dataset.num[,j])$stats[5]), #outliers
  #         "min_orderm"=round(log10(abs(range( as.numeric(dataset.num[,j]) ))))[1], #Order of magnitude of the minimum value of the feature
  #         "max_orderm"=round(log10(abs(range( as.numeric(dataset.num[,j]) ))))[2], #Order of magnitude of the maximum value of the feature
  #         "correlation"=cor(dataset.num[,j],as.numeric(dataset[,semantic_types=="T"]=="Other_Faults")), # TODO: Make a binary preprocessing of the target variable properly. Avoid hardcoding "Other_Faults" logical comparison here####
  #         "outliers"=list(
  #           "number"=sum(dataset.num[,j]>boxplot.stats(dataset.num[,j])$stats[5]),
  #           "actual_values"=dataset.num[dataset.num[,j]>boxplot.stats(dataset.num[,j])$stats[5],j]
  #         )
  #     )
  #
  #   }
  #   names(numfeats)<-names(dataset.num)
  # }
  # ## Section :Computing metafeatures for categorical data####
  # if(sum(semantic_types=="C")>0){
  #   print("Analyzing categorical features")
  #   dataset.cat<-dataset[,semantic_types=="C"]
  #   # print(head(dataset.cat))
  #   catfeats<-list()
  #   for (j in 1:ncol(dataset.cat)) {
  #     # eval(parse(text = paste0("catfeats$",names(dataset.cat)[j])))
  #     catfeats[[j]]<-list(
  #         "missing_values"=sum(is.na(dataset.cat[,j])), #Missing values
  #         "nr_levels"=length(levels(as.factor(dataset.cat[,j]))),  # Number of levels in the feature
  #         "levels"= tapply(dataset.cat[,j], dataset.cat[,j], length),
  #         "imbalance"=max(tapply(dataset.cat[,j], dataset.cat[,j], length))/min(tapply(dataset.cat[,j], dataset.cat[,j], length)), # How many times is the most popular level more popular than the least popular level
  #         "correlation"=list(
  #           "pval"=chisq.test(as.factor(dataset.cat[,j]),as.factor(dataset[,semantic_types=="T"]=="Other_Faults"))$p.value, # TODO: Make a binary preprocessing of the target variable properly. Avoid hardcoding "Other_Faults" logical comparison here
  #           "chisq_correlated"=chisq.test(as.factor(dataset.cat[,j]),as.factor(dataset[,semantic_types=="T"]=="Other_Faults"))$p.value<=0.05
  #
  #         )
  #
  #     )
  #     names(catfeats[[j]]$levels)<-gsub(pattern = ".",replacement = ",",x = names(catfeats[[j]]$levels),fixed = T) #Removes dots from key values to insert in Mongo
  #   }
  #   names(catfeats)<-names(dataset.cat)
  # }
  #
  #
  #
  # ## Section :Computing metafeatures for datetime data####
  # ## TODO!: Heustistics for datetime data####
  # if(sum(semantic_types=="D")>0){
  #   dataset.dat<-dataset[,semantic_types=="D"]
  # }
  #
  # ## Section :Computing metafeatures for unstructured data####
  # ## TODO!:Heuristics for unstructured data####
  # if(sum(semantic_types=="U")>0){
  #   dataset.txt<-dataset[,semantic_types=="U"]
  # }


  # When data features are built in R
  # data_features<-list(
  #   "dataset_name"=dataset_name,
  #   "features"=featbase,
  #   "observations"=nrow(dataset),
  #   "numeric_ratio"=sum(semantic_types=="N")/featbase,
  #   "categorical_ratio"=sum(semantic_types=="C")/featbase,
  #   "datetime_ratio"=sum(semantic_types=="D")/featbase,
  #   "unstructured_ratio"=sum(semantic_types=="U")/featbase
  # )

  # steel.plates.fault[demofeats=="C"]

  # write(rjson::toJSON(data_features),file = "data_features.json")

## Section Mongo Mode query data:Obtaining data features from Mongo after python data_profiler computed them####
  mongodata<-mongolite::mongo("datasets","assistml",url = "mongodb://admin:admin@mongodb")


  current_data<-mongodata$find(query = eval(parse(text = paste0("'{\"Info.use_case\":\"",use_case,"\",\"Info.dataset_name\":\"",dataset_name,"\"}'"))),
                               fields = '{"_id":0}')

  data_features<-list(
    "dataset_name"=current_data$Info$dataset_name,
    "features"=current_data$Info$features,
    "observations"=current_data$Info$observations,
    "numeric_ratio"=current_data$Info$numeric_ratio,
    "categorical_ratio"=current_data$Info$categorical_ratio,
    "datetime_ratio"=current_data$Info$datetime_ratio,
    "unstructured_ratio"=current_data$Info$unstructured_ratio
  )


  if(length(current_data$Features$Numerical_Features)>0){
    data_features$numerical_features<-current_data$Features$Numerical_Features
  }

  if(length(current_data$Features$Categorical_Features)>0){
    data_features$categorical_features<-current_data$Features$Categorical_Features
  }

  if(length(current_data$Features$Datetime_Features)>0){
    data_features$datetime_features<-current_data$Features$Datetime_Features
  }

  if(length(current_data$Features$Unstructured_Features)>0){
    data_features$unstructured_features<-current_data$Features$Unstructured_Features
  }


  working_dir <- "/app/working"
  if (!dir.exists(working_dir)) {
    dir.create(working_dir)
  }
  file_path <- file.path(working_dir, "python_data_features.json")
  write(rjson::toJSON(data_features),file = file_path)


  return(data_features)

}



#' Generate preferred training settings for new dataset
#'
#' @param lang Preferred language
#' @param algofam Must be in 3-char format.
#' @param platform Must match one of the predefined platforms or option "other"
#' @param tuning_limit Threshold number of hyperparameters that are considered acceptable
#'
#' @return List of technical settings for training the ML model.
#' @export
#'
#' @examples
query_settings<-function(lang,algofam,platform,tuning_limit){

  algo<-c()
  if(!algofam%in%c("DLN","RFR","DTR","NBY","LGR","SVM","KNN","GBE","GLM")){
    print("Error: Algorithm unknown.")
  }
  pform<-c()
  if(platform %in% c("scikit","sklearn","scikit-learn"))
    pform<-"scikit"
  else if(platform %in% c("h2o","H2O","h2O","h2O_cluster_version","mojo"))
    pform<-"h2o"
  else if(platform %in% c("rweka","WEKA","R"))
    pform<-"rweka"
  return(list("language"=lang,"algorithm"=algofam,platform=pform,"hparams"=tuning_limit))
}


#' @title Generates query performance preferences to define acceptable performances
#' @description Generates list of performance preferences used to define what can be considered as acceptable and nearly acceptable models
#'
#' @param accuracy_range The width of acceptable accuracy for values going from 1 to 0.
#' @param precision_range The width of acceptable precision for values going from 1 to 0.
#' @param recall_range The width of acceptable recall for values going from 1 to 0.
#' @param trtime_range The width of acceptable training time for standardized values going from 1 to 0.
#'
#' @return List \code{preferences} with ranges accuracy, precision, recall and training time.
#' @export
#'
#' @examples
query_preferences<-function(accuracy_range, precision_range, recall_range, trtime_range){
preferences<-list(
  "acc_width"=accuracy_range,
  "pre_width"=precision_range,
  "rec_width"=recall_range,
  "tra_width"=trtime_range)
  return(preferences)
}
