#' @title Generate single model report
#' @description CReates list structure with human-readable details of a single model
#'
#' @param picked_model String with the model code
#' @param query_df Dataframe containing the query details relevant for Hamming
#'
#' @return List with model details in human-readable form
#' @export
#'
#' @examples
generate_model_report<-function(picked_model,details){
  ## Pull all MongoDB and model_data data useful about this model

  verbose<-F
  if(verbose){
    print(paste("Entered generate_model_report for",picked_model))
    print(details)
  }



  con<-mongolite::mongo("base_models",db="assistml",url="mongodb://admin:admin@mongodb")
  con$info()

  accmodel_json<-con$find(query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",picked_model,"\"]}}'"))) #,
                              # fields = '{"Model.Training Characteristics":true,"_id":false}'
                          )


  enriched_models<-mongolite::mongo(collection = "enriched_models",
                                    db="assistml",
                                    url="mongodb://admin:admin@mongodb")
  accmodel_data<-enriched_models$find(query =eval(parse(text = paste0("'{\"model_name\":{\"$in\":[\"",picked_model,"\"]}}'"))))


  # Getting the dataset annotation
  datasetsMongo<-mongolite::mongo("datasets",db = "assistml",url = "mongodb://admin:admin@mongodb")
  mongo_feats<-datasetsMongo$find(query = eval(parse(text = paste0("'{\"Info.dataset_name\":\"", accmodel_json$Model$Data_Meta_Data$dataset_name,"\" }'") )),
                                  fields = '{"Info":1,"_id":0}')





  # model_name<-""
  switch (as.character(accmodel_data$fam_name),
    "DTR" = model_name<-"Model trained with Decision Trees",
    "RFR" = model_name<-"Model trained with Random Forests",
    "LGR" = model_name<-"Model trained with Logistic Regression",
    "SVM" = model_name<-"Model trained with Support Vector Machines",
    "NBY" = model_name<-"Model trained with Naive Bayes",
    "DLN" = model_name<-"Model trained with Deep Learning",
    "GBE" = model_name<-"Model trained with Gradient Boosting Ensemble",
    "GLM" = model_name<-"Model trained with General Linear Model"
  )




  # print(list("name"=model_name,
  #             "language"=accmodel_json$Model$Training_Characteristics$language,
  #             "platform"=accmodel_json$Model$Training_Characteristics$algorithm_implementation,
  #             "nr_hparams"=accmodel_json$Model$Training_Characteristics$Hyper_Parameters$nr_hyperparams,
  #             "similarity_to_query"=details$query,
  #             "differences"="None",
  #             "overall_score"=signif(details$performance_score,4),
  #             "performance"=list(
  #               "accuracy"=substr(as.character(accmodel_data$quantile_accuracy),start = 5,stop = nchar(as.character(accmodel_data$quantile_accuracy))),
  #               "precision"=substr(as.character(accmodel_data$quantile_precision),start = 5,stop = nchar(as.character(accmodel_data$quantile_precision))),
  #               "recall"=substr(as.character(accmodel_data$quantile_recall),start = 5,stop = nchar(as.character(accmodel_data$quantile_recall))),
  #               "training_time"=substr(as.character(accmodel_data$quantile_training_time),start = 5,stop = nchar(as.character(accmodel_data$quantile_training_time)))
  #             ),
  #             "code"=picked_model,
  #             "rules"="None"
  # ))

  ## TODO!: Add output analysis    ####

  if(!is.null(accmodel_json$Model$Metrics$Explainability)){
    feats_totals<-c()
    for (i in 1:length(accmodel_json$Model$Metrics$Explainability)) {
      feats_totals<-c(feats_totals,accmodel_json$Model$Metrics$Explainability[[i]]$total)
    }
    names(feats_totals)<-names(accmodel_json$Model$Metrics$Explainability[1:length(accmodel_json$Model$Metrics$Explainability)])
  }else{
    accmodel_json<-explain_python(accmodel_json$Model$Info$name)
    if(verbose){
      print(paste("Testing existence of Explainability content:", !is.null(accmodel_json$Model$Metrics$Explainability) ))
    }
    feats_totals<-c()
    for (i in 1:length(accmodel_json$Model$Metrics$Explainability)) {
      feats_totals<-c(feats_totals,accmodel_json$Model$Metrics$Explainability[[i]]$total)
    }
    names(feats_totals)<-names(accmodel_json$Model$Metrics$Explainability[1:length(accmodel_json$Model$Metrics$Explainability)])
  }


  confmat_text<-paste0("Suitable to detect variations in data like ",paste0(names(feats_totals[feats_totals<0.05]),collapse = ", "),
                       ". Unsuitable to detect variations in data like ",paste0(names(feats_totals[feats_totals>0.4]),collapse = ", "),".")


  preprocessing_description<-""

  # adding the preprocessing info of numerical data if different to none
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$numerical_encoding %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"The numerical data is read (with) ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$numerical_encoding,
                                      ". ")
  }
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$numerical_selection %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Numerical data was filtered based on ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$numerical_selection,
                                      ". ")
  }

  # adding the preprocessing info of categorical data if different to none
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$categorical_encoding %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Categorical data is read (with) ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$categorical_encoding,
                                      ". ")
  }
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$categorical_selection %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Categorical data was filtered based on ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$categorical_selection,
                                      ". ")
  }

  # adding the preprocessing info of datetime data if different to none
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$datetime_encoding %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Date time data is read (with) ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$datetime_encoding,
                                      ". ")
  }
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$datetime_selection %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Date time data was filtered based on ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$datetime_selection,
                                      ". ")
  }

  # adding the preprocessing info of text data if different to none
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$text_encoding %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Unstructured text is read (with) ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$text_encoding,
                                      ". ")
  }
  if(!( accmodel_json$Model$Data_Meta_Data$Preprocessing$text_selection %in% c("None","none") ) ){
    preprocessing_description<-paste0(preprocessing_description,"Unstructured text was filtered based on ",
                                      accmodel_json$Model$Data_Meta_Data$Preprocessing$textselection,
                                      ". ")
  }




  print(paste("Generated report for",picked_model))


  return(list("name"=model_name,
              "language"=accmodel_json$Model$Training_Characteristics$language,
              "platform"=accmodel_json$Model$Training_Characteristics$algorithm_implementation,
              "nr_hparams"=accmodel_json$Model$Training_Characteristics$Hyper_Parameters$nr_hyperparams,
              "nr_dependencies"=accmodel_json$Model$Training_Characteristics$Dependencies$nr_dependencies,
              "implementation"=accmodel_json$Model$Training_Characteristics$implementation,
              "deployment"=accmodel_json$Model$Training_Characteristics$deployment,
              "cores"=accmodel_json$Model$Training_Characteristics$cores,
              "power"=accmodel_json$Model$Training_Characteristics$ghZ,
              "out_analysis"=confmat_text,
              "preprocessing"=preprocessing_description,
              # "similarity_to_query"=details$query,
              # "differences"="None",
              "overall_score"=signif(details$performance_score,4),
              "performance"=list(
                "accuracy"=substr(as.character(accmodel_data$quantile_accuracy),start = 5,stop = nchar(as.character(accmodel_data$quantile_accuracy))),
                "precision"=substr(as.character(accmodel_data$quantile_precision),start = 5,stop = nchar(as.character(accmodel_data$quantile_precision))),
                "recall"=substr(as.character(accmodel_data$quantile_recall),start = 5,stop = nchar(as.character(accmodel_data$quantile_recall))),
                "training_time"=substr(as.character(accmodel_data$quantile_training_time),start = 5,stop = nchar(as.character(accmodel_data$quantile_training_time)))
                ),
              "code"=picked_model,
              "rules"="None"
              ))

}

search_more<-function(){
  ## TODO!: Look for models with the closest intracluster Hamming distance

}

#' @title Add rules to model report
#' @description Selects relevant rules to include for the model based on its described report contents.
#'
#' @importFrom stringr str_match
#'
#' @param model_report List containing the report of a single model
#' @param found_rules Full list of generated rules of acceptable and nearly acceptable models.
#'
#' @return Array of strings containing the rules that mention the algorithm family used in the supplied model.
#' @export
#'
#' @examples
add_rules<-function(model_report,found_rules){
  # library(stringr)
  verbose<-F
  if(verbose){
    print("Entered add_rules()")
    print(head(found_rules))
  }


  ## Find association rules for the specified model components
  model_leads<-unlist(strsplit(model_report$code,"_"))


  relevant_rules<-c()



  for (i in 1:length(found_rules)) {
    if (is.null(found_rules[[i]]$full_rule) || is.null(model_leads[1]) ||
      found_rules[[i]]$full_rule == "" || model_leads[1] == "") {
        next
    }
    if(!is.na(
      stringr::str_match(found_rules[[i]]$full_rule,model_leads[1]) #Checks if the fam name matches that of the model being reported
    )){
      if(verbose){ print(paste("Found a rule for family name",model_leads[1])) }

      if(length(relevant_rules)<10){ # To restrict the maximum number of rules that are given back
        rulename<-stringr::str_remove_all(stringr::str_remove_all(found_rules[[i]]$full_rule,":"),"'")
        rule_begins<-unlist(gregexpr(pattern = "[",text = rulename,fixed = T))
        rule_stops<-unlist(gregexpr(pattern = "]",text = rulename,fixed = T))
        rel_rule<-paste0("IF ML solution has ",
          substr(rulename,rule_begins[1],rule_stops[1]),
                         " then ",
                         substr(rulename,rule_begins[2],rule_stops[2]),
                         ". ")

        relevant_rules<-c(relevant_rules,
                          #Removing colons from the full rule text
                          # list( rulename = list("confidence"=found_rules[[i]]$confidence,
                          #                       "conviction"= found_rules[[i]]$conviction,
                          #                       "leverage"=found_rules[[i]]$leverage,
                          #                       "lift"=found_rules[[i]]$lift
                          #                       )
                          #      )
                          rel_rule
        )
      }

    }


     ## Add other ways to identify relevant rules for the model as if snippets

    if( !is.na(stringr::str_match(found_rules[[i]]$full_rule,paste0("nr_dependencies_",model_report$nr_dependencies)) ) ){

      if(length(relevant_rules)<10){ # To restrict the maximum number of rules that are given back
        rulename<-stringr::str_remove_all(stringr::str_remove_all(found_rules[[i]]$full_rule,":"),"'")
        rule_begins<-unlist(gregexpr(pattern = "[",text = rulename,fixed = T))
        rule_stops<-unlist(gregexpr(pattern = "]",text = rulename,fixed = T))
        rel_rule<-paste0("IF ML solution has ",
                         substr(rulename,rule_begins[1],rule_stops[1]),
                         " then ",
                         substr(rulename,rule_begins[2],rule_stops[2]),
                         ". ")

        relevant_rules<-c(relevant_rules,
                          rel_rule )
      }

    }


    if( !is.na(stringr::str_match(found_rules[[i]]$full_rule,paste0("nr_hparams_",model_report$nr_hparams)) ) ){

      if(length(relevant_rules)<10){ # To restrict the maximum number of rules that are given back
        rulename<-stringr::str_remove_all(stringr::str_remove_all(found_rules[[i]]$full_rule,":"),"'")
        rule_begins<-unlist(gregexpr(pattern = "[",text = rulename,fixed = T))
        rule_stops<-unlist(gregexpr(pattern = "]",text = rulename,fixed = T))
        rel_rule<-paste0("IF ML solution has ",
                         substr(rulename,rule_begins[1],rule_stops[1]),
                         " then ",
                         substr(rulename,rule_begins[2],rule_stops[2]),
                         ". ")

        relevant_rules<-c(relevant_rules,
                          rel_rule )
      }

    }

  }

  return(relevant_rules)

}


#' @title Generate all results of the assistML analysis
#' @description Produces object and JSON with results of the analysis process
#'
#' @param models_choice List of two dataframes containing only the selected acc and nearly acc models
#' @param query_df Dataframe containing the query fields relevant for Hamming calculations.
#' @param usecase_rules Full list of all generated rules for both acc and nearly acc models.
#'
#' @return List with human readable details of chosen acceptable and nearly acceptable models
#' @export
#'
#' @examples
generate_results<-function(models_choice,usecase_rules,warnings,distrust_points,query_record,distrust_basis){
  verbose<-T

  ## Produces final report with all the results


  if(models_choice$naccms_choice[1] %in% c("none")){
    models_choice_dfs<-retrieve_settings(selected_models = list(
      "acceptable_models"=as.array(as.character(models_choice$accms_choice$model_name)),
      "nearly_acceptable_models"=as.array("none")
    ))
    print(" ")

    if(verbose){ print(paste("Obtained dataframes to get distance details for",nrow(models_choice_dfs$accmodels),"acceptable models")) }

  }else{
    models_choice_dfs<-retrieve_settings(selected_models = list(
      "acceptable_models"=as.array(as.character(models_choice$accms_choice$model_name)),
      "nearly_acceptable_models"=as.array(as.character(models_choice$naccms_choice$model_name))
    ))
    print(" ")

    if(verbose){ print(paste("Obtained dataframes to get distance details for",nrow(models_choice_dfs$accmodels),"acceptable models and",
                             nrow(models_choice_dfs$naccmodels),"nearly acceptable models")) }
  }






  # print(models_choice_dfs$accmodels[,1])
  # print(models_choice_dfs$naccmodels[,1])


  # Fields used to do the hamming distance details
  # comparison.query<-c("model_name","fam_name", "deployment","first_datatype","second_datatype","implementation","language","nr_hyperparams_label","rows")

  # print(names(models_choice_dfs$accmodels))
  # print(" ")
  # print(paste("Fields retrieved from generate results to do distance to query:"))
  # print(names(models_choice_dfs$accmodels)[names(models_choice_dfs$accmodels) %in% comparison.query])
  # print(models_choice_dfs$accmodels[,names(models_choice_dfs$accmodels) %in% comparison.query])
  # print(" ")


  if(verbose){ print("Acceptable Solutions") }

  accms_report=list()
  for (i in 1:nrow(models_choice$accms_choice)) {
    accms_report[[i]]<-generate_model_report(as.character(models_choice$accms_choice$model_name[i]),models_choice$accms_choice[i,2:ncol(models_choice$accms_choice)])


    # print("Fetching differences between selected model and query...")
    # print(nrow(models_choice_dfs$accmodels[as.character(models_choice_dfs$accmodels[,1]) %in% as.character(models_choice$accms_choice$model_name[i]),comparison.query]))

    # print("models_choice_dfs$accmodels...")
    # print(names(models_choice_dfs$accmodels))



    ## Pick first and second datatypes for both...

    # main_datatypes_acc<-data.frame("first_datatype","second_datatype",stringsAsFactors = F)
    # names(main_datatypes_acc)<-c("first_datatype","second_datatype")
    #
    # data_types_acc<-models_choice_dfs$accmodels[,names(models_choice_dfs$accmodels) %in% c("model_name","numeric_ratio","categorical_ratio", "datetime_ratio", "text_ratio")]
    # for (k in 1:nrow(data_types_acc)) {
    #   main_datatypes_acc<-rbind(main_datatypes_acc,names(sort(data_types_acc[k,2:5],decreasing = T)[1:2]))
    # }
    # main_datatypes_acc<-main_datatypes_acc[2:nrow(main_datatypes_acc),]

    # print(head(main_datatypes_acc))
    # print(main_datatypes_acc)

    # main_datatypes_nacc<-data.frame("first_datatype","second_datatype",stringsAsFactors = F)
    # names(main_datatypes_nacc)<-c("first_datatype","second_datatype")
    #
    # data_types_nacc<-models_choice_dfs$naccmodels[,names(models_choice_dfs$naccmodels) %in% c("model_name","numeric_ratio","categorical_ratio", "datetime_ratio", "text_ratio")]
    #
    # for (j in 1:nrow(data_types_nacc)) {
    #   main_datatypes_nacc<-rbind(main_datatypes_nacc,names(sort(data_types_nacc[j,2:5],decreasing = T)[1:2]))
    # }
    # main_datatypes_nacc<-main_datatypes_nacc[2:nrow(main_datatypes_nacc),]

    # print(main_datatypes_nacc)


# Actually looking for the hamming details
    # print("Launching hamming_details()")

    # accms_report[[i]]$differences<-hamming_details(cbind(models_choice_dfs$accmodels[as.character(models_choice_dfs$accmodels[,1]) %in% as.character(models_choice$accms_choice$model_name[i]),names(models_choice_dfs$accmodels) %in% comparison.query],main_datatypes_acc[i,]),
                                                   # query_df)

    # print(paste("Found differences for",as.character(models_choice$accms_choice$model_name[i])))

    accms_report[[i]]$rules<-add_rules(accms_report[[i]],usecase_rules)
  }





  if(all(models_choice$naccms_choice %in% c("none"))) {

    return(list(
      "summary"=list(
        "query_issued"=query_record,
        "acceptable_models"=length(accms_report),
        ## Dynamically compute the basis for the distrust score
        "distrust_score"=signif(distrust_points/distrust_basis,4),
        "warnings"=warnings
      ),
      "acceptable_models"=accms_report
    ))

  }else{
    #  Case there are nearly acceptable solutions chosen
    if(verbose){ print("Nearly Acceptable Solutions") }

    naccms_report=list()
    for (j in 1:nrow(models_choice$naccms_choice)) {
      naccms_report[[j]]<-generate_model_report(as.character(models_choice$naccms_choice$model_name[j]),models_choice$naccms_choice[j,2:ncol(models_choice$naccms_choice)])

      # print(nrow(models_choice_dfs$naccmodels[as.character(models_choice_dfs$naccmodels[,1]) %in% as.character(models_choice$naccms_choice$model_name[j]),comparison.query]))
      # naccms_report[[j]]$differences<-hamming_details(cbind(models_choice_dfs$naccmodels[as.character(models_choice_dfs$naccmodels[,1]) %in% as.character(models_choice$naccms_choice$model_name[j]),names(models_choice_dfs$naccmodels) %in% comparison.query],main_datatypes_nacc[j,]),
      #                                                 query_df)

      naccms_report[[j]]$rules<-add_rules(naccms_report[[j]],usecase_rules)
    }

    # write(rjson::toJSON(accms_report,indent = 3),file = "accms_report.json",append = F)
    # write(rjson::toJSON(naccms_report,indent = 3),file = "naccms_report.json",append = F)

    return(list(
      "summary"=list(
        "query_issued"=query_record,
        "acceptable_models"=length(accms_report),
        "nearly_acceptable_models"=length(naccms_report),
        "distrust_score"=signif(distrust_points/distrust_basis,4),
        "warnings"=warnings
      ),
      "acceptable_models"=accms_report,
      "nearly_acceptable_models"=naccms_report

    ))
  }



}
