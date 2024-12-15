

#' @title Choose models relevant for the usecase and dataset
#' @description Filters out models by filtering based on characteristics of the usecase task and dataset.
#'
#' @param task_type String to describe the kind of classif task (\code{binary} or \code{multiclass}). Must match output of \code{\link{query_usecase}}
#' @param output_type String to describe the type of output expected. Must match the output of \code{\link{query_usecase}}
#' @param data_features List describing all the details of the dataset features
#' @importFrom mongolite mongo
#'
#' @return A data frame with the model rows that match the description criteria.
#' @export
#'
#' @examples
choose_models<-function(task_type,output_type,data_features){

  verbose<-T
  if(verbose){
    print("Entered choose_models() with values:")
    print(task_type)
    print("")
    print(output_type)
    print("")

  }
    # print(data_features)

  datasetsMongo<-mongolite::mongo(collection = "datasets",
                                  db = "assistml",
                                  url = "mongodb://admin:admin@mongodb")

  base_models<-mongolite::mongo(collection = "base_models",
                                db = "assistml",
                                url = "mongodb://admin:admin@mongodb")

  enriched_models<-mongolite::mongo(collection = "enriched_models",
                                    db="assistml",
                                    url="mongodb://admin:admin@mongodb")






  ## Determine dataset similarity 0 ####

  if(verbose){
    print("Checking similarity 0")
  }


  # Getting data from base models
  sim_0<-base_models$find('{}', fields='{"Model.Data_Meta_Data.classification_output":1,"Model.Data_Meta_Data.classification_type":1,"Model.Info.name":1,"_id":0}')



  # Finding the actual models with similar output and type from base models
  # sim_0_codes<-sim_0$Model$Info$name[sim_0$Model$Data_Meta_Data$classification_type %in% "Binary" & sim_0$Model$Data_Meta_Data$classification_output %in% "single"]
  sim_0_codes<-sim_0$Model$Info$name[sim_0$Model$Data_Meta_Data$classification_type %in% task_type & sim_0$Model$Data_Meta_Data$classification_output %in% output_type]

  if(length(sim_0_codes)>0){
    sim_level<-0
    if(verbose){
      print(paste("Models with similarity 0:",length(sim_0_codes)))
    }



    ## Determine datatset similarity 1####

    # Feature ratios from the new dataset being analyzed
    type_ratios<-c(data_features$numeric_ratio,
                   data_features$categorical_ratio,
                   data_features$datetime_ratio,
                   data_features$unstructured_ratio)
    names(type_ratios)<-c("numeric_ratio","categorical_ratio","datetime_ratio","unstructured_ratio")


    # Getting the data type ratios for the new dataset being used in the query.
    data_repo<-datasetsMongo$find('{}',fields = '{"Info":1,"_id":0}')
    data_repo<-data_repo$Info[c(2,7:10)]




    if(verbose){
      print("")
      print("Checking similarity 1")
    }

    if(verbose){ writeLines(paste("Looking for matches to\n",paste(names(type_ratios)[!type_ratios==0],collapse = ", "))) }



    # similar_ratios<-c()
    sim_1_usecases<-c()
    for (i  in 1:nrow(data_repo)) {
      overlap_datatypes<-names(data_repo[,2:5])[!data_repo[i,2:5]==0] %in% names(type_ratios)[!type_ratios==0]

      # print(" ")
      # print(paste("Case",i))
      # print(paste("Total number of data types", length(names(data_repo[,2:5])[!data_repo[i,2:5]==0]) ))
      # print(paste("Actual overlap of data types", sum(overlap_datatypes) ))

      if(length(names(data_repo[,2:5])[!data_repo[i,2:5]==0]) == sum(overlap_datatypes)){

        # print(paste("Dataset",data_repo[i,1] ,"has similarity 1"))
        sim_1_usecases<-c(sim_1_usecases,data_repo[i,1])

      }else{
        print(paste("Dataset",data_repo[i,1] ,"has NO similarity 1"))
      }

      # if( sum(names(sort(type_ratios,decreasing = T)[1:2]) %in% names(sort(data_ratios[i,2:5],decreasing = T)[1:2]))>0 ){
        # print(paste("Found relevant model:",data_ratios[i,1]))
        # similar_ratios<-c(similar_ratios,data_ratios[i,1])
      # }
    }



    # Keeping only the models trained on a dataset with similarity 1
    # Sim 1 codes can only be those already obtained in sim_0_codes
    sim_1_codes<-base_models$find(query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",paste0(sim_0_codes,sep = "",collapse = "\", \""),"\"]}}'"))),
                     fields = '{"Model.Info.name":true,"Model.Info.use_case":true,"_id":false}' )

    sim_1_codes<-sim_1_codes$Model$Info$name[ sim_1_codes$Model$Info$use_case %in% sim_1_usecases ]

    if(length(sim_1_codes)>0){
      if(verbose){
        print(paste("Models with similarity 1:",length(sim_1_codes)))
      }

      sim_level<-1
      ## Determine datatset similarity 2####

      if(verbose){
        print("")
        print("Checking similarity 2")
      }


      decile_upper<-type_ratios[type_ratios!=0]+0.05
      decile_lower<-type_ratios[type_ratios!=0]-0.05

      sim_1_datarepo<-data_repo[data_repo$use_case %in% sim_1_usecases,]

      sim_2_usecases<-c()
      # Checking whether the data type ratios of each dataset are within 1 decile of distance from the ratios of the new dataset
      for (j in 1:nrow(sim_1_datarepo)) {

        overlap_datatypes<-names(sim_1_datarepo[,2:5])[sim_1_datarepo[j,2:5]!=0]


        sim_decile<-F
        for (k in 1:length(overlap_datatypes) ) {
          if(sim_1_datarepo[j,names(sim_1_datarepo) %in% overlap_datatypes[k] ] < decile_upper[overlap_datatypes[k] ] &
             sim_1_datarepo[j,names(sim_1_datarepo) %in% overlap_datatypes[k] ] > decile_lower[overlap_datatypes[k] ]){
            sim_decile<-T
          }else{
            sim_decile<-F
          }
        }

        if(sim_decile){
          #  Only if all overlapping data types are in the same decile range
          print(paste("Usecase",sim_1_datarepo[j,1],"has similarity level 2" ))
          sim_2_usecases<-c(sim_2_usecases,sim_1_datarepo[j,1])
        }else{
          print(paste("Usecase",sim_1_datarepo[j,1],"has NO similarity level 2" ))
        }
      }

      sim_2_codes<-base_models$find(query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",paste0(sim_1_codes,sep = "",collapse = "\", \""),"\"]}}'"))),
                                    fields = '{"Model.Info.name":true,"Model.Info.use_case":true,"_id":false}' )

      sim_2_codes<-sim_2_codes$Model$Info$name[ sim_2_codes$Model$Info$use_case %in% sim_2_usecases ]

      if(length(sim_2_codes)>0){
        sim_level<-2
        if(verbose){
          print(paste("Models with similarity 2:",length(sim_2_codes)))
        }

        ## Determine datatset similarity 3

        if(verbose){
          print("")
          print("Checking similarity 3")
          print("")
        }



        # Collecting feature numbers from S2 datasets
        feat_numbers<-datasetsMongo$find(query = eval(parse(text =  paste0("'{\"Info.use_case\":{\"$in\":[\"",paste0(sim_2_usecases, collapse = "\",\""),"\"]}}'") )),
                                         fields ='{"Info.dataset_name":1,"Info.features":1,"_id":0}'  )
        feat_numbers<-data.frame("dataset_name"=feat_numbers$Info$dataset_name,
                                 "features"=feat_numbers$Info$features)


        # Getting all numerical metafeatures of the new dataset
        new_featnum_analysis<-data.frame("monotonous_filtering"=0,"anova_f1"=0,"anova_pvalue"=0,"mutual_info"=0,"missing_values"=0,"min_orderm"=0,"max_orderm"=0)
        for(j in 1:length(data_features$numerical_features)){

          # if(verbose){
          #   print("Adding to DF of numeric")
          #   print(c(data_features$numerical_features[[j]]$monotonous_filtering,
          #           data_features$numerical_features[[j]]$anova_f1,
          #           data_features$numerical_features[[j]]$anova_pvalue,
          #           data_features$numerical_features[[j]]$mutual_info,
          #           data_features$numerical_features[[j]]$missing_values,
          #           data_features$numerical_features[[j]]$min_orderm,
          #           data_features$numerical_features[[j]]$max_orderm))
          # }

          new_featnum_analysis<-rbind(new_featnum_analysis,
                                      c(data_features$numerical_features[[j]]$monotonous_filtering,
                                        data_features$numerical_features[[j]]$anova_f1,
                                        data_features$numerical_features[[j]]$anova_pvalue,
                                        data_features$numerical_features[[j]]$mutual_info,
                                        data_features$numerical_features[[j]]$missing_values,
                                        data_features$numerical_features[[j]]$min_orderm,
                                        data_features$numerical_features[[j]]$max_orderm) )

        }
        new_featnum_analysis<-new_featnum_analysis[-1,]



        # Getting all categorical metafeatures of the new dataset
        new_featcal_analysis<-data.frame( "missing_values"=0,"nr_levels"=0,"imbalance"=0,"mutual_info"=0,"monotonous_filtering"=0)
        for(j in 1:length(data_features$categorical_features)){
          new_featcal_analysis<-rbind(new_featcal_analysis,
                                      c(data_features$categorical_features[[j]]$missing_values,
                                        data_features$categorical_features[[j]]$nr_levels,
                                        data_features$categorical_features[[j]]$imbalance,
                                        data_features$categorical_features[[j]]$mutual_info,
                                        data_features$categorical_features[[j]]$monotonous_filtering) )
        }

        new_featcal_analysis<-new_featcal_analysis[-1,]

        # print("Collected detailed metafeatures for similarity 3 check of new dataset")

        ## TODO!:Performance improvement. Use the smallest dataset to do the comparison of similarity 3. Uncomment if below and implement ELSE ####
        # if(data_features$Info$features <= feat_numbers[i,"features"] ){
          #   # If the new dataset is smaller than the one in the repo, use the new dataset as basis for the comparison


        s3_datasets<-c()
        for(i in 1:nrow(feat_numbers) ){

          print(paste("Checking similarity 3 for",feat_numbers[i,1]))

              # Getting all numerical metafeatures of current dataset
              featnum_analysis_current<-data.frame("monotonous_filtering"=0,"anova_f1"=0,"anova_pvalue"=0,"mutual_info"=0,"missing_values"=0,"min_orderm"=0,"max_orderm"=0)

              current_numfeats<-datasetsMongo$find(query = eval(parse(text =  paste0("'{\"Info.dataset_name\":\"",feat_numbers[i,1],"\"}'") )),
                                                   fields = '{"Features.Numerical_Features":1,"_id":0}')
              current_numfeats<-current_numfeats$Features$Numerical_Features

              for(k in 1:ncol(current_numfeats)){
                featnum_analysis_current<-rbind(featnum_analysis_current,
                                                c(current_numfeats[[k]]$monotonous_filtering,
                                                  current_numfeats[[k]]$anova_f1,
                                                  current_numfeats[[k]]$anova_pvalue,
                                                  current_numfeats[[k]]$mutual_info,
                                                  current_numfeats[[k]]$missing_values,
                                                  current_numfeats[[k]]$min_orderm,
                                                  current_numfeats[[k]]$max_orderm) )
              }
              featnum_analysis_current<-featnum_analysis_current[-1,]



              # Getting all categorical metafeatures of current dataset
              featcat_analysis_current<-data.frame("missing_values"=0,"nr_levels"=0,"imbalance"=0,"mutual_info"=0,"monotonous_filtering"=0)

              current_catfeats<-datasetsMongo$find(query = eval(parse(text =  paste0("'{\"Info.dataset_name\":\"",feat_numbers[i,1],"\"}'") )),
                                                   fields = '{"Features.Categorical_Features":1,"_id":0}')
              current_catfeats<-current_catfeats$Features$Categorical_Features

              for(k in 1:ncol(current_catfeats)){
                featcat_analysis_current<-rbind(featcat_analysis_current,
                                                c(current_catfeats[[k]]$missing_values,
                                                  current_catfeats[[k]]$nr_levels,
                                                  current_catfeats[[k]]$imbalance,
                                                  current_catfeats[[k]]$mutual_info,
                                                  current_catfeats[[k]]$monotonous_filtering
                                                ) )
              }
              featcat_analysis_current<-featcat_analysis_current[-1,]


              ## Find for every numerical feature in the NEW dataset, a similar feature from the current dataset
              if(verbose){
                print("Numerical features sim 3 checks")
              }

              num_found<-c()
              for (j in 1:nrow(new_featnum_analysis)) {

                # print(head(featnum_analysis_current))
                # print(new_featnum_analysis[i,])

                # print(paste("Round",i))
                # Checking monotonous filtering values
                if( sum( featnum_analysis_current$monotonous_filtering <= new_featnum_analysis$monotonous_filtering[j]+0.05 &
                         featnum_analysis_current$monotonous_filtering >= new_featnum_analysis$monotonous_filtering[j]-0.05 ) >0 ){


                  # print(paste("Feature",i," has similar monotonous value"))

                  mono_matches<-which(featnum_analysis_current$monotonous_filtering <= new_featnum_analysis$monotonous_filtering[j]+0.05 &
                                        featnum_analysis_current$monotonous_filtering >= new_featnum_analysis$monotonous_filtering[j]-0.05)

                  # print("Position to be examined")
                  # print(mono_matches)


                  # Checking mutual information values of those picked from monotonous values

                  for (q in 1:length(mono_matches)) {
                    # print( paste("subround",q) )

                    if( sum(featnum_analysis_current$mutual_info[mono_matches[q]] <= new_featnum_analysis$mutual_info[j]+0.02 &
                            featnum_analysis_current$mutual_info[mono_matches[q]] >= new_featnum_analysis$mutual_info[j]-0.02)>0 ){
                      print(paste("Found similar feature", names(current_numfeats)[mono_matches[q]]))
                      num_found<-c(num_found,
                               names(current_numfeats)[mono_matches[q]])

                    }
                  }


                }

                # End of numeric S3 check. Results in num_found
              }



              ## Find for every categorical feature in the NEW dataset, a similar feature from the current dataset


              if(verbose){
                print("categorical features sim 3 checks")
              }

              cat_found<-c()
              for (j in 1:nrow(new_featcal_analysis)) {

                # print(paste("Round",i))
                # Checking monotonous filtering values
                if( sum( featcat_analysis_current$monotonous_filtering <= new_featcal_analysis$monotonous_filtering[j]+0.05 &
                         featcat_analysis_current$monotonous_filtering >= new_featcal_analysis$monotonous_filtering[j]-0.05 )>0 ){


                  # print(paste("Feature",i," has similar monotonous value"))

                  mono_matches<-which(featcat_analysis_current$monotonous_filtering <= new_featcal_analysis$monotonous_filtering[j]+0.05 &
                                        featcat_analysis_current$monotonous_filtering >= new_featcal_analysis$monotonous_filtering[j]-0.05)


                  # Checking mutual information values of those picked from monotonous values

                  for (q in 1:length(mono_matches)) {
                    # print( paste("subround",q, "for") )

                    if( sum(featcat_analysis_current$mutual_info[mono_matches[q]] <= new_featcal_analysis$mutual_info[j]+0.02 &
                            featcat_analysis_current$mutual_info[mono_matches[q]] >= new_featcal_analysis$mutual_info[j]-0.02)>0 ){
                      # print(paste("Found similar feature", names(current_catfeats)[mono_matches[q]]))
                      cat_found<-c(cat_found,
                                   names(current_catfeats)[mono_matches[q]])

                    }
                  }


                }

                # End of categorical S3 check. Results in cat_found
              }

              s3_ratio<-(length(num_found)+length(cat_found)) / (length(current_numfeats)+length(current_catfeats))

              print(paste("Dataset",feat_numbers[i,1],"has",
                          round(s3_ratio,2)*100,"% of S3 features"))

              if(s3_ratio>=0.5){

                s3_datasets<-c(s3_datasets,feat_numbers[i,1])
              }




            # End of similarity 3 check
        }

        print(paste("s3_datasets:",s3_datasets))


        sim_3_codes<-base_models$find(query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",paste0(sim_2_codes,sep = "",collapse = "\", \""),"\"]}}'"))),
                                      fields = '{"Model.Info.name":true,"Model.Info.use_case":true,"_id":false,"Model.Data_Meta_Data.dataset_name":true}' )

        sim_3_codes<-sim_3_codes$Model$Info$name[ sim_3_codes$Model$Data_Meta_Data$dataset_name %in% s3_datasets ]






          # }else{
          #   # If the new dataset is bigger than the one in the repo, use the dataset in the repo as basis for the comparison
          #   ## TODO!: Implement the logic above, but inversed between repo and new dataset
          # }



        if(length(sim_3_codes)>0){
        sim_level<-3
        print("There are models with similarity level 3")
        }



      # End of IF when there is Sim 2
      }

    # End of IF when there is Sim 1
    }



  ## Return only codes with the highest similarity####




    sim_codes<-switch(sim_level+1,
      sim_0_codes,
      sim_1_codes,
      sim_2_codes,
      sim_3_codes,
    )


    sim_models<-enriched_models$find(query =eval(parse(text = paste0("'{\"model_name\":{\"$in\":[\"",paste0(sim_codes,sep = "",collapse = "\", \""),"\"]}}'"))),
                                     fields = '{"model_name":true,"accuracy":true,"precision":true,"recall":true,"training_time_std":true,"performance_score":true,"_id":false}' )



    # Casting the training time std and performance score as numerics
    ## TODO!: Reload in base and enriched models "LGR_adult_001" "NBY_adult_001" "SVM_adult_001"####
    ## TODO!:Remove workaround####
    # sim_models<-sim_models[!(sim_models$model_name %in% c("LGR_adult_001","NBY_adult_001","SVM_adult_001") ),]
    sim_models$accuracy<-as.numeric(sim_models$accuracy)
    sim_models$precision<-as.numeric(sim_models$precision)
    sim_models$recall<-as.numeric(sim_models$recall)
    sim_models$training_time_std<-as.numeric(sim_models$training_time_std)
    sim_models$performance_score<-as.numeric(sim_models$performance_score)



    ##Return only the columns for clustering to avoid fetching all data before it is actually needed when retrieve_settings() does that for the calculation of hamming distances
    return(list(
      sim_models
      ,
      sim_level
    ))


  }else{
    # Finish assistml

    print("Choose.models() Finished. No dataset similarity of any kind could be determined")
    sim_level<-NULL
    return(list("No dataset similarity of any kind could be determined",sim_level))

  }


}


