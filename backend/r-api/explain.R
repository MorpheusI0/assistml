explain_python<-function(model_code){
 verbose<-T

 if(verbose){
   print(paste("Generating explanation for",model_code))
 }


 script<-"import os\n"

 script<-paste0(script,"os.system(\'python3 /app/python-modules/explainability.py -m ",model_code,"\')")

 # paste0(c("SVM_kick_003","DTR_kick_012","RFR_kick_022","DTR_kick_011","NBY_bank_002"),collapse = "\",\"")


 if(verbose){
    print(paste("Calling python script to generate explanations for",model_code))
 }

 working_dir <- "/app/working"
 if (!dir.exists(working_dir)) {
    dir.create(working_dir)
 }

 explain_gen_py_path <- file.path(working_dir, "explain_gen.py")
 write(script,file = explain_gen_py_path,append = F)
 reticulate::py_run_file(explain_gen_py_path)

 if(verbose){
    print(paste("Retrieving explanations from Mongo for",model_code))
 }
 base<-mongolite::mongo(
    collection = "base_models",
    db = "assistml",
    url = "mongodb://admin:admin@mongodb/"
 )
 return(
    base$find( query =eval(parse(text = paste0("'{\"Model.Info.name\":{\"$in\":[\"",model_code,"\"]}}'") )) )
 )

}
