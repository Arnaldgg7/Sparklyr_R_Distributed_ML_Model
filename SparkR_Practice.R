library(sparklyr)
library(dplyr)
library(reshape2)
library(rlang)

# If needed
# spark_install(version = "2.4")

# To set the right JAVA_HOME variable (if required):
# Sys.setenv(JAVA_HOME="C:/Program Files/Java/jre1.8.0_271")

config <- spark_config()
config$`sparklyr.shell.driver-memory` <- "4G"
config$`sparklyr.shell.executor-memory` <- "4G"

sc <- spark_connect(master = "local", version = "2.4", config = config)

csvPath <- paste0(getwd(), "/PacientesSim.csv")

# We can read from other sources, like HDFS, but in this case we directly
# read from a CSV:
df <- spark_read_csv(sc, "pacientes", csvPath, header=T, null_value="NA", 
                     delimiter=";")

src_tbls(sc) # dplyr function

# How many elements do we have?
df %>% count


# Assigning a value to NA values to then be used in discretization:
naImputation <- function(df) {
  cols <- colnames(df)
  for (col in cols) {
    sym_col <- rlang::sym(col)
    df <- df %>% mutate((!!col) := ifelse(is.na(!!sym_col), -1000, (!!sym_col)))
  }
  return (df)
}




################################################################################
################################# BUCKETIZATION ################################
################################################################################

bucketization <- function(df, vectorvars) {
  # Defining the intervals for each variable:
  DiasEstancia <- c(-Inf, -999,12,42,71, Inf)
  Hemoglobina <- c(-Inf, -999,12, Inf)
  Creatinina <- c(-Inf, -999,1.12, Inf)
  Albumina <- c(-Inf, -999,3.5,5.1, Inf)
  Barthel <- c(-Inf, -999,20,61,91,99, Inf)
  Pfeiffer <- c(-Inf, -999,2,4,8, Inf)
  DiferenciaBarthel <- c(-Inf, -999,-20,21, Inf)
  DiferenciaPfeiffer = c(-Inf, -999,-2,3, Inf)
  
  for (var in vectorvars) {
    sym_col <- rlang::sym(var)
    df <- df %>% ft_bucketizer(sym_col, paste0("B",var), splits= get(var))
  }
  
  # Removing the original columns, so that we only keep the discretized columns:
  df <- df %>% select(., -all_of(vectorvars))
  
  return(df)
}




################################################################################
############################## ONE-HOT ENCODING ################################
################################################################################

# In order to apply One-Hot Encoding to the required variables, we need to first
# deal with the remaining '-1000' values, which are still present in the 'indicator'
# columns. We checked the minimum and maximum values of these columns
# and we realised that all of them don't use the '0' index (which has been used
# for 'NA' when bucketing), except for the column 'IndicadorDemencia'.

# Consequently, we convert the '0' and '1' Binary values of the 'IndicadorDemencia'
# to '1' and '2', respectively, so that they share the same index than the rest
# of columns, and then we can interpret the -1000 of the NA values as '0' for all
# Indicator Columns as well, thereby replacing such -1000 values by the '0' index.


oneHotEncoding <- function(df, vectorvars, previousProcessedCols) {
  # Adding up '1' to all non-null values from 'IndicadorDemencia':
  df <- df %>% mutate(IndicadorDemencia = ifelse(IndicadorDemencia != -1000,
                                                 IndicadorDemencia+1,
                                                 IndicadorDemencia))
  
  # Replacing all null values (represented by '-1000') in all Indicator Columnns
  # by the '0' index:
  for (col in vectorvars) {
    sym_col <- rlang::sym(col)
    df <- df %>% mutate((!!col) := ifelse((!!sym_col) == -1000, 0, (!!sym_col)))
  }
  
  # One-Hot Encoding of the variables (all discretized and indicator variables
  # at once), first concatenating all of them in the same vector:
  one.hot.var <- c(vectorvars, paste0("B", previousProcessedCols), "Reingreso") %>% 
    setdiff(., "BDiasEstancia")
  
  # And now performing the One-Hot Encoding of all those columns:
  for (var in one.hot.var) {
    sym_col <- rlang::sym(var)
    df <- df %>% ft_one_hot_encoder(sym_col, paste0("OH",var))
  }
  
  # Removing original columns, so that we only keep the One-Hot encoded ones:
  df <- df %>% select(., -all_of(one.hot.var))
  
  return(df)
}




################################################################################
######################## TOKENIZATION & VECTORIZATION ##########################
################################################################################

tokenizationAndVectorizacion <- function(df, vectorvars) {
  # Tokenization of List Variables:
  # Removing leading and trailing spaces, tokenizing and vectorizing:
  for (var in vectorvars){
    sym_col <- rlang::sym(var)
    df <- df %>% mutate((!!var) := trimws(!!sym_col))
    df <- df %>% ft_regex_tokenizer(var, paste0("T",var), pattern=",")
    # As stated in the Java Practice, we use CountVectorizer, as it fits best
    # with the Linear SVM we are going to use:
    df <- df %>% ft_count_vectorizer(paste0("T",var), paste0("W",var),
                             vocab_size=1000)
  }
  
  # Removing original columns, so that we only keep the One-Hot encoded ones:
  df <- df %>% select(., -all_of(vectorvars)) %>% 
    select(., -all_of(paste0("T", vectorvars)))
  
  return(df)
}




################################################################################
################################# ASSEMBLING ###################################
################################################################################

vectorAssembling <- function(df) {
  # Vectorization assembling of all Predictor Variables in a vector we call
  # 'features', so that we can call it from a Spark MLLib implementation underneath:
  
  predictors <- c(colnames(df)) %>% setdiff(., c("Id", "BDiasEstancia"))
  
  df <- df %>% ft_vector_assembler(predictors, output_col="features") %>% 
    select("features", "BDiasEstancia")
  
  return(df)
}




################################################################################
############################## TRAIN-TEST SPLITTING ############################
################################################################################

trainTestSplit <- function(df, train) {
  # We will use rain-Test Split using 70-30% rule, as it was also stated in the
  # Java model implementation, as well as stating the same 'seed' to be able
  # to compare:
  splits <- df %>% sdf_random_split(
    training= train,
    test = 1-train,
    seed=42
  )
  return(splits)
}




################################################################################
############################## FINAL MODEL FITTING #############################
################################################################################
# Vectors to use:
discretizedVariables <- c("DiasEstancia", "Hemoglobina", "Creatinina", "Albumina",
                          "Barthel", "Pfeiffer", "DiferenciaBarthel", "DiferenciaPfeiffer")

indicatorInputCols <- c("IndicadorDemencia", "IndicadorConstipacion",
                        "IndicadorSordera", "IndicadorAltVisual")

listVars <- c("ListaDiagnosticosPri", "ListaDiagnosticosSec", "ListaProcedimientosPri",
               "ListaProcedimientosSec", "ListaCausasExternas")


# Pre-Processing Pipeline (we have implemented all function using for-loops, as
# they internally are transformed into individual Spark queries but we end up
# writing less code and all becomes more neat):
splits <-  df %>% naImputation() %>% bucketization(., discretizedVariables) %>% 
  oneHotEncoding(., indicatorInputCols, discretizedVariables) %>% 
  tokenizationAndVectorizacion(., listVars) %>% vectorAssembling() %>% 
  trainTestSplit(., 0.70)

# Provided that we cannot use OVR with Cross Validation in GridSearch (in our
# case, as stated above, we cannot even use Cross Validation), we have created
# a function to test several parameters with the Linear SVM. Using a Linear SVM
# here, although it does not include a Kernel Function, it also performs very
# well given the huge dimensionality of the data because of the use of the
# Vectorization with the List Variables:
SVC_ParamSelection <- function(param, train){
  bestAcc <- 0
  bestParam <- param[1]
  bestConfMat <- NULL
  for (p in param) {
    SVC <- ml_linear_svc(sc,
                         reg_param = p,
                         max_iter = 20,
                         standardization = TRUE,
                         features_col = "features",
                         label_col = "BDiasEstancia",
                         prediction_col = "prediction"
    )
    
    ovr <- train %>% ml_one_vs_rest(.,
                                    classifier = SVC,
                                    features_col = "features",
                                    label_col = "BDiasEstancia",
                                    prediction_col = "prediction"
    )
    
    pred <- ml_predict(ovr, splits$test)
    
    cf <- pred %>% count(prediction, BDiasEstancia, sort=T) %>% collect
    
    confLocal <- dcast(cf, BDiasEstancia~prediction, value.var = 'n',
                       fill=0)
    
    # Confusion Matrix:
    confLocalMat <- as.matrix.data.frame(confLocal[,-1])
    
    # Accuracy:
    acc <- sum(diag(confLocalMat))/sum(confLocalMat)
    
    if (acc > bestAcc) {
      bestParam <- p
      bestAcc <- acc
      bestConfMat <- confLocalMat
    }
  }
  cat("Best parameter: ", bestParam)
  
  return(list(param=bestParam, acc=bestAcc, cm=bestConfMat))
}

params <- c(100, 10, 1, 0.1, 0.01)

bestModel <- SVC_ParamSelection(params, splits$training)




################################################################################
######################## SUMMARY OF THE MODEL METRICS ##########################
################################################################################
# Function to get Confidence Intervals for our metrics:
getIntervals <- function(metric, n) {
  pct_metric <- round(metric*100,2)
  StdError <- 1.967*sqrt((metric*(1-metric))/n)
  return(list(val=pct_metric, low=round((metric-StdError)*100,2),
              high=round((metric+StdError)*100,2)))
}

cat("Training samples: ", sdf_nrow(splits$training), "\n")
cat("Test samples: ", sdf_nrow(splits$test), "\n\n")

cat("Global Accuracy of the model = ", round(bestModel$acc*100,2), "\n\n")

ranges <- c("for less than 12 days = ", "for between 12 and 41 days = ",
  "for between 42 to 70 days = ", "for more than 70 days = ")

n <- sdf_nrow(splits$test)

i <- 1
while (i<5){
  precision <- as.numeric(bestModel$cm[i,i]/sum(bestModel$cm[,i]))
  precision_vals <- getIntervals(precision, n)
  
  recall <- as.numeric(bestModel$cm[i,i]/sum(bestModel$cm[i,]))
  recall_vals <- getIntervals(recall, n)
  
  f1Score <- (2*precision*recall)/(precision+recall)
  f1Score_vals <- getIntervals(f1Score, n)
  
  cat("Precision ", ranges[i], precision_vals$val, "%, with Confidence Intervals (95%):
        (", precision_vals$low, "%, ", precision_vals$high, "%)\n")
  cat("Recall ", ranges[i], recall_vals$val, "%, with Confidence Intervals (95%):
        (", recall_vals$low, "%, ", recall_vals$high, "%)\n")
  cat("Precision ", ranges[i], f1Score_vals$val, "%, with Confidence Intervals (95%):
        (", f1Score_vals$low, "%, ", f1Score_vals$high, "%)\n")
  i <- i+1
}

print("Confusion Matrix:")
print(as.matrix.data.frame(bestModel$cm))

