"""
Combined code for AMLLS project
"""
# Import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score ,StratifiedGroupKFold , train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, make_scorer, balanced_accuracy_score, roc_curve, auc, log_loss
from sklearn.impute import KNNImputer
from sklearn.multiclass import OneVsRestClassifier
import random
import lightgbm as lgb
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import svm
from catboost import CatBoostClassifier
import re
import string
import pickle
import copy
import glob
import shap

import matplotlib.pyplot as plt
import seaborn as sns




# from imblearn.over_sampling import SMOTENC
# Our classes and functions
import Functions_ML_Project
import Class_ML_Project

# Global variables
# Define categorical feature list
# Subset all categorical (removing the non informative medication)
CATEGORICAL = ['age','race', 'gender', 'medical_specialty', 'max_glu_serum',
               'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
               'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
               'diabetesMed', 'admission_type_descriptor', 'discharge_disposition_descriptor',
               'admission_source_descriptor']

# Define numerical feature list
NUMERICAL = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
             'number_diagnoses', 'number_outpatient', 'number_emergency', 'number_inpatient']

# Define irrelevant feature list
IRRELEVANT_FEATURES = ["payer_code",'diag_1','diag_2','diag_3','repaglinide','nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor','admission_type_id','discharge_disposition_id','admission_source_id','patient_nbr']


# Define columns for OHE
OHE_regular_cols = ['race', 'gender', 'medical_specialty', 'insulin', 'diabetesMed', 'admission_type_descriptor',
                    'discharge_disposition_descriptor']
OHE_4_to_2_cols = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
diagnoses_cols = ['diag_1_cat', 'diag_2_cat', 'diag_3_cat']

# Note age feature will change to numerical later in the code!


# Define default models to initail test
models_defualt = {'Logisitic' : LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000),
          'XGBOOST' : XGBClassifier(use_label_encoder=False,random_state = 42, enable_categorical = True),
          'Tree' : DecisionTreeClassifier(random_state=42),
          'LGBM' : lgb.LGBMClassifier(random_state=42),
          'CatBoost' : CatBoostClassifier(random_seed = 42 , cat_features = ['age']),
                 'SVM': svm.SVC(kernel='linear',random_state=42, probability=True)}
random.seed(42)


# read the data file and gets train and test databases
db_train_df, db_test_df = Functions_ML_Project.clean_data_and_return_train_and_test('diabetic_data.csv')

# features pre-processing
id_names, mapping_dict = Functions_ML_Project.preform_ids_maping('IDS_mapping.csv')

# replace the disease code by disease name
training_df_new = Functions_ML_Project.apply_mapping(db_train_df, id_names, mapping_dict)

#
training_df_new = Functions_ML_Project.feature_engineering(training_df_new)

# Define the pipeline

pipeline = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES)),
                     ('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL)),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

# Fit and transform the DataFrame
training_clean_imputed = pipeline.fit_transform(training_df_new)

# Get the removed column names
removed_column_names = pipeline.named_steps['feature_remover'].features_to_remove


# Finally print the results
print("DataFrame head after feature selection and imputation:")
print(training_clean_imputed.head())
print("DataFrame shape after feature selection and imputation:")
print(training_clean_imputed.shape)



# remove readmitted above 30 days 
training_clean_imputed = training_clean_imputed[training_clean_imputed.readmitted.isin(['<30', 'NO'])]

# run default models and compare with cross validation  
X = training_clean_imputed.drop('readmitted',axis  = 1 )

y = training_clean_imputed['readmitted']
y = LabelEncoder().fit_transform(y)
multi_model_cv = Class_ML_Project.MultiModelCV(models=models_defualt,
                                               score=balanced_accuracy_score,
                                               balance_threshold = 0.3)
multi_model_cv.fit(X, y)
defualt_models_original_dataframe = multi_model_cv.get_results()
print(defualt_models_original_dataframe)

#defualt_models_original_dataframe.to_csv('default_models_original_dataframe.csv', index=False)


# --------------------------------------------------------------- #
#   run pipeline and defualt models on oversampled data by GANs   #
# --------------------------------------------------------------- #

# --------------------------------- #
#                CTGAN              #
# --------------------------------- #


IRRELEVANT_FEATURES_for_GAN = ['repaglinide','nateglinide','chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone',
 'tolazamide','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-pioglitazone',
 'admission_source_descriptor']

pipeline_GAN = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES_for_GAN)),
                     ('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='Other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL)),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

id_fold = pd.read_csv('id_fold.csv')

GAN_synthesized_data = [f for f in os.listdir() if 'CTGAN' in f and os.path.isfile(os.path.join(f))]
GAN_synthesized_data_COPULA = [f for f in os.listdir() if 'Copula' in f and os.path.isfile(os.path.join(f))]

GAN_method = 'CT'
# GAN_method = 'Copula'

#####################################
# compare default models preformace #
#####################################

if GAN_method == 'CT':
    

    results_GAN_df = pd.DataFrame()
    for syn_data in GAN_synthesized_data:
        fold = int(re.findall(r'\d+',syn_data)[0])
        GAN_train_fold, GAN_test_fold = Functions_ML_Project.GAN_data_preprocessing(syn_data,
                                                                                     training_clean_imputed, IRRELEVANT_FEATURES_for_GAN,
                                                                                     id_fold,OHE_regular_cols,fold,'max',pipeline_GAN)

        model_results = Functions_ML_Project.run_models_with_GAN(GAN_train_fold, GAN_test_fold, models = models_defualt)
        temp_df = pd.DataFrame([model_results])
        results_GAN_df = pd.concat([results_GAN_df, temp_df], ignore_index=True)
    results_GAN_df.loc['mean'] = results_GAN_df.mean()
    #results_GAN_df.to_csv('default_models_CT_GAN_dataframe.csv', index=True)
    
else:
    GAN_synthesized_data_COPULA = [f for f in os.listdir() if 'Copula' in f and os.path.isfile(os.path.join(f))]

    results_GAN_df = pd.DataFrame()
    for syn_data in GAN_synthesized_data_COPULA:
        fold = int(re.findall(r'\d+',syn_data)[0])
        GAN_train_fold, GAN_test_fold = Functions_ML_Project.GAN_data_preprocessing(syn_data,
                                                                                     training_clean_imputed, IRRELEVANT_FEATURES_for_GAN,
                                                                                     id_fold,OHE_regular_cols,fold,'max',pipeline_GAN)

        model_results = Functions_ML_Project.run_models_with_GAN(GAN_train_fold, GAN_test_fold, models = models_defualt)
        temp_df = pd.DataFrame([model_results])
        results_GAN_df = pd.concat([results_GAN_df, temp_df], ignore_index=True)
    results_GAN_df.loc['mean'] = results_GAN_df.mean()
    #results_GAN_df.to_csv('default_models_COPULA_dataframe.csv', index=False)

# --------------------------------------------------------------- #
# run pipeline and defualt models on oversampled data by SMOTE-NC #
# --------------------------------------------------------------- #

training_df_for_smote = training_df_new[training_df_new.readmitted.isin(['<30', 'NO'])]

pre_pipeline_smote = Pipeline([('feature_remover', Class_ML_Project.FeatureRemover(features_to_remove = IRRELEVANT_FEATURES)),
                     ('numerical_scaler',Class_ML_Project.NumericalTransformer(columns=NUMERICAL))])

post_pipeline_smote = Pipeline([('imputer_race', Class_ML_Project.DataFrameImputer(strategy='constant', fill_value='Other', columns = ['race'])),
                     ('imputer_medical', Class_ML_Project.DataFrameImputer(strategy='most_frequent',columns = ['medical_specialty'])),
                     ('age_encoder', Class_ML_Project.MultiColumnLabelEncoder(columns=['age'])),
                     ('OHE', Class_ML_Project.CustomOHEncoder(OHE_regular_cols= OHE_regular_cols, OHE_4_to_2_cols=OHE_4_to_2_cols,
                       change_col='change', diag_cols=diagnoses_cols))])

results_SOMTE_df = pd.DataFrame()
for fold in id_df.fold_num.unique():
    training_post_smote,subset_test = Functions_ML_Project.SMOTE_data_preprocessing(training_df_for_smote,IRRELEVANT_FEATURES,
                                                               id_df,OHE_regular_cols,fold,
                                                               pre_pipeline_smote,post_pipeline_smote)

    model_results = Functions_ML_Project.run_models_with_GAN(training_post_smote, subset_test, models = models_defualt)
    temp_df = pd.DataFrame([model_results])
    results_SOMTE_df = pd.concat([results_SOMTE_df, temp_df], ignore_index=True)

    results_SOMTE_df.loc['mean'] = results_SOMTE_df.mean()
#results_SOMTE_df.to_csv('default_models_smote_dataframe.csv', index=True)
print(results_SOMTE_df)


###################################################################
#         Further analysis running by GAN overasmpled data        #
###################################################################

############################################
# tune and test weight for imbalanced data #
############################################

test_imbalace = False

if test_imbalace: 
    samples_sizes = [0,0.25,0.5,0.75,'max']

    params = {"objective":"binary:logistic", 'max_depth':5,
                  'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8,
                  "seed":42, 'device':'gpu', 'tree_method': 'hist'}

    best_score_after_tuning = []
    baseline_scores = []
    params_tuning = {}

    for oversample in samples_sizes:

        all_resampled_data = []
        fold_inx =[]

        # read and process GAN data from folds 
        for syn_data in GAN_synthesized_data:
            fold = int(re.findall(r'\d+',syn_data)[0])
            GAN_train_fold, GAN_test_fold = Functions_ML_Project.GAN_data_preprocessing(syn_data,
                                                                                         training_clean_imputed, IRRELEVANT_FEATURES_for_GAN,
                                                                                         id_fold,OHE_regular_cols,fold,oversample,pipeline_GAN)
            # change indexes to unique one in each fold for hyperparameter tuning
            if len(all_resampled_data) == 0:
                GAN_train_fold.index = [row for row in range(len(GAN_train_fold))]
                GAN_test_fold.index = [row+len(GAN_train_fold) for row in range(len(GAN_test_fold))]
            else:
                GAN_train_fold.index = [row+len(all_resampled_data) for row in range(len(GAN_train_fold))]
                GAN_test_fold.index = [row+len(GAN_train_fold)+len(all_resampled_data) for row in range(len(GAN_test_fold))]

            print('train index overlap:', GAN_train_fold.index.isin(GAN_test_fold.index).sum(),'\n',
                  'test index overlap:',GAN_test_fold.index.isin(GAN_train_fold.index).sum())

            index_tuple = (GAN_train_fold.index , GAN_test_fold.index)
            fold_inx.append(index_tuple)

            # comnbine all the data together fot hyperparameter tuning 
            if len(all_resampled_data) == 0:
                all_resampled_data = pd.concat([GAN_train_fold, GAN_test_fold],axis = 0)
            else:
                all_resampled_data = pd.concat([all_resampled_data,GAN_train_fold, GAN_test_fold],axis = 0)

        # tune just for size and imbalance weights
        best_value,new_params, baseline_score = Functions_ML_Project.tune_hyperparameters_XGB(objective = Functions_ML_Project.objective_tune_scale_pos_XGB,
                                                                                              base_param = params,
                                                                                             OE_data = all_resampled_data,
                                                                                             fold_index = fold_inx,
                                                                                             n_trials=100)
        best_score_after_tuning.append(best_value)
        baseline_scores.append(baseline_score)
        params_tuning[str(oversample)] = new_params

    # save results 
    logloss_scores_for_size_tuning = pd.DataFrame({"baseline_score" : baseline_scores,
                                                   'post_tuning_score' : best_score_after_tuning}, index = samples_sizes)

    logloss_scores_for_size_tuning.to_csv('sample_size_tuning_scores.csv')
    print('logloss scores for base/tuned weights for imbalanced \nacross ',logloss_scores_for_size_tuning)
#     with open('Sample_size_tuning_parameters.pickle', 'wb') as output_file:
#                     pickle.dump(params_tuning, output_file)


#########################################
#        Predicting with XGBOOST        # 
#########################################


##    tune hyperparameter for XGBOOST     ##
#__________________________________________#

params = {"objective":"binary:logistic", 'max_depth':5,
              'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8,
              "seed":42, 'device':'gpu', 'tree_method': 'hist'}

all_resampled_data = []
fold_inx =[]

    # read and process GAN data from folds 
for syn_data in GAN_synthesized_data:
    fold = int(re.findall(r'\d+',syn_data)[0])
    GAN_train_fold, GAN_test_fold = Functions_ML_Project.GAN_data_preprocessing(syn_data,
                                                                                 training_clean_imputed,
                                                                                IRRELEVANT_FEATURES_for_GAN,
                                                                                 id_fold,OHE_regular_cols,fold,'max',pipeline_GAN)
    # change indexes to unique one in each fold for hyperparameter tuning
    if len(all_resampled_data):
        GAN_train_fold.index = [row for row in range(len(GAN_train_fold))]
        GAN_test_fold.index = [row+len(GAN_train_fold) for row in range(len(GAN_test_fold))]
    else:
        GAN_train_fold.index = [row+len(all_resampled_data) for row in range(len(GAN_train_fold))]
        GAN_test_fold.index = [row+len(GAN_train_fold)+len(all_resampled_data) for row in range(len(GAN_test_fold))]

    print('train index overlap:', GAN_train_fold.index.isin(GAN_test_fold.index).sum(),'\n',
          'test index overlap:',GAN_test_fold.index.isin(GAN_train_fold.index).sum())

    index_tuple = (GAN_train_fold.index , GAN_test_fold.index)
    fold_inx.append(index_tuple)

    # comnbine all the data together fot hyperparameter tuning 
    if len(all_resampled_data) == 0:
        all_resampled_data = pd.concat([GAN_train_fold, GAN_test_fold],axis = 0)
    else:
        all_resampled_data = pd.concat([all_resampled_data,GAN_train_fold, GAN_test_fold],axis = 0)

tune_hyperparameter = False

if tune_hyperparameter:

        tuned_xgboost_score, xgb_best_param, baseline_score = Functions_ML_Project.tune_hyperparameters_XGB(objective = Functions_ML_Project.objective_tune_stage_1,
                                                                                                            base_param = params,
                                                                                                            OE_data = all_resampled_data,
                                                                                                            fold_index = fold_inx,
                                                                                                            n_trials=200,
                                                                                                            tune_steps = 'two')
        # with open('XGBOOST_tuned_parameters.pickle', 'wb') as output_file:
        #                  pickle.dump(tuned_params, output_file)
    
else:
    tuned_params = {'objective': 'binary:logistic',
 'max_depth': 4,
 'subsample': 0.8017681217877746,
 'gamma': 2.430402439508061,
 'colsample_bytree': 0.8301598634480943,
 'learning_rate': 0.1629114048891201,
 'min_child_weight': 7,
 'seed': 42,
 'device': 'gpu',
 'tree_method': 'hist',
 'lambda': 3.394448397873383,
 'alpha': 2.583061480631797,
 'colsample_bynode': 0.9631059074619839,
 'max_delta_step': 4,
 'grow_policy': 'depthwise',
 'sampling_method': 'uniform',
 'colsample_bylevel': 0.7183281335547681,
 'max_leaves': 104,
 'num_estimators': 803}
    

# fit tuned model to all data # 
#_____________________________#

xgb_best_param_for_sklearn = copy.deepcopy(tuned_params)
xgb_best_param_for_sklearn['device'] = 'cpu'
xgb_best_param_for_sklearn['n_estimators'] = xgb_best_param_for_sklearn['num_estimators']
xgb_best_param_for_sklearn.pop('num_estimators')
xgb_best_param_for_sklearn.pop('tree_method')


base_params = {"objective":"binary:logistic", 'max_depth':5,
              'subsample':0.8, 'gamma':0, 'colsample_bytree':0.8,
              "seed":42}

final_xgb = XGBClassifier(use_label_encoder=False,random_state = 42, enable_categorical = True)
final_xgb.set_params(**xgb_best_param_for_sklearn)

base_xgb = XGBClassifier(use_label_encoder=False,random_state = 42, enable_categorical = True)
base_xgb.set_params(**base_params)


X_train = all_resampled_data.drop('readmitted', axis=1)
y_train = all_resampled_data['readmitted']

final_xgb.fit(X_train,y_train)
base_xgb.fit(X_train,y_train)

# read and process test set #
#___________________________#
testing_df_new = Functions_ML_Project.apply_mapping(db_test_df, id_names, mapping_dict)

testing_df_new = Functions_ML_Project.feature_engineering(testing_df_new)

testing_clean_imputed = pipeline.fit_transform(testing_df_new)

testing_clean_imputed = testing_clean_imputed[testing_clean_imputed.readmitted.isin(['<30', 'NO'])]

testing_clean_imputed = testing_clean_imputed.iloc[:,testing_clean_imputed.columns.isin(all_resampled_data.columns)]

missing_columns_test_df = pd.DataFrame(0,index = np.arange(testing_clean_imputed.shape[0]),
                                       columns = all_resampled_data.columns[~all_resampled_data.columns.isin(testing_clean_imputed.columns)])

testing_clean_imputed = pd.concat([testing_clean_imputed.reset_index(drop = True),missing_columns_test_df],axis = 1)

X_test = testing_clean_imputed.drop('readmitted',axis  = 1 )

# order columns the same as in training
X_test = X_test[X_train.columns]

y_test = testing_clean_imputed['readmitted']
y_test = LabelEncoder().fit_transform(y_test)

# calcualte the balanced accuracy
tuned_acc_score = balanced_accuracy_score(y_test,final_xgb.predict(X_test))
base_acc_score = balanced_accuracy_score(y_test,base_xgb.predict(X_test))
print('Balanced accuracy for tuned model: ',round(tuned_acc_score,3))
print('Balanced accuracy for base model: ',round(base_acc_score,3))
print('An improvment of ',round(round(tuned_acc_score,3)-round(base_acc_score,3),3))


# plot ROC-AUC curves    #
#________________________#
# AUC curves
base_pred_prob = base_xgb.predict_proba(X_test)[:, 1]
tuned_pred_prob = final_xgb.predict_proba(X_test)[:, 1]
# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, base_pred_prob)
roc_auc_baseline = auc(fpr, tpr)

fpr_t, tpr_t, _ = roc_curve(y_test, tuned_pred_prob)
roc_auc_tuned = auc(fpr_t, tpr_t)


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Untuned ROC curve (area = %0.2f)' % roc_auc_baseline)
plt.plot(fpr_t, tpr_t, color='darkgreen', lw=2, label='Tuned ROC curve (area = %0.2f)' % roc_auc_tuned)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Speseficity')
plt.ylabel('Sensitivity')
plt.title('GAN Balanced completely\nReceiver Operating Characteristic')
plt.legend(loc="lower right")

# plt.savefig('roc_curve_GAN_fully_balanced.png',format='png', dpi=300,
#            transparent=True)

plt.show()


#################################################
#     tune hyperparameter for RandomForset      #
#################################################

tune_rf = False

if tune_rf:
    tuned_rf_params, best_rf_score,baseline_score_rf =  Functions_ML_Project.tune_hyperparameters_rf(all_resampled_data,
                                                                                                     eval_metric='logloss',
                                                                                                     n_trials=100,folds=fold_inx)
else:
    tuned_rf_params = {'max_depth': 19,
                       'n_estimators': 83,
                       'min_samples_split': 9,
                       'min_samples_leaf': 10,
                       'random_state': 42,
                       'max_features': 'sqrt',
                       'criterion': 'log_loss',
                       'max_leaf_nodes': 248}
    
    
# parameters for base model
base_params = {'max_depth':10,
               "n_estimators":100,
               "min_samples_split":5,
               "min_samples_leaf":3,
               "random_state":42,
              "max_features": 'sqrt',
              "criterion":'log_loss',
              }
    

# Applying best parameters to the RandomForestClassifier
final_rf_tuned = RandomForestClassifier(**tuned_rf_params,n_jobs=-1)
base_rf = RandomForestClassifier(**base_params,n_jobs=-1)

final_rf_tuned.fit(X_train, y_train)
base_rf.fit(X_train, y_train)

# Evaluate on test data
tuned_acc_score_rf = balanced_accuracy_score(y_test, final_rf_tuned.predict(X_test))
base_acc_score_rf = balanced_accuracy_score(y_test, base_rf.predict(X_test))
print("Tuned Accuracy Score:", round(tuned_acc_score_rf,3))
print("Base Accuracy Score:", round(base_acc_score_rf,3))
print('Improvemnt of: ',round((tuned_acc_score_rf - base_acc_score_rf),3))


# plot ROC-AUC for random forest result 

# AUC curves
base_pred_prob = base_rf.predict_proba(X_test)[:, 1]
tuned_pred_prob = final_rf_tuned.predict_proba(X_test)[:, 1]
# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, base_pred_prob)
roc_auc_baseline = auc(fpr, tpr)

fpr_t, tpr_t, _ = roc_curve(y_test, tuned_pred_prob)
roc_auc_tuned = auc(fpr_t, tpr_t)


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Untuned ROC curve (area = %0.2f)' % roc_auc_baseline)
plt.plot(fpr_t, tpr_t, color='darkgreen', lw=2, label='Tuned ROC curve (area = %0.2f)' % roc_auc_tuned)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Speseficity')
plt.ylabel('Sensitivity')
plt.title('GAN Balanced completely Random Forest\nReceiver Operating Characteristic')
plt.legend(loc="lower right")

plt.savefig('roc_curve_GAN_fully_balanced_RF.png',format='png', dpi=300,
           transparent=True)

plt.show()

#################################################
#     tune hyperparameter for RandomForset      #
#################################################

## Since BalancedRandomForestClassifier showed great resuls in the defualt level ##
## We decided to examine its prediction but on a data without Oversampling ##

#for balanced stratification of folds
patient_id = training_df_new.patient_nbr[training_df_new.readmitted.isin(['<30', 'NO'])]

tune_brf = False

if tune_brf:
    tuned_brf_params, best_brf_score,baseline_score_brf =  Functions_ML_Project.tune_hyperparameters_brf(X_train_brf,y_train_brf,5,
                                                                                eval_metric='balanced_accuracy',
                                                                                groupstrat=patient_id,
                                                                                n_trials=100)
else:
    tuned_brf_params = {'n_estimators': 618,
                        'max_depth': 12,
                        'min_samples_split': 10,
                        'min_samples_leaf': 2,
                        'max_leaf_nodes': 1171}


X_train_brf = training_clean_imputed.drop('readmitted', axis =1)
y_train_brf = training_clean_imputed['readmitted']
y_train_brf = LabelEncoder().fit_transform(y_train_brf)


brf_model = BalancedRandomForestClassifier(replacement = True,
                                           bootstrap = True,
                                           random_state=42)

brf_tuned_model = BalancedRandomForestClassifier(**new_brf_param,replacement = True,
                                           bootstrap = True,
                                           random_state=42)

brf_model.fit(X_train_brf,y_train_brf)
brf_tuned_model.fit(X_train_brf,y_train_brf)

base_acc_score_brf = balanced_accuracy_score(y_test, brf_model.predict(X_test))
tuned_acc_score_brf = balanced_accuracy_score(y_test, brf_tuned_model.predict(X_test))
print("Tuned Accuracy Score:", round(tuned_acc_score_brf,3))
print("Base Accuracy Score:", round(base_acc_score_brf,3))
print('Improvemnt of: ',round((tuned_acc_score_brf - base_acc_score_brf),3))


# AUC curves
base_pred_prob = brf_model.predict_proba(X_test)[:, 1]
tuned_pred_prob = brf_tuned_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, base_pred_prob)
roc_auc_baseline = auc(fpr, tpr)

fpr_t, tpr_t, _ = roc_curve(y_test, tuned_pred_prob)
roc_auc_tuned = auc(fpr_t, tpr_t)


# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Untuned ROC curve (area = %0.2f)' % roc_auc_baseline)
plt.plot(fpr_t, tpr_t, color='darkgreen', lw=2, label='Tuned ROC curve (area = %0.2f)' % roc_auc_tuned)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Speseficity')
plt.ylabel('Sensitivity')
plt.title('Balanced Random Forest\nReceiver Operating Characteristic')
plt.legend(loc="lower right")

plt.savefig('roc_curve_unbalanced_BRF.png',format='png', dpi=300,
           transparent=True)

plt.show()

###################################
#  feature importance with SHAP   #
###################################

# Create a SHAP TreeExplainer using the trained model
explainer = shap.TreeExplainer(brf_tuned_model)
# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# dot plot of feature importance
shap.summary_plot(shap_values[:,:,1], X_test, plot_type="dot", max_display=15,plot_size = (18,10),
                 cmap='coolwarm',show = True)
# plt.savefig('BRF_SHAP_dotplot.png',format='png', dpi=300,
#            transparent=True)


# plot specific for age 

age_shap_values = shap_values[:, X_test.columns.get_loc('age'),1]

colors = sns.color_palette("coolwarm", 9)  # Generate a palette with 9 colors from cool to warm
center_color = ["grey"]  # Center color for the category '4'
full_palette = colors[:2] + center_color + colors[3:]

# Creating a dictionary to map each category to the appropriate color
palette = {i: color for i, color in enumerate(full_palette, start=0)}

# Plot the distribution of SHAP values for 'age'
plt.figure(figsize=(10, 6))
age_shap_df = pd.DataFrame({'Age category':age_shap_values.data,
                           'Shap value' : age_shap_values.values})

sns.scatterplot(age_shap_df,x='Shap value',
                y = np.random.normal(size=len(age_shap_df), scale=0.1,loc=1),
                hue='Age category',
              palette = palette)
plt.title('Distribution of SHAP Values for Age')
plt.xlabel('SHAP Value')
plt.ylabel('')  # Remove y-axis label
plt.yticks([])  # Remove y-axis ticks
plt.gca().axes.yaxis.set_visible(False)  # Hide the y-axis

# plt.savefig('BRF_SHAP_dotplot_age.png',format='png', dpi=300,
#            transparent=True)
# plt.close()
plt.show()



# bar plot of absolute calue for feature importance
shap.plots.bar(shap_values[:,:,1], max_display=15,show = True)

# plt.savefig('BRF_SHAP_barplot.png',format='png', dpi=300,
#            transparent=True,bbox_inches='tight')
# plt.close()


#########################################
# plot stability for feature importance #
#########################################

seeds_list = [42,1,0,71,10,111,777,710,911,810,8,4,6,13,1948]
scores_lists = []
feature_importance_impurity = pd.DataFrame(columns=X_test.columns)

for seed in seeds_list:
    brf_tuned_model = BalancedRandomForestClassifier(**tuned_brf_params,replacement = True,
                                               bootstrap = True,
                                               random_state=seed)
    
    brf_tuned_model.fit(X_train_brf,y_train_brf)
    feature_importance_impurity.loc[seed] = brf_tuned_model.feature_importances_
    scores_lists.append(balanced_accuracy_score(y_test, brf_tuned_model.predict(X_test)))

# order by most important feautres 
ordered_cols = feature_importance_impurity.mean().sort_values(ascending=False).index
feature_importance_impurity = feature_importance_impurity[ordered_cols]

# Stability of balanced accuracy across random seeds #
pd.Series(scores_lists, index = feature_importance_impurity.index).plot.bar()
plt.ylim(0,1)
plt.title('Stability of balanced accuracy across random seeds')
plt.ylabel('balanced accuracy')

# plt.savefig('BRF_accuracy_across_seeds.png',format='png', dpi=300,
#            transparent=True,bbox_inches='tight')

plt.show()

# Stability of Feature importance across random seeds #
feature_importance_impurity.std().plot.bar()
ax = plt.gca()
plt.title('Stability of Feature importance across random seeds')
plt.ylabel('STD across 15 pseudoseeds')  # Remove y-axis label
ax.set_xticks(ax.get_xticks()[::2])

# plt.savefig('BRF_Feature_importance_stability.png',format='png', dpi=300,
#            transparent=True,bbox_inches='tight')

plt.show()

# top 10 most iomportat feautres box plot #
sns.boxplot(data = pd.melt(feature_importance_impurity.iloc[:,:10]),
            x = 'variable', y = 'value',
            hue = 'variable', legend = False)
plt.xticks(rotation=90)

plt.title('Feature importance across random seeds')
plt.ylabel('mean decrease in impurity across 15 pseudoseeds')  # Remove y-axis label

# plt.savefig('BRF_top_10_Feature_importance.png',format='png', dpi=300,
#            transparent=True,bbox_inches='tight')

plt.show()

#########################################
#        Predicting with LightGBM        # 
#########################################

### Hyperparameter tuning with Optuna ###

# Comment out this junk of code when finished tuning and use lgb_params_best instead of lgb_params

# def objective(trial):
#     lgb_params = {# "device_type": trial.suggest_categorical("device_type", ['gpu']),
#                     "num_estimators": trial.suggest_categorical("num_estimators", [1000,10000]), #10000
#                     "learning_rate": trial.suggest_float("learning_rate", 0.3, 0.4,step = 0.01),
#                     "num_leaves": trial.suggest_int("num_leaves", 200, 270, step=5), #default 31 1000
#                     #"max_depth": trial.suggest_int("max_depth", 3, 12), #defalt -1
#                     "min_data_in_leaf":200,
#                     #"min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 400, step=50),
#                     "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=10),
#                     "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=10),
#                     "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.2, 0.3,step=0.001),
#                     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.9, step=0.1),
#                     "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
#                     "feature_fraction": trial.suggest_float("feature_fraction", 0.75, 0.9, step=0.01),
#                     "random_state": 42
#                      }

    # model = lgb.LGBMClassifier(**lgb_params)

    # score_list = list()

    # for fold in range(1,6):
    #     File = open(f'training_validation_clean_imputed_{fold}.pkl', 'rb')
    #     training_clean_imputed, val_clean_imputed = pickle.load(File)

    #     # Fit and transform the DataFrame
    #     # training_clean_imputed = pipeline.fit_transform(train)
    #     X_train = training_clean_imputed.drop('readmitted', axis=1)
    #     y_train = training_clean_imputed['readmitted']
    #     y_train = LabelEncoder().fit_transform(y_train)
    #     # val = pd.read_csv(f'validation_fold_{fold}.csv')
    #     # val = val[list(set(val.columns).intersection(set(train.columns)))]
    #     # val_clean_imputed = pipeline.fit_transform(val)
    #     X_val = val_clean_imputed.drop('readmitted', axis=1)
    #     y_val = val_clean_imputed['readmitted']
    #     y_val = LabelEncoder().fit_transform(y_val)
    #     cl_list = list(set(X_val.columns).intersection(set(X_train.columns))) #QC
    #     X_train = X_train[cl_list] #same order
    #     X_val = X_val[cl_list]
    #     model.fit(X_train, y_train, eval_set=(X_val, y_val))
    #     y_pred = model.predict(X_val)
    #     score_list.append(balanced_accuracy_score(y_val, y_pred))
    #     #File = open(f'training_validation_clean_imputed_{fold}.pkl', 'wb')
    #     #pickle.dump((training_clean_imputed, val_clean_imputed), File)
    # return sum(score_list) / len(score_list),  #Mean balanced accuracy

## Optuna study (commented out when finished tuning)
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000, timeout=400)
# print('Best hyperparameters:', study.best_params)
# print('Best balance accuracy:', study.best_value)

## Save optuna output for GPU purposes to see reaults
# f = open("LGBM_hyperparameters_.txt","a") #"w" check for append
# f.write(f'Best params:{study.best_params}/n/n'
#         f'Mean CV score (balanced_accuracy)::{study.best_value}/n/n')
# f.close
### Visualizing Optuna results ###
#Plot optimization history of all trials in a study
# fig_optuna = optuna.visualization.plot_optimization_history(study)
# fig_optuna.show()

# fig_importance = optuna.visualization.plot_param_importances(
#     study, target=lambda t: t.duration.total_seconds(), target_name="duration")
# fig_importance.show()

### Test cohort ###
testing_df_new = pd.read_csv("test_cohort.csv")
all_resampled_data = pd.read_csv("all_resampled_data.csv")

# Preprocessing Pipline on test set
id_names, mapping_dict = Functions_ML_Project.preform_ids_maping('IDS_mapping.csv')

testing_df_new = Functions_ML_Project.apply_mapping(testing_df_new, id_names, mapping_dict)
testing_df_new = Functions_ML_Project.feature_engineering(testing_df_new)
testing_clean_imputed = pipeline.fit_transform(testing_df_new)
testing_clean_imputed = testing_clean_imputed[testing_clean_imputed.readmitted.isin(['<30', 'NO'])]
testing_clean_imputed = testing_clean_imputed.iloc[:,testing_clean_imputed.columns.isin(all_resampled_data.columns)]
missing_columns_test_df = pd.DataFrame(0,index = np.arange(testing_clean_imputed.shape[0]),
                                       columns = all_resampled_data.columns[~all_resampled_data.columns.isin(testing_clean_imputed.columns)])
testing_clean_imputed = pd.concat([testing_clean_imputed.reset_index(drop = True),missing_columns_test_df],axis = 1)

X_test = testing_clean_imputed.drop('readmitted',axis  = 1)
y_test = testing_clean_imputed['readmitted']
y_test = LabelEncoder().fit_transform(y_test)

# Order columns the same as in training
X_train = all_resampled_data.drop('readmitted', axis=1)
X_test = X_test[X_train.columns] 
y_train = all_resampled_data['readmitted']
y_train = LabelEncoder().fit_transform(y_train)

# Best params based on Optuna objective
lgb_params_best = {"num_estimators": 1000, #10000
                    "learning_rate": 0.31,
                    "num_leaves": 230,
                    #"max_depth": trial.suggest_int("max_depth", 3, 12), #defalt -1
                    # "min_data_in_leaf":200,
                    "lambda_l1": 10,
                    "lambda_l2": 80,
                    "min_gain_to_split":0.2380984754136244,
                    "bagging_fraction": 0.7,
                    "bagging_freq": 1,
                    "feature_fraction": 0.7999999999999999,
                    "random_state": 42
                     }
#According to Ortal
base_params = {"objective":"binary",'max_depth':5,
              'subsample':0.8, 'colsample_bytree':0.8,
              "seed":42}

final_lgb = lgb.LGBMClassifier()
final_lgb.set_params(**lgb_params_best)

base_lgb = lgb.LGBMClassifier(random_state = 42)
base_lgb.set_params(**base_params)

final_lgb.fit(X_train,y_train)
base_lgb.fit(X_train,y_train)

# Calcualte the balanced accuracy LGBM
tuned_acc_score = balanced_accuracy_score(y_test,final_lgb.predict(X_test))
base_acc_score = balanced_accuracy_score(y_test,base_lgb.predict(X_test))
print('Balanced accuracy for tuned model: ',round(tuned_acc_score,3))
print('Balanced accuracy for base model: ',round(base_acc_score,3))
print('An improvment of ',round(round(tuned_acc_score,3)-round(base_acc_score,3),3))

# plot ROC-AUC curves    #
#________________________#
# AUC curves
base_pred_prob = base_lgb.predict_proba(X_test)[:, 1]
tuned_pred_prob = final_lgb.predict_proba(X_test)[:, 1]
# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, base_pred_prob)
roc_auc_baseline = auc(fpr, tpr)

fpr_t, tpr_t, _ = roc_curve(y_test, tuned_pred_prob)
roc_auc_tuned = auc(fpr_t, tpr_t)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Untuned ROC curve (area = %0.2f)' % roc_auc_baseline)
plt.plot(fpr_t, tpr_t, color='darkgreen', lw=2, label='Tuned ROC curve (area = %0.2f)' % roc_auc_tuned)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Speseficity')
plt.ylabel('Sensitivity')
plt.title('LGBM - GAN Balanced completely\nReceiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve_GAN_fully_balanced.png',format='png', dpi=300,
           transparent=True)
#plt.show()
plt.close()

### Barplot balanced accuracy ###
# Labels for the conditions
conditions = ['Untuned', 'Tuned']

# Values for the conditions
values = [base_acc_score, tuned_acc_score]

plt.rcParams.update({'font.size': 14})

# Plotting the bar plot using Seaborn
sns.barplot(x=conditions, y=values, palette=['darkorange', 'darkgreen'])

# Adding labels and title
plt.ylim(0.3, 0.7)
plt.xlabel('Condition')
plt.ylabel('Balanced Accuracy')
plt.title('LGBM - Tuned vs. Untuned')
plt.tight_layout()
# Display the plot
#plt.show()




