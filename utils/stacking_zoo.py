#import os
#os.environ['PATH'] = pathx + ';' + os.environ['PATH']

from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

lrcParams = {'solver':'liblinear',
             'random_state':736283,
             'C':10}

rfcParams = {'n_estimators':400,
             'max_features':200,
             'min_samples_leaf':1,
             'max_depth':None,
             'random_state':736283,
             'n_jobs':-1}

etcParams = {'n_estimators':400, 
             'random_state':736283, 
             'max_depth':None, 
             'max_features':200, 
             'min_samples_split':2, 
             'n_jobs':-1}

xgbParams = {'learning_rate':0.01, 
             'n_estimators':100, 
             'max_depth':10, 
             'min_child_weight':1,
             'gamma':0,
             'subsample':0.7,
             'colsample_bytree':0.6,
             'scale_pos_weight':1,
             'random_state':736283, 'n_jobs':-1}

lgbmParams = {'boosting_type':'gbdt',
              'random_state':736283,
              'objective':'binary',
              'metric':'auc',
              'max_bin':255,
              'num_leaves':200,
              'learning_rate':0.1,
              'tree_learner':'feature',
              'n_estimators':100,
              'n_jobs':-1,
              'verbosity':-1,
              'reg_lambda':0.001,
              'feature_fraction':0.9
}

def build_StackingModelCV(cvRun=5):

    lrcModel = LogisticRegression(**lrcParams)
    xgbModel = XGBClassifier(**xgbParams)
    etcModel = ExtraTreesClassifier(**etcParams)
    rfcModel = RandomForestClassifier(**rfcParams)
    lgbmModel = LGBMClassifier(**lgbmParams)
    
    metaModel = LogisticRegression(solver='liblinear')

    stk_classifier = StackingCVClassifier(classifiers=[xgbModel, rfcModel, lgbmModel], meta_classifier=metaModel, cv=cvRun)
    
    return stk_classifier