import argparse, os, re, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score, average_precision_score, precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn import ensemble, naive_bayes
from xgboost import XGBClassifier

debug = False

def classification_scores(model_name, y_true, y_pred):
    scores = {}
    scores["recall_score"] = recall_score(y_true, y_pred)
    scores["f1_score"] = f1_score(y_true, y_pred)
    scores["balanced_accuracy_score"] = balanced_accuracy_score(y_true, y_pred)
    scores['roc_auc_score'] = roc_auc_score(y_true, y_pred) 
    scores['average_precision_score'] = average_precision_score(y_true, y_pred)
    
    print("'Model': '{}', 'recall_score': {:.2f}, 'f1_score': {:.2f}, 'balanced_accuracy_score': {:.2f}, 'roc_auc_score': {:.2f}, 'average_precision_score': {:.2f}".format(model_name,\
        recall_score(y_true, y_pred), f1_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred),\
            roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)))
    
    return scores

def get_last_version(model_name):
    int(re.search(r'v(\d+)', model_name).group(1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='absolute path to dataset')
    parser.add_argument('--to_save_path', type=str, required=True, help='absolute path where to save the model')
    
    args = parser.parse_args()
    
    if not args.data_path or not args.to_save_path:
        raise("path is not provided")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The file '{args.data_path}' does not exist.")
    if not os.path.exists(args.to_save_path):
        raise FileNotFoundError(f"The directory '{args.to_save_path}' does not exist.")
    
    print("===> Reading the dataset")
    creditCard = pd.read_csv(args.data_path)
    
    # Data cleaning
    print("===> Performing data cleaning")
    # Removing missing values
    creditCard.dropna(inplace=True)
    
    #dropping time/id column
    to_drop_cols = ["Time", "id"]
    for drop_col in to_drop_cols:
        if drop_col in creditCard.columns:
            creditCard.drop(columns=[drop_col], inplace=True)
    Y = creditCard["Class"]
    creditCard.drop(columns=["Class"], inplace=True)
    
    # Peforming normalization
    print("===> Performing data normalization")
    scaler = StandardScaler()
    creditCard_normalized = scaler.fit_transform(creditCard)
    
    print("===> Splitting the dataset into train and test; Validating model performance.")
    X_train, X_test, y_train, y_test = train_test_split(creditCard_normalized, Y, test_size=0.8, stratify=Y)
    
    #VotingClassifier
    #classifiers
    #Validation
    clf1 = XGBClassifier(eval_metric = recall_score)
    clf2 = ensemble.RandomForestClassifier()
    clf3 = naive_bayes.GaussianNB()
    voting_classifier = ensemble.VotingClassifier(estimators=[
        ('xgb', clf1), ('random_forest', clf2), ('naive_bayes', clf3)], verbose=2, n_jobs = -1)
    voting_classifier = voting_classifier.fit(X_train, y_train)
    classification_scores("voting_classifier-train", y_train, voting_classifier.predict(X_train))
    classification_scores("voting_classifier-test",y_test, voting_classifier.predict(X_test))
    
    #training model on the whole dataset
    print("===> Training the model on the whole dataset")
    if debug:
        classifier = voting_classifier
    else:
        clf1 = XGBClassifier(eval_metric = recall_score)
        clf2 = ensemble.RandomForestClassifier()
        clf3 = naive_bayes.GaussianNB()
        classifier = ensemble.VotingClassifier(estimators=[
            ('xgb', clf1), ('random_forest', clf2), ('naive_bayes', clf3)], verbose=2, n_jobs = -1)
        
        classifier = classifier.fit(creditCard_normalized, Y)
    
    model_performance = classification_scores("[performance-on-the-whole-dataset] voting_classifier-train", Y, classifier.predict(creditCard_normalized))
    peformance_log_file_path = "{}/model_performance_log.csv".format(args.to_save_path)
    
    if not os.path.exists(peformance_log_file_path):
        performance_log = pd.DataFrame(columns = ["model_name"] + list(model_performance))
        model_performance["model_name"] = "voting_classifier_v1"
    else:
        performance_log = pd.read_csv(peformance_log_file_path)
        last_version = get_last_version(performance_log["model_name"].iloc[-1])
        if debug:
            print(performance_log["model_name"].iloc[-1])
            print("last version: ", last_version)
        model_performance["model_name"] = "voting_classifier_v" + str(last_version+1)
    
    print("===> Saving the ML model, StandardScaler and updating the log file.")
    joblib.dump(classifier, '{}/{}.joblib'.format(args.to_save_path, model_performance["model_name"]))
    if debug:
        print('ML model saved at {}/{}.joblib'.format(args.to_save_path, model_performance["model_name"]))
    joblib.dump(scaler, '{}/{}_normalizer.joblib'.format(args.to_save_path, model_performance["model_name"]))
    if debug:
        print('Scaler saved at {}/{}_normalizer.joblib'.format(args.to_save_path, model_performance["model_name"]))
    performance_log = pd.concat([performance_log, pd.DataFrame([model_performance])])
    performance_log.to_csv('{}/model_performance_log.csv'.format(args.to_save_path), index=False)
    if debug:
        print('log updated at {}/model_performance_log.csv'.format(args.to_save_path))
    
    
    
    
    
    
    