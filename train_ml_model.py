import os, re, joblib, datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score, average_precision_score, precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn import ensemble, naive_bayes
from xgboost import XGBClassifier

debug = True

def preprocess(data, normalizer):
    # Data cleaning
    print("===> Performing data cleaning")
    # Removing missing values
    preprocessed_data = data.dropna(inplace=False)
    
    #dropping time/id column
    to_drop_cols = ["Time", "id"]
    for drop_col in to_drop_cols:
        if drop_col in preprocessed_data.columns:
            preprocessed_data.drop(columns=[drop_col], inplace=True)
    class_col = preprocessed_data["Class"]
    preprocessed_data.drop(columns=["Class"], inplace=True)
    
    # Peforming normalization
    print("===> Performing data normalization")
    preprocessed_data = normalizer.tranform(preprocessed_data)
    
    return preprocessed_data, class_col

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

def train_model(dataset, to_save_path):

    if debug:
        print("<<<< PLEASE NOTE THAT DEBUG MODE IS ON >>>>")

    if not os.path.exists(to_save_path):
        raise FileNotFoundError(f"The directory '{to_save_path}' does not exist.")
    
    print("===> Reading the dataset")
    creditCard = dataset
    
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
    scaler = scaler.fit(creditCard)
    
    print("===> Splitting the dataset into train and test; Validating model performance.")
    if debug:
        X_train, X_test, y_train, y_test = train_test_split(creditCard, Y, test_size=0.8, stratify=Y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(creditCard, Y, test_size=0.2, stratify=Y)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #VotingClassifier
    #classifiers
    #Validation
    clf1 = XGBClassifier(eval_metric = recall_score)
    clf2 = ensemble.RandomForestClassifier()
    clf3 = naive_bayes.GaussianNB()
    testing_classifier = ensemble.VotingClassifier(estimators=[
        ('xgb', clf1), ('random_forest', clf2), ('naive_bayes', clf3)], verbose=2, n_jobs = -1)
    testing_classifier = testing_classifier.fit(X_train, y_train)
    train_scores = classification_scores("testing_classifier-train", y_train, testing_classifier.predict(X_train))
    test_scores = classification_scores("testing_classifier-test",y_test, testing_classifier.predict(X_test))

    #training model on the whole dataset
    print("===> Training the model on the whole dataset")
    if debug:
        classifier = testing_classifier
    else:
        clf1 = XGBClassifier(eval_metric = recall_score)
        clf2 = ensemble.RandomForestClassifier()
        clf3 = naive_bayes.GaussianNB()
        classifier = ensemble.VotingClassifier(estimators=[
            ('xgb', clf1), ('random_forest', clf2), ('naive_bayes', clf3)], verbose=2, n_jobs = -1)
        
        creditCard_normalized = scaler.transform(creditCard)
        classifier = classifier.fit(creditCard_normalized, Y)
    
    model_performance = classification_scores("[performance-on-the-whole-dataset] classifier-train", Y, classifier.predict(creditCard_normalized))
    peformance_log_file_path = "{}/model_performance_log.csv".format(to_save_path)
    
    if not os.path.exists(peformance_log_file_path):
        performance_log = pd.DataFrame(columns = ["time"] + list(model_performance))
    else:
        performance_log = pd.read_csv(peformance_log_file_path)
    
    model_performance["time"] = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    
    print("===> Saving the ML model, StandardScaler and updating the log file.")
    
    joblib.dump(classifier, '{}/classifier.joblib'.format(to_save_path))
    if debug:
        print('ML model saved at {}/classifier.joblib'.format(to_save_path))
    
    joblib.dump(scaler, '{}/normalizer.joblib'.format(to_save_path))
    if debug:
        print('Scaler saved at {}/normalizer.joblib'.format(to_save_path))
    
    performance_log = pd.concat([performance_log, pd.DataFrame([model_performance])])
    performance_log.to_csv('{}/model_performance_log.csv'.format(to_save_path), index=False)
    if debug:
        print('log updated at {}/model_performance_log.csv'.format(to_save_path))
    
    
    
    
    
    