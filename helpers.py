import os
import pandas as pd
import joblib
def remove_non_csv_files(directory_contents):
    contains_non_csv_files = False
    for file in directory_contents:
        if file[-3:] != 'csv':
            contains_non_csv_files = True
            directory_contents.remove(file)
    
    if contains_non_csv_files:
        print("WARNING: Please remove all the irrelevant files and directories from the streaming_data_path")
        print("Excluded all the non-csv files from processing")
    return directory_contents

def combine_streaming_data(path, directory_contents):
    datasets = []
    for file_name in directory_contents:
        datasets.append(pd.read_csv("{}/{}".format(path,file_name)))
    
    return pd.concat(datasets, axis=0, ignore_index=True)

def delete_individual_chunks(path, directory_contents):
    directory_contents.remove("new_data.csv")
    for file_name in directory_contents:
        os.remove("{}/{}".format(path,file_name))
    print("Deleted individual chunks of streamed data")

def calculate_fraud_instances(dataset):
    return sum(dataset["Class"] == 1)

def load_model(model_path):
    return joblib.load('{}/classifier.joblib'.format(model_path)), joblib.load('{}/normalizer.joblib'.format(model_path))