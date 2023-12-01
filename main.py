from train_ml_model import train_model, classification_scores, preprocess
from helpers import remove_non_csv_files, combine_streaming_data, delete_individual_chunks, calculate_fraud_instances, load_model
import argparse, os, joblib, time
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='absolute path to dataset')
    parser.add_argument('--to_save_path', type=str, required=True, help='absolute folder path where to save the model')
    parser.add_argument('--streaming_data_path', type=str, required=True, help='absolute folder path to where new data will stream. \
        Note: Kindly make sure that this folder does not contain any other files, not related to and compatible with the dataset (except the ones created by the program itself)')
    
    args = parser.parse_args()
    
    if not args.data_path or not args.to_save_path:
        raise("path is not provided")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The file '{args.data_path}' does not exist.")
    if not os.path.exists(args.streaming_data_path):
        raise FileNotFoundError(f"The directory '{args.streaming_data_path}' does not exist.")
    if not os.path.exists(args.to_save_path):
        print(f"The directory '{args.to_save_path}' does not exist.")
        print("Creating the directory - {}".format(args.to_save_path))
        os.makedirs(args.to_save_path)
        print("Directory created")
        print("==> Training the machine learning model")
        train_model(data_path=args.data_path, to_save_path=args.to_save_path)
        #raise FileNotFoundError(f"The directory '{args.to_save_path}' does not exist.")
    
    if not os.path.exists("{}/classifier.joblib".format(args.to_save_path)):
        print("No trained machine learning exists")
        print("==> Training the machine learning model")
        train_model(data_path=args.data_path, to_save_path=args.to_save_path)
        
    #preset values
    fraud_instances_threshold = 20
    recall_score_threshold = 0.75
    
    #make sure streaming data directory is fine
    # to check - whether it's empty, all csv, contains new_data.csv or not, if not then create; or also maybe if all the csv file are
    # compatible with the dataset
    
    # directory_contents = os.listdir(args.streaming_data_path)
    # #need to improve this section by including cases for directories and other possible errors
    # if len(directory_contents) == 0:
    #     #directory empty; create new_data.csv
    #     new_data = pd.DataFrame
    
    ml_model, normalizer = load_model(args.to_save_path)
    
    try:
        while(True):
            directory_contents = os.listdir(args.streaming_data_path)
            num_files = len(directory_contents)
            if num_files > 1 or (num_files == 1 and 'new_data.csv' not in directory_contents):
                #keeping only csv files
                directory_contents = remove_non_csv_files(directory_contents)

                combined_data = combine_streaming_data(args.streaming_data_path, directory_contents)
                combined_data.to_csv("{}/new_data.csv".format(args.streaming_data_path), index=False)
                print("Combined and appended all the chunks of streamed data into new_data.csv")
                delete_individual_chunks(args.streaming_data_path, directory_contents)
                
                new_data = combined_data
                
                fraud_instances = calculate_fraud_instances(new_data)
                
                if fraud_instances > fraud_instances_threshold:
                    processed_new_data, y_true = preprocess(new_data, normalizer)
                    recall_score = classification_scores("current_model-new_data", y_true, ml_model.predict(processed_new_data))["recall_score"]
                    
                    if recall_score > recall_score_threshold:
                        continue
                    else:
                        
 
            else:
                check performance on that
                if performance greather than threshold:
                    continue
                else:
                    merge new_data.csv with our dataset
                    trigger train ml pipeline
                    ?? if new model is not used, should we add new_data to our dataset or not 
                    (if we don't it back then increase the threshold by step size)
                    ?? should I test the model on test split or new_data because poor performance on new_data
                    triggered the training pipeline so the new model should be tested on that data only
            time.sleep(3)
    except KeyboardInterrupt:
        # Saving the dataframe as csv file
        print("\nMonitoring interrupted. Saving the data.")

                
                
    
    
    
    
    