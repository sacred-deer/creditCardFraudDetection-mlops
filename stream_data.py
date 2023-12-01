import argparse, os, time, math, random, string
import pandas as pd

def get_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters, k=length))

if __name__ == '__main__':
    debug = False
    if debug:
        print("<<<< PLEASE NOTE THAT DEBUG MODE IS ON >>>>")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='absolute path to source dataset')
    parser.add_argument('--streaming_data_path', type=str, required=True, help='absolute folder path to where new data will stream')
    
    args = parser.parse_args()
    
    if not args.data_path:
        raise("data path is not provided")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"The file '{args.data_path}' does not exist.")
    if not os.path.exists(args.streaming_data_path):
        raise FileNotFoundError(f"The directory '{args.streaming_data_path}' does not exist.")
    
    print("===> Reading Source Data")
    source_data = pd.read_csv(args.data_path)
    
    total_normal, total_fraud = source_data["Class"].value_counts()
    
    cases_fraud = source_data[source_data["Class"] == 1]
    cases_normal = source_data[source_data["Class"] == 0]
    
    cases_fraud = cases_fraud.sample(frac=1, axis=0, ignore_index=True)
    cases_normal = cases_normal.sample(frac=1, axis=0, ignore_index=True)
    
    #manually set values
    step_size_fraud = 20
    wait_period = 30
    
    if debug:
        wait_period = 3
    
    step_size_normal = math.ceil(total_normal/(math.ceil(float(total_fraud)/step_size_fraud)))
    
    startIndex_fraud = 0
    startIndex_normal = 0
    
    while(startIndex_fraud < total_fraud and startIndex_normal < total_normal):
        
        if(startIndex_fraud+step_size_fraud <= total_fraud and startIndex_normal+step_size_normal <= total_normal):
            chunk_fraud = cases_fraud.iloc[startIndex_fraud:startIndex_fraud+step_size_fraud, :]
            chunk_normal = cases_normal.iloc[startIndex_normal:startIndex_normal+step_size_normal, :]
            chunk = pd.concat([chunk_fraud, chunk_normal], axis=0, ignore_index=True).sample(frac=1, axis=0, ignore_index=True)
            chunk.to_csv("{}/verifiedTransactions_{}.csv".format(args.streaming_data_path, get_random_string(10)),index=False)
            
        else:
            chunk_fraud = cases_fraud.iloc[startIndex_fraud:, :]
            chunk_normal = cases_normal.iloc[startIndex_normal:, :]
            chunk = pd.concat([chunk_fraud, chunk_normal], axis=0, ignore_index=True).sample(frac=1, axis=0, ignore_index=True)
            chunk.to_csv("{}/verifiedTransactions_{}.csv".format(args.streaming_data_path, get_random_string(10)),index=False)
        
        print("streamed a transaction (startIndex_fraud: {})".format(startIndex_fraud))
        time.sleep(wait_period)
        startIndex_fraud = startIndex_fraud + step_size_fraud
        startIndex_normal = startIndex_normal + step_size_normal
    