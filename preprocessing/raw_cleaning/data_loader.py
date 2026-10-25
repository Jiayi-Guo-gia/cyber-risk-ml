import psutil
import argparse
import time
import pandas as pd

parser = argparse.ArgumentParser(description="set arguments")
parser.add_argument("--source_file", type=str, help="source file path, json type")
parser.add_argument("--output_file", type=str, help="source file path, csv type")
args = parser.parse_args()

if __name__ == "__main__" :
    
    SOURCE_FILE = args.source_file
    OUTPUT_FILE = args.output_file
    
    # initial RAM usage
    print(
        "Initial RAM usage:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    start_time = time.time()

    print("Reading data ...\n")
    raw_data = pd.read_json(SOURCE_FILE)
    
    # initial RAM usage
    print(
        "RAM usage after load data:\nUsed RAM: {:.4f} GB, Total RAM: {:.4f} GB.\n".format(
            psutil.virtual_memory().used / (2**30),
            psutil.virtual_memory().total / (2**30),
        )
    )

    print("Write data to csv ...\n")
    raw_data.to_csv(OUTPUT_FILE, index = False)

    end_time = time.time()

    print("Data Loader takes: {:.4f} seconds\n".format(end_time - start_time))