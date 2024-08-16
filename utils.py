import os
import pandas as pd

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"creating path {path}")
        return
    print(f"check path exists {path}")

def eva_sim(data_path):
    df = pd.read_csv(data_path)
