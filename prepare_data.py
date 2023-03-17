# __author__ == ChiWun Yang
# email == christiannyang37@gmail.com

import os
import datasets
import pandas as pd
from config import task, src_tgt, train_data_path, val_data_path, num_data


def main():
    print(f"Downloading {task.upper()} Dataset {src_tgt.upper()}")
    wmt_data = datasets.load_dataset(task, src_tgt)
    train_df = pd.DataFrame(wmt_data['train']).iloc[:num_data]
    val_df = pd.DataFrame(wmt_data['validation'])

    folder_path = '/'.join(train_data_path.split('/')[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)


if __name__ == '__main__':
    main()
