import os
import argparse
import kagglehub
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def run(args):
    data_path = args.data_path
    save_path = args.save_path
    prefix_path = args.prefix_path
    print("Starting data download ....")
    kagglehub.dataset_download("prathmeshzade/iiit5k-words", output_dir=data_path)
    print("Starting data preprocessing ....")


    trainvaldata = pd.read_csv(f'{data_path}/trainCharBound.csv')
    testdata = pd.read_csv(f'{data_path}/testCharBound.csv')

    traindata, valdata = train_test_split(trainvaldata,test_size=0.1,random_state=42,shuffle=True)

    traindata.reset_index(inplace=True)
    valdata.reset_index(inplace=True)
    testdata.reset_index(inplace=True)

    traindata.drop(columns=['charBB', 'index'], inplace=True)
    valdata.drop(columns=['charBB', 'index'], inplace=True)
    testdata.drop(columns=['charBB', 'index'], inplace=True)

    
    traindata['ImgName'] = prefix_path + '/' + traindata['ImgName']
    valdata['ImgName'] = prefix_path + '/' + valdata['ImgName']
    testdata['ImgName'] = prefix_path + '/' + testdata['ImgName']



    alltexts = list(traindata['chars'])

    vocab_set = set(''.join(alltexts))
    vocab_set.add('j')
    vocab = sorted(vocab_set)

    ctoi = {c: i+1 for i, c in enumerate(vocab)}
    itoc = {i: c for c, i in ctoi.items()}

    os.makedirs(save_path, exist_ok=True)

    traindata.to_csv(f'{save_path}/train.csv', index=False)
    valdata.to_csv(f'{save_path}/val.csv', index=False)
    testdata.to_csv(f'{save_path}/test.csv', index=False)



    data = {
        "vocab": vocab,
        "ctoi": ctoi,
        "itoc": itoc
    }

    with open(f"{save_path}/vocab.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--data_path", type=str, default="data/iiit5k-words")
    parser.add_argument("--save_path", type=str, default="data/processed")
    parser.add_argument("--prefix_path", type=str, default="data/iiit5k-words/IIIT5K-Word_V3.0/IIIT5K")

    args = parser.parse_args()
    
    run(args=args)


if __name__ == "__main__":
    main()