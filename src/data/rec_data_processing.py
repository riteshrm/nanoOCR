import os
import argparse
import kagglehub
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import subprocess
import random
from pathlib import Path
import shutil
import time


def merge(df, save_path,csv_name, data_type):
    type_csv_path = f'{save_path}/{data_type}_{csv_name}.csv'
    df.to_csv(type_csv_path, index=False)
    csv_path = f'{save_path}/{csv_name}.csv'
    if os.path.exists(csv_path):
        prev_df = pd.read_csv(csv_path)
        df = pd.concat([df, prev_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    
def save_vocab(vocab_path, alltexts):
    pre_vocab = None
    if os.path.exists(vocab_path):
        pre_vocab = set(torch.load(vocab_path)['vocab'])

    vocab_set = set(''.join(alltexts))
    vocab_set.add('j')
    if pre_vocab is not None:
        vocab_set = vocab_set.union(pre_vocab)
    vocab = sorted(vocab_set)

    ctoi = {c: i+1 for i, c in enumerate(vocab)}
    itoc = {i: c for c, i in ctoi.items()}
    torch.save({
            "vocab": vocab,
            "ctoi": ctoi,
            "itoc": itoc
        }, vocab_path)


def mjsynth(args):
    if args.download_data=="True":
        print("Starting data download mjsynth....")
        url = "https://thor.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz"

        subprocess.run([
            "aria2c",
            "-x", "16",             # Maximum connections per server
            "-s", "16",             # Split download into 16 segments
            "-k", "1M",             # Minimum segment size
            "-c",                  # Resume interrupted downloads
            url,
            "-o", "mjsynth.tar.gz"
        ], check=True)

    start = time.perf_counter()


    print("Starting data preprocessing mjsynth....")
    result = subprocess.run(
        ["tar", "-tzf", "mjsynth.tar.gz"],
        capture_output=True,
        text=True,
        check=True
    )

    files = [
        f for f in result.stdout.splitlines()
        if f.lower().endswith((".jpg", ".png"))
    ]

    selected = random.Random(42).sample(files, args.num_samples)

    with open("selected_files_mjsynth.txt", "w") as f:
        f.write("\n".join(selected) + "\n")

    os.makedirs(os.path.join(args.data_path,"mjsynth_subset"), exist_ok=True)

    subprocess.run([
        "tar",
        "-xzf", "mjsynth.tar.gz",
        "-C", os.path.join(args.data_path,"mjsynth_subset"),
        "-T", "selected_files_mjsynth.txt"
    ], check=True)



    src = Path(f"{args.data_path}/mjsynth_subset/mnt/ramdisk/max/90kDICT32px")
    dst = Path(f"{args.data_path}/mjsynth_subset/images")
    dst.mkdir(parents=True, exist_ok=True)

    for img_path in src.rglob("*.jpg"):
        shutil.move(str(img_path), str(dst / img_path.name))

    # Remove original nested directory structure
    try:
        shutil.rmtree(f"{args.data_path}/mjsynth_subset/mnt")
    except:
        pass

    image_paths = []
    image_texts = []

    data_folder = f"{args.data_path}/mjsynth_subset/images"
    for path in os.listdir(data_folder):
        image_paths.append(data_folder + "/" + path)
        image_texts.append(path.split("_")[1])

    df = pd.DataFrame({
            "ImgName": image_paths,
            "chars": image_texts
        })

    traindata, val_test = train_test_split(df,test_size=0.1,random_state=42,shuffle=True)
    valdata, testdata = train_test_split(val_test,test_size=0.5,random_state=42,shuffle=True)
    os.makedirs(args.save_path, exist_ok=True)
    testdata.to_csv(f"{args.save_path}/mjsynth_test.csv", index=False)
    merge(traindata, args.save_path, 'train', 'mjsynth')
    merge(valdata, args.save_path, 'val', 'mjsynth')
    alltexts = list(traindata['chars'])
    
    vocab_path = f"{args.save_path}/vocab.pt"
    save_vocab(vocab_path, alltexts)
    elapsed = time.perf_counter() - start
    print(f"Time taken: {elapsed / 60:.2f} minutes")

def iiit5k(args):
    data_path = args.data_path+"/iiit5k-words"
    save_path = args.save_path
    prefix_path = args.prefix_path
    print("Starting data download IIIT5K....")
    kagglehub.dataset_download("prathmeshzade/iiit5k-words", output_dir=data_path)
    start = time.perf_counter()
    print("Starting data preprocessing IIIT5K....")


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

    os.makedirs(save_path, exist_ok=True)

    merge(traindata, args.save_path, 'train', 'iiit5k')
    merge(valdata, args.save_path, 'val', 'iiit5k')
    merge(testdata, args.save_path, 'test', 'iiit5k')

    alltexts = list(traindata['chars'])

    vocab_path = f"{save_path}/vocab.pt"
    save_vocab(vocab_path, alltexts)
    elapsed = time.perf_counter() - start
    print(f"Time taken: {elapsed / 60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Process raw text to tokens")
    parser.add_argument("--data_path", type=str, default="rec_data")
    parser.add_argument("--save_path", type=str, default="rec_data/processed")
    parser.add_argument("--prefix_path", type=str, default="rec_data/iiit5k-words/IIIT5K-Word_V3.0/IIIT5K")
    parser.add_argument("--data_type", type=str, default="both")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--download_data", type=str, default="False")

    args = parser.parse_args()
    
    if args.data_type=='iiit5k':
        iiit5k(args)
    elif args.data_type=='mjsynth':
        mjsynth(args)
    elif args.data_type=='both':
        iiit5k(args)
        mjsynth(args)
    else:
        print("Unknown data type")


if __name__ == "__main__":
    main()