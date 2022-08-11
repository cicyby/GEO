import json
import os
import re
import random
import pickle
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="observed", type=str)
parser.add_argument("--save_dir", default="data", type=str)
parser.add_argument("--seed", default=2022, type=int)
parser.add_argument("--data_split_ratio", default=[0.7, 0.2, 0.1], type=list)
args = parser.parse_args()


def data_split(full_list: list, ratio: list[float], args, shuffle=False):
    random.seed(args.seed)
    n_total = len(full_list)
    offset1 = int(n_total*ratio[0])
    offset2 = int(n_total*(ratio[0]+ratio[1]))
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3


def gen_pkl_data(args):
    if not os.path.exists(args.data_dir):
        print(f"{args.data_dir} not exist!")
        exit()
    else:
        data_path_list = os.listdir(args.data_dir)
    # sorted
    data_path_list = sorted(data_path_list, key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))

    st_vp, st_vs, noise = [], [], []
    rdispph_list, rdispph_with_noise_list = [], []
    prf_list, prf_with_noise_list = [], []
    rwe_list, rwe_with_noise_list = [], []

    for data_path in data_path_list:

        # append noise
        if data_path.startswith("noise"):
            data = pd.read_csv(os.path.join(args.data_dir, data_path), header=None)
            noise.append(data[:][0].tolist())
            print(f'{data_path} over')

        # append st_* label
        if data_path.startswith("st"):
            data = pd.read_csv(os.path.join(args.data_dir, data_path), sep="\t")
            st_vp.append(data[:]["vp"].tolist())
            st_vs.append(data[:]["vs"].tolist())
            print(f'{data_path} over')

        # append syn data
        if data_path.startswith("syn"):
            # with noise
            if "noise" in data_path:
                if "rdispph" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep=" ", header=None)
                    rdispph_with_noise_list.append(data[:][1].tolist())
                    print(f'{data_path} over')
                if "prf" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep=" ", header=None)
                    prf_with_noise_list.append(data[:][1].tolist())
                    print(f'{data_path} over')
                if "rwe" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep=" ", header=None)
                    rwe_with_noise_list.append(data[:][1].tolist())
                    print(f'{data_path} over')

            # without noise
            else:
                if "rdispph" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep="\t", header=None)
                    rdispph_list.append(data[:][1].tolist())
                    print(f'{data_path} over')
                if "prf" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep="\t", header=None)
                    prf_list.append(data[:][1].tolist())
                    print(f'{data_path} over')
                if "rwe" in data_path:
                    data = pd.read_csv(os.path.join(args.data_dir, data_path), sep="\t", header=None)
                    rwe_list.append(data[:][1].tolist())
                    print(f'{data_path} over')

    if len(st_vp) != len(st_vs) != len(rdispph_list)\
            != len(rdispph_with_noise_list)\
            != len(rwe_list) != len(rwe_with_noise_list)\
            != len(prf_list) != len(prf_with_noise_list):
        raise "Datasets are not aligned"

    # split train val test data set
    train_label_vp, val_label_vp, test_label_vp = data_split(st_vp, ratio=args.data_split_ratio, args=args)
    train_label_vs, val_label_vs, test_label_vs = data_split(st_vs, ratio=args.data_split_ratio, args=args)

    train_rdispph, val_rdispph, test_rdispph = data_split(rdispph_list, ratio=args.data_split_ratio, args=args)
    train_rdispph_noise, val_rdispph_noise, test_rdispph_noise = data_split(rdispph_with_noise_list, ratio=args.data_split_ratio, args=args)

    train_prf, val_prf, test_prf = data_split(prf_list, ratio=args.data_split_ratio, args=args)
    train_prf_noise, val_prf_noise, test_prf_noise = data_split(prf_with_noise_list, ratio=args.data_split_ratio, args=args)

    train_rwe, val_rwe, test_rwe = data_split(rwe_list, ratio=args.data_split_ratio, args=args)
    train_rwe_noise, val_rwe_noise, test_rwe_noise = data_split(rwe_with_noise_list, ratio=args.data_split_ratio, args=args)

    data_set = {"train": {
        "with_noise": {
            "rdispph": train_rdispph_noise,
            "prf": train_prf_noise,
            "rwe": train_rwe_noise
        },
        "without_noise": {
            "rdispph": train_rdispph,
            "prf": train_prf,
            "rwe": train_rwe
        },
        "labels": {
            "vp": train_label_vp,
            "vs": train_label_vs
        }
    }, "test": {
        "with_noise": {
            "rdispph": test_rdispph_noise,
            "prf": test_prf_noise,
            "rwe": test_rwe_noise
        },
        "without_noise": {
            "rdispph": test_rdispph,
            "prf": test_prf,
            "rwe": test_rwe
        },
        "labels": {
            "vp": test_label_vp,
            "vs": test_label_vs
        }
    }, "val": {
        "with_noise": {
            "rdispph": val_rdispph_noise,
            "prf": val_prf_noise,
            "rwe": val_rwe_noise
        },
        "without_noise": {
            "rdispph": val_rdispph,
            "prf": val_prf,
            "rwe": val_rwe
        },
        "labels": {
            "vp": val_label_vp,
            "vs": val_label_vs
        }
    }}

    with open("./simulation_data_pkl.json", "w") as jf:
        jf.write(json.dumps(data_set))
    # write to pkl
    with open("./simulation_data.pkl", "wb") as pf:
        pickle.dump(data_set, pf)
        pf.close()


if __name__ == '__main__':
    gen_pkl_data(args)