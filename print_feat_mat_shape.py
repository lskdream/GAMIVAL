import argparse
import scipy.io

parser = argparse.ArgumentParser(description='Print shape of feats_mat from a .mat file')
parser.add_argument('mat_file', help='Path to .mat file containing feats_mat')
args = parser.parse_args()

mat = scipy.io.loadmat(args.mat_file)
if 'feats_mat' not in mat:
    raise KeyError(f"feats_mat not found in {args.mat_file}")
shape = mat['feats_mat'].shape
print(f"{shape[0]} x {shape[1]}")
