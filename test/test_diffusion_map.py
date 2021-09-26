# -*- coding: utf-8 -*- 
# @Time : 2021/9/15 20:34 
# @Author : lepold
# @File : test_diffusion_map.py

import os
import numpy as np
import h5py
from sklearn.metrics import pairwise_distances
from tools import diffusion_map
# from helpers.plotfuncs import create_fig
from helpers.plotcolors import myblue
import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
data_dir = os.path.join(project_path, "data/raw_data")


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def read_fmri(mode="task_bold"):
    data_path = os.path.join(data_dir, "DTI_voxel_network_mat_zenglongbin_new.mat")
    data = h5py.File(data_path, "r")
    if mode == "task_bold":
        fmri = data["dti_high_low_ts"][:]
    elif mode == "rest_bold":
        fmri = data["dti_rest_state"][:]
    else:
        raise NotImplementedError
    return fmri


def process_aff():
    write_path = os.path.join(data_dir, "../processed_data")
    os.makedirs(write_path, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 6))
    axes = axes.flatten()
    task_fmri = load_if_exist(read_fmri, data_dir, "task_fmri", mode="task_bold")
    voxels, T = task_fmri.shape
    print(f"task_fmri: voxels {voxels}, time_length {T}")
    fc = np.corrcoef(task_fmri)
    print(f"fc.shape: {fc.shape}")
    fc_data = fc.flatten()
    weights = np.ones_like(fc_data) / float(len(fc_data))
    axes[0].hist(fc_data, weights=weights, density=True, color=myblue, bins=100)
    axes[0].set_xlabel('corrcef')
    axes[0].set_ylabel('probablity')
    axes[0].set_title('original fc')

    perc = np.array([np.percentile(x, 90) for x in fc])
    for i in range(fc.shape[0]):
        fc[i, fc[i, :] < perc[i]] = 0
    print("Minimum value of processed fc is %f" % fc.min())
    neg_values = np.array([sum(fc[i, :] < 0) for i in range(voxels)])
    print("Negative values occur in %d rows" % sum(neg_values > 0))
    fc[fc < 0] = 0
    fc_data = fc.flatten()

    axes[1].hist(fc_data, weights=weights, density=True, color=myblue)
    # axes[1].set_xlim((-1., 1.))
    axes[1].set_xlabel('corrcef')
    axes[1].set_ylabel('probablity')
    axes[1].set_title('processed fc')
    del fc_data

    aff = np.load(os.path.join(write_path, "affinity.npy"))
    # aff = 1 - pairwise_distances(fc, metric='cosine')
    aff_data = aff.flatten()
    weights = np.ones_like(aff_data) / float(len(aff_data))
    axes[2].hist(aff_data, weights=weights, density=True, color=myblue)
    axes[2].set_xlabel('affinity')
    axes[2].set_ylabel('probablity')
    axes[2].set_title('affinity')
    # np.save(os.path.join(write_path, "affinity.npy"), aff)
    fig.savefig(os.path.join(write_path, "preprocessed_data.png"), dpi=300)
    print("Done!!!")


def first_gradient_compotment():
    aff = np.load(os.path.join(data_dir, "processed_data", "affinity.npy"))
    emb, res = diffusion_map.compute_diffusion_map(aff, alpha=0.5)
    np.save(os.path.join(data_dir, "processed_data", "emb.npy"), emb)
    np.save(os.path.join(data_dir, "processed_data", "res.npy"), res)
    a = [res['vectors'][:, i] / res['vectors'][:, 0] for i in range(100)]
    emb = np.array(a)[1:, :].T
    print(f"len(emb):{len(emb)}")
    print("Done")


if __name__ == '__main__':
    process_aff()