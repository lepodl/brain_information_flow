# -*- coding: utf-8 -*- 
# @Time : 2021/9/15 20:34 
# @Author : lepold
# @File : test_diffusion_map.py

import os
import numpy as np
import h5py
import logging
from sklearn.metrics import pairwise_distances
from tools import diffusion_map

logging.basicConfig(level=logging.DEBUG)
data_dir = os.getcwd() + "/../data"


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def read_fmri(mode="task_bold"):
    data_path = os.path.join(data_dir, "DTI_voxel_network_mat_0719.mat")
    data = h5py.File(data_path, "r")
    if mode == "task_bold":
        fmri = data["dti_evaluation"][:]
    elif mode == "rest_bold":
        fmri = data["dti_rest_state"][:]
    else:
        raise NotImplementedError
    return fmri


def process_aff():
    task_fmri = load_if_exist(read_fmri, data_dir, "task_fmri", mode="task_bold")
    voxels, T = task_fmri.shape
    print(f"task_fmri: voxels {voxels}, time_length {T}")
    dcon = np.tanh(task_fmri)
    print("N", dcon.shape[0])
    perc = np.array([np.percentile(x, 90) for x in dcon])
    for i in range(dcon.shape[0]):
        dcon[i, dcon[i, :] < perc[i]] = 0
    print("Minimum value is %f" % dcon.min())
    neg_values = np.array([sum(dcon[i, :] < 0) for i in range(voxels)])
    print("Negative values occur in %d rows" % sum(neg_values > 0))
    dcon[dcon < 0] = 0
    aff = 1 - pairwise_distances(dcon, metric='cosine')
    write_path = os.path.join(data_dir, "processed_data")
    os.makedirs(write_path, exist_ok=True)
    np.save(os.path.join(write_path, "affinity.npy"), aff)
    print("Done!!!")


def first_gradient_compotment():
    log = logging.getLogger('test_first_gradient_compotment')
    aff = np.load(os.path.join(data_dir, "processed_data", "affinity.npy"))
    emb, res = diffusion_map.compute_diffusion_map(aff, alpha=0.5)
    np.save(os.path.join(data_dir, "processed_data", "emb.npy"), emb)
    np.save(os.path.join(data_dir, "processed_data", "res.npy"), res)
    a = [res['vectors'][:, i] / res['vectors'][:, 0] for i in range(100)]
    emb = np.array(a)[1:, :].T
    log.info(f"len(emb):{len(emb)}")
    log.info("Done")








