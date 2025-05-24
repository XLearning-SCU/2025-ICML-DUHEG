from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
import os
import h5py
from tqdm import tqdm

def compress(retrieval, query, model, backbone, device):
    retrievalB = list([])
    retrievalL = list([])
    with torch.no_grad():
        for batch_step, (data, target) in enumerate(retrieval):
            var_data = data.to(device)
            if backbone != None:
                var_data = backbone(var_data)
                var_data = nn.functional.normalize(var_data, dim=1)
            code = model(var_data, var_data, only_img=True)
            retrievalB.extend(code.cpu().data.numpy())
            retrievalL.extend(target.tolist())

    queryB = list([])
    queryL = list([])
    with torch.no_grad():
        for batch_step, (data, target) in enumerate(query):
            var_data = data.to(device)
            if backbone != None:
                var_data = backbone(var_data)
                var_data = nn.functional.normalize(var_data, dim=1)
            code = model(var_data, var_data, only_img=True)
            queryB.extend(code.cpu().data.numpy())
            queryL.extend(target.tolist())

    retrievalB = np.array(retrievalB)
    retrievalL = np.stack(retrievalL)
    
    queryB = np.array(queryB)
    queryL = np.stack(queryL)

    return retrievalB, retrievalL, queryB, queryL


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def calculate_top_map(qB, rB, queryL, retrievalL, topk):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    topkmap = 0

    for iter in range(num_query):
        if queryL.ndim == 1:
            gnd = (queryL[iter]==retrievalL.transpose()).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd

        tgnd = gnd[0:topk]
        tsum = int(np.sum(tgnd))
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def calculate_PR_curve(qB, rB, queryL, retrievalL, dataset_name):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    retrieval_query = retrievalL.shape[0]
    precisions, recalls = [], []

    for iter in tqdm(range(num_query), desc="PR_curve"):
        if queryL.ndim == 1:
            gnd = (queryL[iter]==retrievalL.transpose()).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd

        hit_counts = gnd.sum()
        if hit_counts <= 10e-6:
            hit_counts = 1.0
        hit_cumsum_list = gnd.cumsum()
        position_list = np.arange(1, retrieval_query+1, dtype=np.float)

        precision_list = hit_cumsum_list / position_list
        recall_list = hit_cumsum_list / hit_counts
        
        precisions.append(precision_list)
        recalls.append(recall_list)
    precision_axis = (np.stack(precisions)).mean(axis=0)
    recall_axis = (np.stack(recalls)).mean(axis=0)
    PR_curve = np.stack([recall_axis, precision_axis])
    np.savetxt(os.path.join("/media/hdd4/sqh/Hash_Part/Hash_TAC/Plots/DUH-EG2", dataset_name+'PR_curve.txt'), PR_curve)


def calculate_P_at_topK_curve(qB, rB, queryL, retrievalL, topk, dataset_name):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    precisions = []

    for iter in tqdm(range(num_query), desc="P_at_topK_curve"):
        if queryL.ndim == 1:
            gnd = (queryL[iter]==retrievalL.transpose()).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd
        tgnd = gnd[0:topk]

        hit_cumsum_list = tgnd.cumsum()
        position_list = np.arange(1, topk+1, dtype=np.float)

        precision_list = hit_cumsum_list / position_list
        
        precisions.append(precision_list)
    precision_axis = (np.stack(precisions)).mean(axis=0)

    P_at_topK_curve = np.stack([np.arange(1, topk+1, dtype=np.float), precision_axis])
    np.savetxt(os.path.join("/media/hdd4/sqh/Hash_Part/Hash_TAC/Plots/DUH-EG2", dataset_name+'P_at_topK_curve.txt'), P_at_topK_curve)

def calculate_top10_retrieval(qB, rB, queryL, retrievalL, dataset_name):
    """
    :param qB: {-1,+1}^{mxq} query bits
    :param rB: {-1,+1}^{nxq} retrieval bits
    :param queryL: {0,1}^{mxl} query label
    :param retrievalL: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = queryL.shape[0]
    hit_counts, tgnds, tinds = [], [], []

    for iter in tqdm(range(num_query), desc="top10"):
        if queryL.ndim == 1:
            gnd = (queryL[iter]==retrievalL.transpose()).astype(np.float32)
        else:
            gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind] # reorder gnd
        tgnd = gnd[0:10]
        tind = ind[0:10]

        hit_count = tgnd.sum()
        
        index_list = np.arange(1, num_query+1, dtype=np.float)
        hit_counts.append(hit_count)
        tgnds.append(tgnd)
        tinds.append(tind)
    tgnds = np.stack(tgnds)
    tinds = np.stack(tinds)
    PR_curve = np.stack([index_list, hit_counts])
    path = os.path.join("/media/hdd4/sqh/Hash_Part/Hash_TAC/Plots/DUH-EG", dataset_name+'top10.txt')
    with h5py.File(path, 'w') as f:
        f.create_dataset('tgnds', data=tgnds)
        f.create_dataset('tinds', data=tinds)
        f.create_dataset('PR_curve', data=PR_curve)