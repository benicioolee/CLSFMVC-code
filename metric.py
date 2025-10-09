from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding
from scipy.sparse import csr_matrix


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    """
    :return:
    total_pred: prediction among all modalities
    pred_vectors: predictions of each modality, list
    labels_vector: true label
    Hs: high-level features
    Zs: low-level features
    """
    model.eval()
    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            hs, _, _, zs = model.forward(xs)
            q = sum(qs)/view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return total_pred, pred_vectors, Hs, labels_vector, Zs



def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
    test_loader = DataLoader(
        dataset,
        batch_size=data_size,  
        shuffle=False,
    )

  
    full_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    views_all, labels_all, _ = next(iter(full_loader))
    views_all = [v.to(device) for v in views_all]
    model.eval()
    with torch.no_grad():
        z1s, z2s, z3s, zs,xrs, \
            Z_L1, Z_L2, Z_L3, \
            W1, W2, W3, W,W_fuse, \
            q1s, q2s, q3s = model(views_all)

    Wf = W_fuse.detach().cpu().numpy()
    sc = SpectralClustering(
            n_clusters=class_num,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=0
    )
    preds = sc.fit_predict(Wf)


    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'y'):
        labels = dataset.y
    else:
        labels = labels_all.cpu().numpy()
    labels = np.array(labels).reshape(-1)
    preds = preds.reshape(-1)

    nmi, ari, acc, pur = evaluate(labels, preds)
    print(f"   ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}")
    return acc, nmi, pur

