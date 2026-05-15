import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class AutoEncoders(nn.Module):
    def __init__(self, input_size, feature_dim):
        super(AutoEncoders, self).__init__()

        # Encoder layers
        self.encoderlayer1 = self.fc_block(input_size, 500)
        self.encoderlayer2 = self.fc_block(500, 500)
        self.encoderlayer3 = self.fc_block(500, 2000)
        self.encoderlayer4 = nn.Linear(2000, feature_dim)

        self.decoderlayer1 = self.fc_block(feature_dim, 2000)
        self.decoderlayer2 = self.fc_block( 2000, 500)
        self.decoderlayer3 = self.fc_block( 500, 500)

        self.final_layer = nn.Linear(500, input_size)

    def fc_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoderlayer1(x)#500
        enc2 = self.encoderlayer2(enc1)#500
        enc3 = self.encoderlayer3(enc2)#2000
        z = self.encoderlayer4(enc3)

        z1 = enc1
        z2 = enc2
        z3 = enc3

        # Decoder
        dec4 = self.decoderlayer1(z)

        dec3 = self.decoderlayer2(dec4)

        dec2 = self.decoderlayer3(dec3)

        xr = self.final_layer(dec2)

        return z, z1, z2, z3, xr  # 返回多个值

class LayerFusion(nn.Module):
    def __init__(self, dims, fusion_dim):
        super().__init__()
        # 每层投影到 fusion_dim
        self.proj_layers = nn.ModuleList([
            nn.Linear(d, fusion_dim) for d in dims
        ])
        # 每层的可训练 logit
        self.alpha = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, zs):
        projs = [proj(z) for proj, z in zip(self.proj_layers, zs)]
        w = F.softmax(self.alpha, dim=0)
        fused = sum(w[i] * projs[i] for i in range(len(projs)))
        return fused, w


class GraphConstructor(nn.Module):
    def __init__(self, k=15, metric='.'):#n最好是大于聚类中簇的个数
        super().__init__()
        self.k = k
        self.metric = metric
        #边相似度计算方法，目前有两种：1、余弦相似度，2、欧氏距离+高斯核

    def forward(self, Z):

        orig_device = Z.device
        Z_cpu = Z.detach().cpu()
        if self.metric == 'cosine':
            Z_cpu = F.normalize(Z_cpu, p=2, dim=1)
            sim = Z_cpu @ Z_cpu.t()
        else:
            dist = torch.cdist(Z_cpu, Z_cpu)
            sigma = dist.median()
            sim = torch.exp(- dist ** 2 / (sigma ** 2 + 1e-8))


        vals, idxs = sim.topk(self.k+1, dim=-1)
        vals, idxs = vals[:,1:], idxs[:,1:]
        N = Z.size(0)
        rows = torch.arange(N).unsqueeze(1).repeat(1, self.k).reshape(-1)
        cols = idxs.reshape(-1)
        weights = vals.reshape(-1)
        # 构造稠密矩阵
        W = torch.zeros_like(sim)
        W[rows, cols] = weights
        W = (W + W.t()) / 2  # 对称
        return W.to(orig_device)
        # return W

class GraphConstructor1(nn.Module):#NoisyMNIST
    def __init__(self, k=25, metric='euclidean'):
        super().__init__()
        self.k = k
        self.metric = metric

    def forward(self, Z):

        Z_cpu = Z.detach().cpu().numpy()


        nbrs = NearestNeighbors(n_neighbors=self.k+1,
                                metric=self.metric,
                                n_jobs=-1).fit(Z_cpu)
        distances, indices = nbrs.kneighbors(Z_cpu)  # 都是 [N, k+1]

        distances = distances[:, 1:]
        indices   = indices[:, 1:]


        if self.metric == 'cosine':
            weights = 1.0 - distances
        else:

            sigma = distances.std()
            weights = np.exp(- (distances**2) / (sigma**2 + 1e-8))

        N = Z_cpu.shape[0]
        rows = np.repeat(np.arange(N), self.k)   # [N*k]
        cols = indices.flatten()                 # [N*k]
        wts  = weights.flatten()                 # [N*k]

        edge_index = torch.tensor(
            [rows, cols], dtype=torch.long, device=Z.device)
        edge_weight = torch.tensor(wts, dtype=torch.float, device=Z.device)

        return edge_index, edge_weight

class SemanticMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)



class GraphFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, Ws, Xs):

        device = Ws[0].device
        alphas = []
        for W, X in zip(Ws, Xs):

            deg = W.sum(dim=1)                 # [N]
            D_inv_sqrt = torch.diag( (deg + 1e-8).rsqrt() )
            I = torch.eye(W.size(0), device=device)
            L = I - D_inv_sqrt @ W @ D_inv_sqrt

            smooth = torch.sqrt(torch.trace(X.t() @ L @ X) + 1e-8)

            alphas.append(1.0 / (smooth / torch.sqrt(X.new_tensor(X.size(1)) + 1e-8) + 1e-8))

            # alphas.append( torch.sqrt(smooth + 1e-8) )
        alphas = torch.stack(alphas)
        weights = alphas / alphas.sum()


        W_fuse = sum(w * W for w, W in zip(weights, Ws))
        return W_fuse, weights


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):

        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):

        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = self.conv2(h, edge_index, edge_weight)
        return h


class Network(nn.Module):
    def __init__(self,view, input_size, feature_dim, fusion_dim,
                 knn_k, num_classes, mlp_hidden,device):
        super().__init__()
        self.view=view
        self.AES=[]
        for v in range(view):
            self.AES.append(AutoEncoders(input_size[v],feature_dim).to(device))
        self.AES = nn.ModuleList(self.AES)

        self.fusions = nn.ModuleList([
            LayerFusion(dims=[500]*self.view, fusion_dim=500),
            LayerFusion(dims=[500]*self.view, fusion_dim=500),
            LayerFusion(dims=[2000]*self.view, fusion_dim=2000),
            LayerFusion(dims=[feature_dim] * self.view, fusion_dim=fusion_dim)
        ])
        self.graph_constructors = nn.ModuleList([
            GraphConstructor(k=knn_k),
            GraphConstructor(k=knn_k),
            GraphConstructor(k=knn_k),
            GraphConstructor(k=knn_k)
        ])

        self.graph_fusion = GraphFusion()#


        # self.z_proj1 = nn.Linear(fusion_dim,fusion_dim)
        # self.z_proj2 = nn.Linear(fusion_dim, fusion_dim)
        # self.z_proj3 = nn.Linear(fusion_dim, fusion_dim)

        self.gcn=GCNEncoder(fusion_dim,fusion_dim,fusion_dim)
        # self.gcn2 = GCNEncoder(fusion_dim, fusion_dim, fusion_dim)
        # self.gcn3 = GCNEncoder(fusion_dim, fusion_dim, fusion_dim)

        self.sem_mlp1 = SemanticMLP(in_dim=500, hidden_dim=mlp_hidden, num_classes=num_classes)  #
        self.sem_mlp2 = SemanticMLP(in_dim=500, hidden_dim=mlp_hidden, num_classes=num_classes)  #
        self.sem_mlp3 = SemanticMLP(in_dim=2000, hidden_dim=mlp_hidden, num_classes=num_classes)  #

    def forward(self, views):

        B = views[0].size(0)
        z1s, z2s, z3s,zs,xrs = [], [], [],[],[]

        for v, x in enumerate(views):
            z, z1, z2, z3, xr = self.AES[v](x)
            z1s.append(z1)
            z2s.append(z2)
            z3s.append(z3)
            zs.append(z)
            xrs.append(xr)

        Z_L1, w1 = self.fusions[0](z1s)
        Z_L2, w2 = self.fusions[1](z2s)
        Z_L3, w3 = self.fusions[2](z3s)
        Z   , w4 = self.fusions[3](zs)


        W1 = self.graph_constructors[0](Z_L1)
        W2 = self.graph_constructors[1](Z_L2)
        W3 = self.graph_constructors[2](Z_L3)
        W=self.graph_constructors[3](Z)

        W_fuse, fuse_weights = self.graph_fusion([W1,W2,W3,W],[Z_L1,Z_L2,Z_L3,Z])#自适应融合


        q1s = self.sem_mlp1(Z_L1 )
        q2s = self.sem_mlp2(Z_L2 )
        q3s = self.sem_mlp3(Z_L3 )

        return z1s,z2s,z3s,zs,xrs,Z_L1,Z_L2,Z_L3,W1,W2,W3,W,W_fuse,q1s, q2s, q3s

    def forward_cluster(self, views):


        self.eval()
        with torch.no_grad():

            *_, q1s, q2s, q3s, _ = self.forward(views)

        qs = [q1s, q2s, q3s]
        preds = [torch.argmax(q, dim=1).cpu().numpy() for q in qs]
        return qs, preds
