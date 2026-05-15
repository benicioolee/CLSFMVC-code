import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss
from dataloader import load_data
import os
from metric import evaluate
from sklearn.cluster import SpectralClustering
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time


# MNIST-USPS
# BDGP
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.9)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--mlp_hidden",default=128)
parser.add_argument("--knn_k",default=15,type=int)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--fusion_dim", default=512)
parser.add_argument("--lambda_fc",  default=1,   type=float)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 70
    seed = 10
if args.dataset == "BDGP":
    args.con_epochs = 50
    seed = 10
if args.dataset == "NoisyMNIST":
    args.con_epochs = 50#50
    seed = 10
if args.dataset == "CCV":
    args.con_epochs = 50
    seed = 3
if args.dataset == "Fashion":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Caltech-2V":
    args.con_epochs = 100
    seed = 8#8
if args.dataset == "Caltech-3V":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 100#
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 60
    seed = 10
if args.dataset == "Hdigit":
    args.con_epochs = 50
    seed = 10
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    seed = 100
if args.dataset == "Prokaryotic":
    args.con_epochs = 50
    seed = 10000
if args.dataset == "Cifar10":
    args.con_epochs = 15
    seed = 10
if args.dataset == "YouTubeFace":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Cora":
    args.mse_epochs = 10
    args.con_epochs = 50   #Just lower the number of reconstruction rounds
    seed = 100
if args.dataset == "Handwritten":
    args.con_epochs = 150
    seed = 10
if args.dataset == "BBCSport":
    args.con_epochs = 70
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

dataset, dims, view, data_size, class_num = load_data(args.dataset)
full_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        z1s,z2s,z3s,zs,xrs,_,_,_,_,_,_,_,_,_, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        z1s,z2s,z3s,zs,_,_,_,_,_,_,_,_,_,q1s, q2s, q3s  = model(xs)
        loss_fc = 0.0

        for z_views in (z1s, z2s, z3s,zs):
            V = len(z_views)
            for i in range(V):
                for j in range(i + 1, V):
                    lf = criterion.forward_feature(z_views[i], z_views[j])

                    lf = torch.nan_to_num(lf, nan=0.0, posinf=0.0, neginf=0.0)
                    loss_fc += lf

        loss = args.lambda_fc * loss_fc
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    avg = tot_loss / len(data_loader)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
    return avg, acc, nmi, pur



accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    overall_start = time.time()
    setup_seed(seed)
    model = Network(view,dims, args.feature_dim, args.fusion_dim, args.knn_k,class_num, args.mlp_hidden,device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    pre_start = time.time()
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    print(f"[TIME] Pretraining: {time.time() - pre_start:.2f}s")
    con2end_start = time.time()
    while epoch <= args.mse_epochs + args.con_epochs:
        contrastive_train(epoch)



        if epoch == args.mse_epochs + args.con_epochs:
            # acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)


            model.eval()

            views_all, _, _ = next(iter(full_loader))
            views_all = [v.to(device) for v in views_all]

            z1s,z2s,z3s,zs,xrs,Z_L1,Z_L2,Z_L3,W1,W2,W3,W,W_fuse,q1s, q2s, q3s = model(views_all)

            Wf = W_fuse.detach().cpu().numpy()

            sc = SpectralClustering(
                                n_clusters = class_num,
                            affinity = 'precomputed',
                            assign_labels = 'discretize',
                            random_state = 0
                                    )
            preds = sc.fit_predict(Wf)




            # idx = (W_fuse > 0).nonzero(as_tuple=False)
            # rows, cols = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
            # W_cpu = W_fuse.detach().cpu()
            # vals = W_cpu[rows, cols].numpy()
            #
            # W_sparse = csr_matrix((vals, (rows, cols)), shape=(W_fuse.size(0),) * 2)
            #
            # embed = spectral_embedding(W_sparse, n_components=class_num, eigen_solver='arpack')
            #
            # km = KMeans(n_clusters=class_num, random_state=0)
            # preds = km.fit_predict(embed)



            if hasattr(dataset, 'labels'):
                labels = dataset.labels
            elif hasattr(dataset, 'y'):
                labels = dataset.y
            elif hasattr(dataset, 'Y'):
                labels = dataset.Y
            else:

                labels = []
                for _, lab, _ in full_loader:

                    if isinstance(lab, torch.Tensor):
                        labels.append(lab.cpu().item())
                    else:
                        labels.append(lab)
                labels = np.array(labels)

            labels = labels.reshape(-1)
            preds = preds.reshape(-1)

            nmi, ari, acc, pur = evaluate(labels, preds)
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print(f"Graph clustering → ACC={acc:.4f}, NMI={nmi:.4f}, PUR={pur:.4f}")
            print('Saving model...')
        epoch += 1
    print(f"[TIME] Contrastive → End: {time.time() - con2end_start:.2f}s")
    print(f"[TIME] TOTAL training (this round): {time.time() - overall_start:.2f}s")
