import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

# MNIST-USPS
# BDGP

Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
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
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view,dims, args.feature_dim, args.fusion_dim, args.knn_k,class_num, args.mlp_hidden,device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
# model.load_state_dict(checkpoint, strict=False)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, class_num, eval_h=False)
