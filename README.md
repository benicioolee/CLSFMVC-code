# CLSFMVC-code
code of the paper Cross-Layer Structure Information Fusion for Deep Multi-view Clustering

Running Code:
python train.py --dataset BDGP



--batch_size	Training batch size	256


--mse_epochs	Pretraining (reconstruction) epochs	200


--con_epochs	Contrastive training epochs	50 (reset automatically for BDGP)
--learning_rate	Adam learning rate	0.0003
--feature_dim	Latent feature dimension of AE	512
--fusion_dim	Dimension for multi-layer fusion	512
--mlp_hidden	Hidden dimension of semantic MLP	128
--knn_k	Number of neighbors in kNN graph	15
--temperature_f	Temperature for feature contrastive loss	0.9
--temperature_l	Temperature for label contrastive loss	1.0
--lambda_fc	Weight for feature contrastive loss	1
seed	Fixed random seed for reproducibility	10
