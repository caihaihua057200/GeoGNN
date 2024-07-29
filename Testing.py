
import pickle as pkl
import pandas as pd
import paddle as pdl
import paddle.nn as nn
from pahelix.model_zoo.gem_model import GeoGNNModel
import json
import pgl
from pahelix.datasets.inmemory_dataset import InMemoryDataset
import random
import paddle
from paddle import optimizer
import numpy as np
import paddle.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


def collate_fn(data_batch):
    """
    Dataloader中的数据处理函数
    该函数输入一个batch的数据, 返回一个batch的(atom_bond_graph, bond_angle_graph, label)
    """
    atom_names = ["atomic_num", "formal_charge", "degree", "chiral_tag", "total_numHs", "is_aromatic", "hybridization"]
    bond_names = ["bond_dir", "bond_type", "is_in_ring"]
    bond_float_names = ["bond_length"]
    bond_angle_float_names = ["bond_angle"]

    atom_bond_graph_list = []  # 原子-键特征图
    bond_angle_graph_list = []  # 键-键角特征图
    label_list = []  # label

    for data_item in data_batch:
        graph = data_item['graph']
        ab_g = pgl.Graph(
            num_nodes=len(graph[atom_names[0]]),
            edges=graph['edges'],
            node_feat={name: graph[name].reshape([-1, 1]) for name in atom_names},
            edge_feat={
                name: graph[name].reshape([-1, 1]) for name in bond_names + bond_float_names})
        ba_g = pgl.Graph(
            num_nodes=len(graph['edges']),
            edges=graph['BondAngleGraph_edges'],
            node_feat={},
            edge_feat={name: graph[name].reshape([-1, 1]) for name in bond_angle_float_names})
        atom_bond_graph_list.append(ab_g)
        bond_angle_graph_list.append(ba_g)
        label_list.append(data_item['label'])

    atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
    bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)

    # TODO: reshape due to pgl limitations on the shape
    def _flat_shapes(d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    _flat_shapes(atom_bond_graph.node_feat)
    _flat_shapes(atom_bond_graph.edge_feat)
    _flat_shapes(bond_angle_graph.node_feat)
    _flat_shapes(bond_angle_graph.edge_feat)

    return atom_bond_graph, bond_angle_graph, np.array(label_list, dtype=np.float32)


def get_data_loader(batch_size, data_list):
    test_dataset = InMemoryDataset(data_list)
    test_data_loader = test_dataset.get_data_loader(
        batch_size=batch_size, num_workers=1, shuffle=False, collate_fn=collate_fn)
    return test_data_loader


class ADMET(nn.Layer):
    def __init__(self):
        super(ADMET, self).__init__()
        compound_encoder_config = json.load(
            open(r'./geognn_l8.json', 'r'))
        self.encoder = GeoGNNModel(compound_encoder_config)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 32, weight_attr=nn.initializer.KaimingNormal()),
            nn.ReLU(),
            nn.Linear(32, 2, weight_attr=nn.initializer.KaimingNormal()),
        )
    def forward(self, atom_bond_graph, bond_angle_graph):
        node_repr, edge_repr, graph_repr = self.encoder(atom_bond_graph.tensor(), bond_angle_graph.tensor())
        return self.mlp(graph_repr)


def trial(model, k):
    state_dict = paddle.load(fr'./weights/{k}.pkl')
    model.set_state_dict(state_dict)
    model.eval()
    with paddle.no_grad():
        metric_train = evaluate(model, test_data_loader)
    return metric_train


def evaluate(model, data_loader):
    label_true = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in data_loader:
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
        label_predict_batch = F.softmax(label_predict_batch)
        label_true = pdl.concat((label_true, label_true_batch.detach()), axis=0)
        label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
    y_pred = label_predict[:, 1].cpu().numpy()
    return y_pred


# 固定随机种子
SEED = 1024
pdl.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
model = ADMET()
batch_size = 500  # batch size
data_list = pkl.load(open(f'data/test_data_list.pkl', 'rb'))
data_df = pd.read_csv(r'./data/test_nolabel.csv')
test_data_loader = get_data_loader(batch_size=batch_size, data_list=data_list)
P = []
for k in range(5):
    predict = trial(model=model,k=k)
    P.append(predict)
P = np.mean(P, axis=0)
data_df['打分'] = P
predict = pd.DataFrame(data_df)
predict.to_csv(r'./predict.csv',index=False)
