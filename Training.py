import matplotlib.pyplot as plt
import pickle as pkl
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
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold


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


def get_data_loader(batch_size, data_list, train_idx, test_idx):
    train_idx_list = train_idx.tolist()
    train_data_list = [data_list[i] for i in train_idx_list]

    test_idx_list = test_idx.tolist()
    valid_data_list = [data_list[i] for i in test_idx_list]

    train_dataset = InMemoryDataset(train_data_list)
    valid_dataset = InMemoryDataset(valid_data_list)
    train_data_loader = train_dataset.get_data_loader(
        batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
    valid_data_loader = valid_dataset.get_data_loader(
        batch_size=batch_size, num_workers=1, shuffle=True, collate_fn=collate_fn)
    return train_data_loader, valid_data_loader


class ADMET(nn.Layer):
    def __init__(self):
        super(ADMET, self).__init__()
        compound_encoder_config = json.load(
            open(r'./geognn_l8.json', 'r'))
        print(compound_encoder_config)
        self.encoder = GeoGNNModel(compound_encoder_config)
        self.encoder.set_state_dict(pdl.load("./weights/class.pdparams"))
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


def trial(EPOCH,model_version, model, criterion, scheduler, opt, train_data_loader, valid_data_loader):
    current_best_metric = -1e10
    AP = []
    AUC = []
    L_val = []
    L_train = []
    for epoch in range(EPOCH):
        model.train()
        train_running_loss = 0.0
        counter = 0
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in train_data_loader:
            counter += 1
            label_predict_batch = model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            loss = criterion(label_predict_batch, label_true_batch)
            train_running_loss += loss.item()
            loss.backward()  # 反向传播
            opt.step()  # 更新参数
            opt.clear_grad()
        scheduler.step()
        TL = train_running_loss / counter
        L_train.append(TL)
        model.eval()
        with paddle.no_grad():
            metric_valid = evaluate(model, valid_data_loader)
        print(metric_valid)
        AP.append(metric_valid['ap'])
        AUC.append(metric_valid['auc'])
        L_val.append(metric_valid['loss'])

        score = round((metric_valid['ap'] + metric_valid['auc']) / 2, 4)
        if score > current_best_metric:
            current_best_metric = score
            pdl.save(model.state_dict(), "weights/" + str(model_version) + ".pkl")
    return AP, AUC, L_val, L_train


def evaluate(model, data_loader):
    label_true = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    counter = 0
    current_test_loss = 0.0

    for (atom_bond_graph, bond_angle_graph, label_true_batch) in data_loader:
        counter += 1
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
        loss = criterion(label_predict_batch, label_true_batch)
        current_test_loss += loss.item()

        label_predict_batch = F.softmax(label_predict_batch)

        label_true = pdl.concat((label_true, label_true_batch.detach()), axis=0)
        label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
    T_loss = current_test_loss / counter

    y_pred = label_predict[:, 1].cpu().numpy()
    y_true = label_true.cpu().numpy()

    ap = round(average_precision_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_pred), 4)

    y_pred = np.where(y_pred >= 0.5, 1, 0)
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    confusion_mat = confusion_matrix(y_true, y_pred)

    metric = {'ap': ap, 'auc': auc, 'loss': T_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
              'f1': f1,
              'confusion_mat': confusion_mat}
    return metric

L5 = []
L5_v = []
T_AUC = []
P_ap = []
K = 5
cl_splits = KFold(n_splits=K, shuffle=True, random_state=666)
data_list = pkl.load(open(f'./data/train_data_list.pkl', 'rb'))
for fold, (train_idx, test_idx) in enumerate(cl_splits.split(np.arange(len(data_list)))):
    model = ADMET()
    batch_size = 1000  # batch size
    EPOCH = 50
    SEED = 1024
    learning_rate = 1e-3
    pdl.seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=15)  # 余弦退火学习率
    opt = optimizer.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-5)  # 优化器
    train_data_loader, valid_data_loader = get_data_loader(batch_size=batch_size, data_list=data_list,
                                                           train_idx=train_idx, test_idx=test_idx)
    AP, AUC, L_val, L_train = trial(EPOCH=EPOCH,model_version=fold, model=model,
                                    criterion=criterion, scheduler=scheduler, opt=opt,
                                    train_data_loader=train_data_loader,
                                    valid_data_loader=valid_data_loader)

    L5.append(L_train)
    L5_v.append(L_val)
    T_AUC.append(AUC)
    P_ap.append(AP)

fig1, ax1 = plt.subplots()
for i, fold_train_loss in enumerate(L5):
    ax1.plot(fold_train_loss, label=f"Fold {i + 1}")
ax1.legend(loc='upper right')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('5-fold cross-validation training losses')

# 绘制验证损失函数折线图
fig2, ax2 = plt.subplots()
for i, fold_val_loss in enumerate(L5_v):
    ax2.plot(fold_val_loss, label=f"Fold {i + 1}")
ax2.legend(loc='upper right')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.set_title('5-fold cross-validation validation losses')

fig3, ax3 = plt.subplots()
for i, fold_val_loss in enumerate(T_AUC):
    ax3.plot(fold_val_loss, label=f"Fold {i + 1}")
ax3.legend(loc='upper right')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation ROC-AUC')
ax3.set_title('Validation ROC-AUC')

fig4, ax4 = plt.subplots()
for i, fold_val_loss in enumerate(P_ap):
    ax4.plot(fold_val_loss, label=f"Fold {i + 1}")
ax4.legend(loc='upper right')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Validation average_precision_score')
ax4.set_title('Validation average_precision_score')

fig1.savefig("./images/train_loss.png")
fig2.savefig("./images/val_loss.png")
fig3.savefig("./images/ROC-AUC.png")
fig4.savefig("./images/average_precision_score.png")
