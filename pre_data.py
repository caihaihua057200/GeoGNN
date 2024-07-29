import pandas as pd

train_df = pd.read_csv('data/train.csv')

print(f'len of train_df is {len(train_df)}')
train_df.head()

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # 屏蔽RDKit的warning

for index, row in train_df.iterrows():
    try:
        mol = Chem.MolFromSmiles(row['SMILES'])
        new_smiles = Chem.MolToSmiles(mol)
        train_df.loc[index, 'SMILES'] = new_smiles
    except:
        # 若转化失败，则认为原始smile不合法，删除该数据
        train_df.drop(index, inplace=True)

print(f'len of train_df is {len(train_df)}')

# 去除重复值
duplicate_rows = train_df[train_df.duplicated('SMILES', keep=False)]

for smiles, group in duplicate_rows.groupby('SMILES'):
    if len(group.drop_duplicates(subset=['label'])) == 1:
        train_df.drop(index=group.index[1:], inplace=True)
    else:
        train_df.drop(index=group.index, inplace=True)

print(f'len of train_df is {len(train_df)}')

train_df.to_csv('data/train_preprocessed.csv', index=0)

# 将smiles列表保存为smiles_list.pkl文件
import pickle as pkl
import pandas as pd

train_df = pd.read_csv('data/train_preprocessed.csv')  # 读取预处理好的训练数据

smiles_list = train_df["SMILES"].tolist()
pkl.dump(smiles_list, open('data/train_smiles_list.pkl', 'wb'))

# 测试集
test_df = pd.read_csv('data/test_nolabel.csv')
smiles_list = test_df["SMILES"].tolist()
pkl.dump(smiles_list, open('data/test_smiles_list.pkl', 'wb'))

# 使用分子力场将smiles转化为3d分子图，并保存为smiles_to_graph_dict.pkl文件
from threading import Thread, Lock
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem

mutex = Lock()  # 互斥锁，防止多个线程同时修改某一文件或某一全局变量，引发未知错误

def calculate_3D_structure_(smiles_list):
    n = len(smiles_list)
    global p
    index = 0
    while True:
        mutex.acquire()  # 获取锁
        if p >= n:
            mutex.release()
            break
        index = p        # p指针指向的位置为当前线程要处理的smiles
        smiles = smiles_list[index]
        print(index, ':', round(index / n * 100, 2), '%', smiles)
        p += 1           # 修改全局变量p
        mutex.release()  # 释放锁
        try:
            molecule = AllChem.MolFromSmiles(smiles)
            molecule_graph = mol_to_geognn_graph_data_MMFF3d(molecule)  # 根据分子力场生成3d分子图
        except:
            print("Invalid smiles!", smiles)
            mutex.acquire()
            with open('data/invalid_smiles.txt', 'a') as f:
                # 生成失败的smiles写入txt文件保存在该目录下
                f.write(str(smiles) + '\n')
            mutex.release()
            continue

        global smiles_to_graph_dict
        mutex.acquire()   # 获取锁
        smiles_to_graph_dict[smiles] = molecule_graph
        mutex.release()   # 释放锁

for mode in ['train', 'test']:
# for mode in ['test']:
    if mode == 'train':
        smiles_list = train_df["SMILES"].tolist()
    else:
        smiles_list = test_df["SMILES"].tolist()
    global smiles_to_graph_dict
    smiles_to_graph_dict = {}
    global p              # p为全局指针，指向即将要处理的smiles
    p = 0
    thread_count = 12      # 线程数。一般根据当前运行环境下cpu的核数来设定
    threads = []
    for i in range(thread_count):
        threads.append(Thread(target=calculate_3D_structure_, args=(smiles_list, )))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pkl.dump(smiles_to_graph_dict, open(f'data/{mode}_smiles_to_graph_dict.pkl', 'wb'))
    print(f'{mode} is Done!')


# 将smiles、graph、label构建成一个列表，并保存为data_list.pkl文件，该文件为GEM读取的数据文件
train_smiles_to_graph_dict = pkl.load(open(f'data/train_smiles_to_graph_dict.pkl','rb'))
test_smiles_to_graph_dict = pkl.load(open(f'data/test_smiles_to_graph_dict.pkl','rb'))

train_data_list = []
test_data_list = []

for index, row in train_df.iterrows():
    smiles = row["SMILES"]
    label = row["label"]
    if smiles not in train_smiles_to_graph_dict:
        continue
    data_item = {
        "smiles": smiles,
        "graph": train_smiles_to_graph_dict[smiles],
        "label": label,
    }
    train_data_list.append(data_item)

for index, row in test_df.iterrows():
    smiles = row["SMILES"]
    if smiles not in test_smiles_to_graph_dict:
        continue
    data_item = {
        "smiles": smiles,
        "graph": test_smiles_to_graph_dict[smiles],
        'label': 0
    }
    test_data_list.append(data_item)

pkl.dump(train_data_list, open('data/train_data_list.pkl', 'wb'))
pkl.dump(test_data_list, open('data/test_data_list.pkl', 'wb'))
