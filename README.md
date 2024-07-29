# 首届全球AI药物研发算法大赛 - Baseline 模型
引用及致敬：https://aistudio.baidu.com/projectdetail/6390399
## 赛题介绍

本次大赛聚焦于 “新冠小分子药物” 等热点议题，旨在鼓励参赛者利用深度学习等方法，发掘治疗新冠病毒的潜在药物。初赛阶段，参赛选手需要利用大赛提供的新冠病毒主蛋白酶抑制活性数据，预测小分子抑制主蛋白酶活性的概率。

## 竞赛数据集

- **训练集**：初赛训练集收集了文献中公开发表的17902个分子对新冠病毒主蛋白酶的抑制活性数据。
- **测试集**：测试集共包含1059个分子，部分数据来自公开发表的文献，部分则来源于未发表的实验数据。

## 环境配置

### 安装 PaddlePaddle

根据您的CUDA版本，参考 [PaddlePaddle安装指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html) 安装PaddlePaddle。

### 安装必要的第三方库

```python
from IPython.display import clear_output

!pip install --upgrade pip
!pip install rdkit
!pip install scikit-learn==1.0.2
!pip install -U scipy
!pip install -U seaborn
!pip install pgl==2.2.3
!pip install -U numpy

clear_output()
print('安装完成')
```
## 预处理数据
- 运行 pre_data.py 脚本进行数据预处理。这一步将执行以下操作：
```
python pre_data.py
```
- 读取data文件夹的csv数据
- 将SMILES转化为RDKit标准SMILES
- 去除重复值
- 将SMILES列表保存为 smiles_list.pkl 文件
- 使用分子力场将SMILES转化为3D分子图，并保存为 smiles_to_graph_dict.pkl 文件
- 将SMILES、图、标签构建成一个列表，并保存为 data_list.pkl 文件，该文件为GEM读取的数据文件
- 最后在 data/ 文件夹中将生成 train_smiles_to_graph_dict.pkl 和 test_smiles_to_graph_dict.pkl 文件。
## 5折交叉验证训练模型
```
python Training.py
```
- 训练完成将在weights保存模型参数及在images生成训练过程图




