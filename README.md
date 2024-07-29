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

