import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def set_seed(seed: int):
    """
    设置所有随机种子以确保实验可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # 以下两行可以确保卷积操作在每次运行时都是确定性的
        # 但可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
class CustomData(Data):
    """
    修复edge_index_batch维度问题的CustomData类
    针对图神经过程模型优化，添加了更好的属性管理和验证
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 line_graph_edge_index=None, edge_index_batch=None,
                 drug_id=None, smiles=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)

        # 核心属性
        self.line_graph_edge_index = line_graph_edge_index
        self.drug_id = drug_id  # 添加药物ID追踪
        self.smiles = smiles    # 添加SMILES字符串追踪

        # 修复：正确处理edge_index_batch
        if edge_index_batch is not None:
            self.edge_index_batch = edge_index_batch
        elif edge_index is not None and edge_index.nelement() > 0:
            # edge_index_batch应该是一维张量，长度等于边的数量
            num_edges = edge_index.size(1)
            self.edge_index_batch = torch.zeros(num_edges, dtype=torch.long)
        else:
            # 空图的情况
            self.edge_index_batch = torch.empty(0, dtype=torch.long)

        # 验证数据完整性
        self._validate_data()

    def _validate_data(self):
        """验证数据的完整性和一致性"""
        if self.x is not None and self.edge_index is not None:
            if self.edge_index.nelement() > 0:
                max_node_idx = self.edge_index.max().item()
                num_nodes = self.x.size(0)
                if max_node_idx >= num_nodes:
                    warnings.warn(
                        f"Edge index contains invalid node indices. Max index: {max_node_idx}, Num nodes: {num_nodes}")

        # 验证edge_index_batch的维度
        if self.edge_index is not None and self.edge_index_batch is not None:
            expected_batch_size = self.edge_index.size(1)
            actual_batch_size = self.edge_index_batch.size(0)
            if expected_batch_size != actual_batch_size:
                warnings.warn(f"edge_index_batch size mismatch. Expected: {expected_batch_size}, Got: {actual_batch_size}")
                # 自动修复
                if expected_batch_size > 0:
                    self.edge_index_batch = torch.zeros(expected_batch_size, dtype=torch.long)
                else:
                    self.edge_index_batch = torch.empty(0, dtype=torch.long)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        elif key == 'edge_index_batch':
            # edge_index_batch不需要增量，因为它表示的是批次索引
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __repr__(self):
        info = []
        for key, value in self.items():
            if value is not None:
                if hasattr(value, 'shape'):
                    info.append(f"{key}={list(value.shape)}")
                else:
                    info.append(f"{key}={value}")
        return f"CustomData({', '.join(info)})"


class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

def load_checkpoint(model_path):
    return torch.load(model_path)


def save_model_dict(model, path, msg, filename=None):
    """保存模型，允许自定义文件名"""
    if not os.path.exists(path):
        os.makedirs(path)

    if filename is None:
        filename = f"model_{time.strftime('%Y%m%d_%H%M%S')}.pth"

    model_path = os.path.join(path, filename)
    state_dict = {
        'model_state_dict': model.state_dict(),
        'msg': msg
    }
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x

#############################################################################################################
def compute_simplified_loss(model, head_graphs, tail_graphs, relations, labels):
    """为简化模型计算损失"""
    # 模型前向传播
    predictions = model(head_graphs, tail_graphs, relations)

    if predictions.dim() > 1:
        predictions = predictions.squeeze(-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    # 确保数据类型正确
    predictions = predictions.float()
    labels = labels.float()

    # 确保标签是0/1二分类
    if labels.max() > 1:
        labels = (labels > 0).float()

    # 计算二分类交叉熵损失
    loss = F.binary_cross_entropy_with_logits(predictions, labels)

    # 计算预测概率用于评估
    probs = torch.sigmoid(predictions)

    return {
        'loss': loss,
        'predictions': probs.detach(),
        'labels': labels.detach()
    }


###############################################################################################################
def compute_loss(pred, label, mu, logvar):
    # 重构误差（如二分类交叉熵）
    recon_loss = F.binary_cross_entropy_with_logits(pred, label)

    # KL 散度：约束潜变量分布接近标准正态分布
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

def accuracy(y_true, y_pred):
    """计算准确率"""
    return np.mean(y_true == y_pred)

def auroc_score(y_true, y_pred):
    """计算AUROC分数"""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def f1_score(y_true, y_pred):
    """计算F1分数"""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)

def precision_score(y_true, y_pred):
    """计算精确率"""
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred)

def recall_score(y_true, y_pred):
    """计算召回率"""
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred)

def average_precision_score(y_true, y_pred):
    """计算平均精确率分数"""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_pred)
