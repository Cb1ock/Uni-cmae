import numpy as np
from sklearn import metrics
import torch
from sklearn.metrics import precision_recall_fscore_support

def calculate_stats(output, target):
    """计算每个类别的统计数据，包括precision、recall、F1和样本数量，以及整体准确率。

    参数:
      output: 2维数组或张量, (样本数, 类别数)
      target: 2维数组或张量, (样本数, 类别数)

    返回:
      stats: 包含每个类别统计数据和整体准确率的字典。
    """

    classes_num = output.shape[1]
    stats = {}

    # 将输出和目标转换为 NumPy 数组
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # 使用argmax获取预测的类别
    predictions = np.argmax(output, axis=1)
    # 使用argmax将one-hot编码的标签转换为单一类别的标签
    target_labels = np.argmax(target, axis=1)


    # 计算整体准确率
    accuracy = metrics.accuracy_score(target_labels, predictions)

    # 计算每个类别的精度、召回率、F1分数和支持度
    precision_cls, recall_cls, f1_cls, support = precision_recall_fscore_support(
        target_labels, predictions, labels=range(classes_num))

    for k in range(classes_num):
        stats[k] = {
            'precision': precision_cls[k],
            'recall': recall_cls[k],
            'f1': f1_cls[k],
            'sample_count': support[k]
        }

    # 添加整体准确率到统计数据中
    stats['overall'] = {'accuracy': accuracy}

    return stats, target_labels, predictions