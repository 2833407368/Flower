import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def simple_contrast_mix(
        base_dataset,
        high_similar_pairs,
        ratio=0.3,
        mix_alpha=0.4
):
    """
    简单对比样本生成：只混合高相似类别的图像，生成"半A半B"的样本
    """
    contrast_samples = []

    # 构建高相似类别映射（双向）
    similar_map = {}
    for a, b in high_similar_pairs:
        similar_map.setdefault(a, []).append(b)
        similar_map.setdefault(b, []).append(a)

    for idx in range(len(base_dataset)):
        img_a, label_a = base_dataset[idx]

        # 只处理高相似类别
        if label_a not in similar_map:
            continue
        if np.random.rand() > ratio:
            continue

        # 随机选一个相似类别图像
        similar_labels = similar_map[label_a]
        img_b, label_b = None, None
        while img_b is None:
            rand_idx = np.random.randint(len(base_dataset))
            tmp_img, tmp_label = base_dataset[rand_idx]
            if tmp_label in similar_labels:
                img_b, label_b = tmp_img, tmp_label

        # 转为Tensor格式 (若Dataset返回PIL Image)
        if not isinstance(img_a, torch.Tensor):
            img_a = F.to_tensor(img_a)  # [C,H,W]
        if not isinstance(img_b, torch.Tensor):
            img_b = F.to_tensor(img_b)

        # 统一尺寸
        img_a = F.interpolate(img_a.unsqueeze(0), size=(600, 600),
                              mode='bilinear', align_corners=False).squeeze(0)
        img_b = F.interpolate(img_b.unsqueeze(0), size=(600, 600),
                              mode='bilinear', align_corners=False).squeeze(0)

        # 混合
        mixed_tensor = img_a * (1 - mix_alpha) + img_b * mix_alpha
        mixed_tensor = torch.clamp(mixed_tensor, 0, 1)  # [0,1] 范围

        # 转回 numpy for PIL
        mixed_np = (mixed_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H,W,C]
        mixed_img = Image.fromarray(mixed_np)

        contrast_samples.append((mixed_img, label_a))

    print(f"✅ 生成 {len(contrast_samples)} 个混合对比样本")
    return contrast_samples

# 提取最易混淆的前5对（高相似类别对）
def get_high_similar_pairs(cm, num_classes, top_k=5):
    non_diag_mask = ~np.eye(num_classes, dtype=bool)  # 排除对角线（正确分类）
    i, j = np.where(non_diag_mask)
    confusions = cm[i, j]
    top_indices = np.argsort(confusions)[::-1][:top_k]  # 按混淆次数从多到少排序
    return [(i[idx], j[idx]) for idx in top_indices]