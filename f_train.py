import torch
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import timm
from torch.utils.tensorboard import SummaryWriter

class FlowerDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None, skip_corrupted=True):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.skip_corrupted = skip_corrupted
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self):
        """筛选出有效的图像索引（排除损坏文件）"""
        valid_indices = []
        for idx in range(len(self.labels)):
            img_name = self.labels.iloc[idx, 0]
            img_path = os.path.join(self.image_dir, img_name)
            try:
                # 简单验证文件是否可打开
                with Image.open(img_path) as img:
                    img.verify()  # 验证图像完整性
                valid_indices.append(idx)
            except (IOError, OSError):
                if self.skip_corrupted:
                    print(f"跳过损坏图像: {img_path}")
                else:
                    valid_indices.append(idx)  # 不跳过则保留，但加载时可能报错
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 从有效索引中获取原始索引
        original_idx = self.valid_indices[idx]
        img_name = self.labels.iloc[original_idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            if self.skip_corrupted:
                # 极端情况：预筛选漏检，此处再次捕获并跳过（返回下一个样本）
                print(f"加载时发现损坏图像: {img_path}，尝试跳过")
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise e  # 不跳过则抛出错误

        label = self.labels.iloc[original_idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(550, scale=(0.8, 1.0)),
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.CenterCrop(600),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    # 数据集路径设置
    TRAIN_IMAGE_DIR = "dataset/trainSet"  # 图像文件夹
    LABEL_CSV_PATH = "dataset/train_labels.csv"  # 标签CSV文件

    # 新增：加载标签并划分训练集和验证集
    labels_df = pd.read_csv(LABEL_CSV_PATH)

    # --- 标签重新映射，确保连续编号 ---
    unique_labels = sorted(labels_df.iloc[:, 1].unique())
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    labels_df.iloc[:, 1] = labels_df.iloc[:, 1].map(label_map)

    num_classes = len(unique_labels)
    print(f"标签重新编号完成，共 {num_classes} 类。编号范围：{labels_df.iloc[:, 1].min()} ~ {labels_df.iloc[:, 1].max()}")
    train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df.iloc[:, 1])

    train_data = FlowerDataset(
        image_dir=TRAIN_IMAGE_DIR,
        labels=train_df,
        transform=train_transform
    )
    val_data = FlowerDataset(
        image_dir=TRAIN_IMAGE_DIR,
        labels=val_df,
        transform=val_transform
    )

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

    model = torchvision.models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[2] = torch.nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = torch.nn.Linear(in_features, num_classes)

    # model.load_state_dict(torch.load("mobilenet_v3_small_best_model.pth", map_location=device))
    state_dict = torch.load("mobilenet_v3_small_0.845503_best_model.pth", map_location=device)
    state_dict.pop("classifier.3.weight", None)
    state_dict.pop("classifier.3.bias", None)
    model.load_state_dict(state_dict, strict=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 100)
    model = model.to(device)

    writer = SummaryWriter("logs/11095")
    epochs = 20

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    print(f"当前使用的设备：{device}")
    print(f"训练集规模：{len(train_data)}，验证集规模：{len(val_data)}")

    best_acc = 0

    for epoch in range(epochs):
        print(f"----- Epoch {epoch + 1}/{epochs} -----")
        model.train()
        total_loss = 0
        total_train_accuracy = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_train_accuracy += (outputs.argmax(1) == labels).sum().item()

        avg_train_loss = total_loss / len(train_data)
        avg_train_acc = total_train_accuracy / len(train_data)
        print(f"Epoch [{epoch + 1}], Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_acc:.6f}")
        writer.add_scalar("Train/Loss", avg_train_loss, epoch + 1)
        writer.add_scalar("Train/Accuracy", avg_train_acc, epoch + 1)

        model.eval()
        total_val_loss = 0.0
        total_val_accuracy = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item()
                total_val_accuracy += (outputs.argmax(1) == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_data)
        avg_val_acc = total_val_accuracy / len(val_data)

        print(f"Val Loss: {avg_val_loss:.6f}, Val Accuracy: {avg_val_acc:.6f}")
        writer.add_scalar("Val/Loss", avg_val_loss, epoch + 1)
        writer.add_scalar("Val/Accuracy", avg_val_acc, epoch + 1)

        scheduler.step(avg_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.8f}")

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), f"mobilenet_v3_small_best_model_dict.pth")
            print(f"新的最高验证准确率：{best_acc:.6f}，模型已保存\n")

    writer.close()