import torch
import torchvision
import timm
import torchvision.transforms as transforms
import os

from PIL import Image
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir="E:\File\Python\PythonProjectFlower\dataset\oxford102"
ants_label_dir="ants"
train_data=MyData(root_dir=root_dir, label_dir=root_dir)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("/logs/1104")

model = timm.create_model('efficientnet_b4', pretrained=True,num_classes=102)
# train_data = torchvision.datasets.oxford102(root='./data', trainSet=True,transform=transforms.Compose([transforms.ToTensor()]))
# test_data = torchvision.datasets.oxford102(root='./data', trainSet=False,transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

total_train_step = 0
for epoch in range(2):
    model.train()  # 训练模式
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"Train Step: {total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("Train/Loss", loss.item(), total_train_step)

    model.eval()
    total_test_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():  # 关闭梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == labels).sum().item()
    avg_test_loss = total_test_loss / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    writer.add_scalar("Test/Loss", avg_test_loss, epoch + 1)
    writer.add_scalar("Test/Accuracy", avg_accuracy, epoch + 1)

    if epoch % 30 == 0:
        torch.save(model.state_dict(), f"cj_SGD_epoch_{epoch + 1}.pth")
        print("模型已保存\n")

writer.close()