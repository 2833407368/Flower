import torch
import torchvision
import timm
import torchvision.transforms as transforms
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model = timm.create_model('efficientnet_b4', pretrained=True,num_classes=102)
print(model)
# model.eval()
# transform = transforms.Compose([transforms.ToTensor()])

for epoch in range(num_epochs):
    model.train()  # 训练模式
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()  # 推理模式
    with torch.no_grad():  # 关闭梯度，节省显存
        for images, labels in val_loader:
            outputs = model(images)
            val_loss = criterion(outputs, labels)