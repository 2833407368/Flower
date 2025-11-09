import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import Flowers102
import timm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(550, scale=(0.8, 1.0)),
    transforms.Resize((600, 600)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=0, shear=10, scale=(0.9, 1.1)),  # 移到ToTensor之后
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # 移到ToTensor之后
    transforms.Normalize(imagenet_mean, imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

train_data = Flowers102(root="E:\File\Python\PythonProjectFlower\dataset\oxford102", split="train", download=False, transform=train_transform)
test_data = Flowers102(root="E:\File\Python\PythonProjectFlower\dataset\oxford102", split="test", download=False, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# model = timm.create_model('resnet34', pretrained=True, num_classes=102)
# model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=102)
model = torchvision.models.mobilenet_v3_small()
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, 102)
model.classifier[2] = torch.nn.Dropout(0.5)

model = model.to(device)
# model.load_state_dict(torch.load("efficientnet_b3_0.8374_best_model.pth", map_location=device))
model.load_state_dict(torch.load("mobilenet_v3_small_0.845503_best_model.pth", map_location=device))

writer = SummaryWriter("logs/11093")
epochs = 1

# model.add_module("add_linear",torch.nn.Linear(1000,10))
# model.classifier.add_module("add_linear",torch.nn.Linear(1000,10))
# model.classifier[6] = torch.nn.Linear(4096,10)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

print(f"当前使用的设备：{device}")

best_acc = 0

for epoch in range(epochs):
    print(f"----- Epoch {epoch+1}/{epochs} -----")
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
    print(f"Epoch [{epoch+1}], Loss: {total_loss/len(train_data):.6f},acc: {total_train_accuracy/len(train_data):.6f}")
    writer.add_scalar("Train/Loss", total_loss/len(train_data), epoch+1)
    writer.add_scalar("Train/Accuracy", total_train_accuracy / len(train_data), epoch + 1)


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
    avg_test_loss = total_test_loss / len(test_data)
    avg_accuracy = total_accuracy / len(test_data)

    print(f"Test Loss: {avg_test_loss:.6f}, Accuracy: {avg_accuracy:.6f}")
    writer.add_scalar("Test/Loss", avg_test_loss, epoch + 1)
    writer.add_scalar("Test/Accuracy", avg_accuracy, epoch + 1)

    if avg_accuracy > best_acc:
        best_acc = avg_accuracy
        # torch.save(model.state_dict(), f"mobilenet_v3_small_{best_acc:.6f}_best_model.pth")
        torch.save(model, f"mobilenet_v3_small_best_model.pth")
        print(f"新的最高准确率：{best_acc:.6f}，模型已保存为 mobilenet_v3_small_best_model.pth\n")

writer.close()
