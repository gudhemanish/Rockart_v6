import os, json, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from rockart_model import build_rockart

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = r"D:\Sunayana\PHD\Project\CNN\Datasets"
TRAIN = os.path.join(DATASET, "train")
VAL   = os.path.join(DATASET, "val")

def main():
    train_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(TRAIN, transform=train_tf)
    val_ds   = datasets.ImageFolder(VAL, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    # Save class order (standalone + embedded in checkpoint so inference cannot drift)
    with open("class_names.json", "w") as f:
        json.dump(train_ds.classes, f)

    model = build_rockart(num_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, 20, gamma=0.1)

    for epoch in range(30):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}/30 completed")

    torch.save(
        {"state_dict": model.state_dict(), "classes": train_ds.classes},
        "petroglyphnet.pth",
    )
    print("Model saved")

if __name__ == "__main__":
    main()