import torch
import torch.nn as nn
from tqdm import tqdm

from model import CNN
from dataset import get_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_loader, classes = get_dataloader("../data/train")

print(classes)
print(len(classes))


model = CNN(num_classes=len(classes)).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} done, Total Loss: {running_loss:.4f}")


torch.save(model.state_dict(), "../model.pth")
print("Model saved!")