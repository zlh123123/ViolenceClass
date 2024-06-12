# train.py
# 本文件是训练模型的代码

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"), transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min(
        [os.cpu_count(), batch_size if batch_size > 1 else 0, 8]
    )  # number of workers
    print("Using {} dataloader workers every process".format(nw))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"), transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    print(
        "using {} images for training, {} images for validation.".format(
            train_num, val_num
        )
    )

    net = resnet34()
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(
        model_weight_path
    )
    net.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = "./resNet34.pth"
    train_steps = len(train_loader)
    train_losses = []  # 用于保存每个epoch的训练集总体损失
    val_losses = []  # 用于保存每个epoch的测试集损失
    train_accuracies = []  # 用于保存每个epoch的训练集准确率
    val_accuracies = []  # 用于保存每个epoch的测试集准确率
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss
            )

        train_losses.append(running_loss / train_steps)
        # 计算训练准确率
        correct = 0
        total = 0
        net.eval()  # 设置模型为评估模式
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        total_val_loss = 0.0  # accumulate total validation loss
        total_batches = 0  # total number of validation batches
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)

                val_loss = loss_function(outputs, val_labels.to(device))
                total_val_loss += val_loss.item()
                total_batches += 1

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_losses.append(total_val_loss / total_batches)

        val_accurate = acc / val_num
        val_accuracies.append(val_accurate)
        print(
            "[epoch %d] train_loss: %.3f  val_accuracy: %.3f"
            % (epoch + 1, running_loss / train_steps, val_accurate)
        )

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("Finished Training")
    plot_loss(train_losses, val_losses, epochs)
    plot_accuracy(train_accuracies, val_accuracies, epochs)


# 绘制loss和val_loss曲线
def plot_loss(train_loss, val_loss, epochs):
    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label="Train Loss", color="blue")
    plt.plot(range(1, epochs + 1), val_loss, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


# 绘制accuracy和val_accuracy曲线
def plot_accuracy(train_accuracy, val_accuracy, epochs):
    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracy, label="Train Accuracy", color="blue")
    plt.plot(
        range(1, epochs + 1), val_accuracy, label="Validation Accuracy", color="red"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
