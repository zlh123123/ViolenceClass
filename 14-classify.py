# classify.py
# 本文件是接口文件，用于调用模型进行预测
# 在testimgs文件夹中放入需要预测的图片，运行main函数即可得到预测结果

import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34


class ViolenceClass:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.data_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # load class_indict
        json_path = "./class_indices.json"
        assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)

        # create model
        self.model = resnet34(num_classes=2).to(self.device)

        # load model weights
        weights_path = "./resNet34.pth"
        assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    # def predict_batch(self, img_path):
    #     assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
    #     img_path_list = [
    #         os.path.join(img_path, i)
    #         for i in os.listdir(img_path)
    #         if i.endswith(".jpg")
    #     ]

    #     # predict
    #     self.model.eval()
    #     batch_size = 8
    #     with torch.no_grad():
    #         for ids in range(0, len(img_path_list) // batch_size):
    #             img_list = []
    #             for img_path in img_path_list[
    #                 ids * batch_size : (ids + 1) * batch_size
    #             ]:
    #                 assert os.path.exists(
    #                     img_path
    #                 ), f"file: '{img_path}' dose not exist."
    #                 img = Image.open(img_path)
    #                 img = self.data_transform(img)
    #                 img_list.append(img)

    #             # batch img
    #             batch_img = torch.stack(img_list, dim=0)
    #             # predict class
    #             output = self.model(batch_img.to(self.device)).cpu()
    #             predict = torch.softmax(output, dim=1)
    #             predict_cla = torch.argmax(predict, dim=1).numpy()

    #             return [str(i) for i in predict_cla]

    def ImgtoTensor(self, img_root_path):
        assert os.path.exists(img_root_path), f"file: '{img_root_path}' dose not exist."
        img_path_list = [
            os.path.join(img_root_path, i)
            for i in os.listdir(img_root_path)
            if i.endswith(".jpg")
        ]

        img_list = []
        for img_path in img_path_list:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            img = Image.open(img_path)
            img = self.data_transform(img)
            img_list.append(img)

        # 输出是`n*3*224*224`的pytorch tensor（n是batch的大小
        # # 输出n的值
        # print(len(img_list))
        return torch.stack(img_list, dim=0)

    def classify(self, img_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor.to(self.device)).cpu()
            predict = torch.softmax(output, dim=1)
            predict_cla = torch.argmax(predict, dim=1).numpy()
            return [i for i in predict_cla]


def main():
    test_img_path = "./testimgs"
    violence_class = ViolenceClass()
    img_tensor = violence_class.ImgtoTensor(test_img_path)
    print(violence_class.classify(img_tensor))


if __name__ == "__main__":
    main()
