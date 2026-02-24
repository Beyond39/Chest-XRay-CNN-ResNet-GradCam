import torch
import cv2
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

from model import ChestCNN
from model import ChestResNet18

# 1. 导入你刚刚写好的模型
from model import ChestResNet18

def main():
    # 2. 准备设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestResNet18(num_classes=2,pretrained=True)
    
    # 【重点】加载你训练好的最佳权重！
    model.load_state_dict(torch.load('archive/v3_resnet18/v3_model.pth', map_location=device))
    model.to(device)
    model.eval() # 必须切换到测试模式

    # # 3. 指定 Grad-CAM 要观察的“目标层”
    # # 对于 ResNet-18，我们通常看最后一块卷积层提取的最高级特征
    # target_layers = [model.resnet.layer4[-1]]

    target_layers = [model.resnet.layer4[-1]]
    
    # 4. 处理一张图片
    image_path = "data/processed/val/PNEUMONIA/person1_virus_9.jpeg" # 换成你的真实图片路径
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   # BGR 转 RGB
    rgb_img = cv2.resize(rgb_img, (224, 224))         # 调整大小
    rgb_img_float = np.float32(rgb_img) / 255         # 归一化到 [0, 1] 供可视化使用

    # 5. 图片预处理 (必须和训练时一模一样！)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 转换图片并增加 Batch 维度: [1, 3, 224, 224]
    input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

    # 6. 初始化 Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 7. 指定你要看哪个类别的热力图（比如肺炎是类别 1）
    # 如果肺炎对应的 index 是 1：
    targets = [ClassifierOutputTarget(1)]

    # 8. 生成热力图 (Grayscale CAM)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 9. 将热力图叠加到原图上
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    # 10. 保存图片！这正是你要在 README 里展示的图！
    cv2.imwrite('cam2_output_pneumonia(1).jpg', visualization[:, :, ::-1])
    print(">>> Grad-CAM 生成成功！请在当前目录下查看 cam_output_pneumonia.jpg")

if __name__ == '__main__':
    main()



# -----------------------------
# 设备与模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestCNN(num_classes=2).to(device)
model.load_state_dict(torch.load('archive/v2_baseline_2/v2_model_2.pth', map_location=device))
model.eval()

# -----------------------------
# 指定 Grad-CAM 目标层
# -----------------------------
target_layer = model.conv4

# -----------------------------
# 图片预处理
# -----------------------------
image_path = "data/processed/val/PNEUMONIA/person53_virus_108.jpeg"
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # BGR -> RGB
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img_float = np.float32(rgb_img) / 255

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = transform(Image.fromarray(rgb_img)).unsqueeze(0).to(device)

# -----------------------------
# 初始化 Grad-CAM
# -----------------------------
cam = GradCAM(model=model, target_layers=[target_layer])

# 指定类别（1 = 肺炎 / 0 = 正常）
targets = [ClassifierOutputTarget(1)]  # 真阳性示例
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
cv2.imwrite('cam2_output_true_positive.jpg', visualization[:, :, ::-1])
print(">>> 真阳性 Grad-CAM 图生成成功！保存为 cam2_output_true_positive.jpg")

# # 如果要生成真阴性，可以修改 targets = [ClassifierOutputTarget(0)]
# targets_neg = [ClassifierOutputTarget(0)]
# grayscale_cam_neg = cam(input_tensor=input_tensor, targets=targets_neg)[0, :]
# visualization_neg = show_cam_on_image(rgb_img_float, grayscale_cam_neg, use_rgb=True)
# cv2.imwrite('cam_output_true_negative.jpg', visualization_neg[:, :, ::-1])
# print(">>> 真阴性 Grad-CAM 图生成成功！保存为 cam_output_true_negative.jpg")