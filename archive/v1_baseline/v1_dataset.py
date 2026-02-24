"""
就是最简单的处理，没有划分训练集以及验证集，但是终归是应该还可以的开始。
"""
import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXRay(Dataset):

    #准备阶段
    def __init__(self, rootdir, split = 'train' , transform = None):
        #保存参数，把外面的参数传到self中。
        self.rootdir = rootdir 
        self.split = split 
        self.transform = transform 
    

        # 2. 拼接路径 
        self.data_dir = os.path.join(rootdir,split)

        # 3. 准备列表
        self.image_paths = [] 
        self.labels = [] 

        # 4. 遍历文件夹 
        # 去 NORMAL 文件夹找一遍，再去 PNEUMONIA 文件夹找一遍
        categories = ['NORMAL' , 'PNEUMONIA'] 

        for label_idx, category in enumerate(categories):
            #拼出子文件夹路径：
            category_dir = os.path.join(self.data_dir, category)

            for file_name in os.listdir(category_dir):
                #过滤一下，只看图片
                if file_name.endswith('.jpeg') or file_name.endswith('.jpg'):
                    #拼出完整绝对路径
                    full_path = os.path.join(category_dir , file_name)
                    
                    #存进我们的清单里面
                    self.image_paths.append(full_path)
                    self.labels.append(label_idx)

    #点名阶段
    def __len__(self,):
        return len(self.image_paths)
    
    #上菜阶段（最核心）
    def __getitem__(self, index):
        # 1.查清单： 根据索引index ，找出路径和标签
        img_path = self.image_paths[index]
        label = self.labels[index]

        # 2.读图
        image = Image.open(img_path)
        
        # 3.强制转RGB
        image = image.convert('RGB')

        # 4. 预处理 
        # 把 PIL 图片变成 PyTorch 的 Tensor (数字矩阵)
        if self.transform is not None :
           image = self.transform(image)

        
        return image, label 
    