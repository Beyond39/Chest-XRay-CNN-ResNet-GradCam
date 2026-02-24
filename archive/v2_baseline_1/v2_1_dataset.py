import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXRay(Dataset):

    def __init__(self, rootdir, split = 'train' , transform = None):
        self.rootdir = rootdir 
        self.split = split 
        self.transform = transform 

        self.data_dir = os.path.join(rootdir,split)

        self.image_paths = [] 
        self.labels = [] 

        categories = ['NORMAL' , 'PNEUMONIA'] 

        for label_idx, category in enumerate(categories):
            category_dir = os.path.join(self.data_dir, category)

            for file_name in os.listdir(category_dir):
                if file_name.endswith('.jpeg') or file_name.endswith('.jpg'):
                    full_path = os.path.join(category_dir , file_name)
                    self.image_paths.append(full_path)
                    self.labels.append(label_idx) #0或1

    def __len__(self,):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transform is not None :
           image = self.transform(image)
        
        return image, label 
    