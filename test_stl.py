import torch
import pandas as pd
from models import MemNet
from sklearn.model_selection import train_test_split
from read_cat import read_cat
from torchvision import transforms
import numpy as np
import PIL.Image
import os


root_dir = os.listdir("/home/myounes9/memcat_prjct/Memnet-Pytorch/samples/vehicles/")
print(root_dir) 

#Load pretrained MemNet
model = MemNet()
checkpoint = torch.load("/home/myounes9/memcat_prjct/Memnet-Pytorch/cat5/9.ckpt")
model.load_state_dict(checkpoint["state_dict"])

def euclidean_distance_loss(y_true, y_pred):
    return torch.sqrt(torch.sum(torch.pow((y_pred - y_true),2), -1))

mean = np.load("image_mean.npy")

transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.BILINEAR),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], mean), #Subtract average mean from image (opposite order channels)
    lambda x: x[15:242, 15:242], #Center crop
    transforms.ToTensor()
])

#train_dataset = LaMemEvalDataset(csv_file=X_train, root_dir=root_dir, transform=transform)
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

eval_dataset = read_cat(root_dir=root_dir, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)
model.eval()
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#model.to(device)



#optimizer = torch.optim.Adam(model.parameters(),0.001)

out = []

with torch.no_grad():
    for idx, img in enumerate(eval_loader):
        #img, target = img.to(device), target.to(device)
        output = model(img)
        out.append(output)
        if idx%50 == 0:
          print(idx)



y_pred = np.asarray(out[0])
#y_test = np.asarray(true_val[0])
print(len(out))
print("****")

for i in range(len(out)-1):
    y_pred = np.vstack((y_pred, np.asarray(out[i+1])))
    #y_test = np.vstack((y_test, np.asarray(true_val[i+1])))
    
print(y_pred)
    
#np.save('horses_ft_mems.npy', y_pred)


  
            
