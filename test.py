import torch
import pandas as pd
from models import MemNet
from sklearn.model_selection import train_test_split
from Read_data import Read_data
from torchvision import transforms
import numpy as np
from scipy.stats import spearmanr
import PIL.Image

root_dir = "/imaging/myounesi/memcat/MemCat_images/MemCat/"
csv_file = "/imaging/myounesi/memcat/MemCat_data/memcat_image_data.csv"
obj_ds = pd.read_csv(csv_file)

i=5
obj_ds = obj_ds.iloc[(i-1)*2000:i*2000]
X_train, X_test= train_test_split(obj_ds, test_size=0.2, random_state=42)

scores_train = list(X_train['memorability_w_fa_correction'])
names_train = list(X_train['image_file'])
names_train1 = list(X_train['category'])
names_train2 = list(X_train['subcategory'])
scores_test = list(X_test['memorability_w_fa_correction'])
names_test = list(X_test['image_file'])
names_test1 = list(X_test['category'])
names_test2 = list(X_test['subcategory'])

X_train = [[names_train[i],names_train1[i],names_train2[i],scores_train[i]] for i in range(len(scores_train))]
X_test = [[names_test[i],names_test1[i],names_test2[i],scores_test[i]] for i in range(len(scores_test))]
X_valid = X_test[:int(len(X_test)/2)]
#scores_test = scores_test[int(len(X_test)/2):]
#X_test = X_test[int(len(X_test)/2):]



#Load pretrained MemNet
model = MemNet()
#checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
#model.load_state_dict(checkpoint["state_dict"])


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

eval_dataset = Read_data(csv_file=X_test, root_dir=root_dir, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#model.to(device)



#optimizer = torch.optim.Adam(model.parameters(),0.001)

for k in range(1):
  checkpoint = torch.load("/home/myounes9/memcat_prjct/Memnet-Pytorch/cat5/"+str(9)+".ckpt")
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()
  out = []
  true_val = []
  with torch.no_grad():
    for idx, (img, target) in enumerate(eval_loader):
        #img, target = img.to(device), target.to(device)
        output = model(img)
        out.append(output)
        true_val.append(target)

  y_pred = np.asarray(out[0])
#y_test = np.asarray(true_val[0])

  for i in range(len(out)-1):
    y_pred = np.vstack((y_pred, np.asarray(out[i+1])))
    #y_test = np.vstack((y_test, np.asarray(true_val[i+1])))
    
  #print(y_pred.shape)
  #print(np.array(scores_test).shape)
    
#np.save('faces_MemNet_AF_h.npy', y_pred)


  coef, p = spearmanr(np.array(scores_test), y_pred)
  print("Epoch " + str(k+1)+": "+str(coef))

  
            
