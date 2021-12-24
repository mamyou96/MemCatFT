import torch
import pandas as pd
from models import MemNet
from sklearn.model_selection import train_test_split
from Read_data import Read_data
from torchvision import transforms
import numpy as np
import PIL.Image

root_dir = "/imaging/myounesi/memcat/MemCat_images/MemCat/"
csv_file = "/imaging/myounesi/memcat/MemCat_data/memcat_image_data.csv"
obj_ds = pd.read_csv(csv_file)

i=2
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
X_test = X_test[int(len(X_test)/2):]

loss = torch.nn.MSELoss()
loss1 = torch.nn.MSELoss()

#Load pretrained MemNet
model = MemNet()
checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
model.load_state_dict(checkpoint["state_dict"])

def euclidean_distance_loss(y_true, y_pred):
    #return torch.cdist(y_true, y_pred, p=2)
    return torch.sqrt(torch.sum(torch.pow((y_pred - y_true),2),-1))
    


mean = np.load("image_mean.npy")

transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], mean), #Subtract average mean from image (opposite order channels)
    lambda x: x[15:242, 15:242], #Center crop
    transforms.ToTensor()
])

train_dataset = Read_data(csv_file=X_train, root_dir=root_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

eval_dataset = Read_data(csv_file=X_valid, root_dir=root_dir, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)



optimizer = torch.optim.Adam(model.parameters(),0.00001)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)

for epoch in range(50):
  train_loss = 0.0
  valid_loss = 0.0
  model.train()
  for idx, (img, target) in enumerate(train_loader):
    img, target = img.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(img)
    output = torch.reshape(output, (-1,))
    #print(output.size())
    out = loss(output, target)
    out.backward()
    optimizer.step()
    train_loss += out.item() * img.size(0)
  model.eval()
  for idx, (img, target) in enumerate(eval_loader):
    img, target = img.to(device), target.to(device)
    output = model(img)
    output = torch.reshape(output, (-1,))
    loss_ = loss1(output, target)
    valid_loss += loss_.item() * img.size(0)

  train_loss = train_loss/len(train_loader)
  valid_loss = valid_loss/len(eval_loader)
  scheduler.step(valid_loss)
  
  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
  checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
  }
  torch.save(checkpoint,"/home/myounes9/memcat_prjct/Memnet-Pytorch/temp1/"+str(epoch)+'.ckpt')
  
            
