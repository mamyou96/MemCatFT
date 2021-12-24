import torch
from models import MemNet
from LaMemDataset import LaMemEvalDataset
from torchvision import transforms
import numpy as np
import PIL.Image

root_dir = "/imaging/myounesi/memcat/MemCat_images/MemCat/"
csv_file = "/imaging/myounesi/memcat/MemCat_data/memcat_image_data.csv"

#Load pretrained MemNet
model = MemNet()
checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
model.load_state_dict(checkpoint["state_dict"])


mean = np.load("image_mean.npy")

transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.BILINEAR),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], mean), #Subtract average mean from image (opposite order channels)
    lambda x: x[15:242, 15:242], #Center crop
    transforms.ToTensor()
])

#cat 1 : Animal
#cat 2 : Food
#cat 3: Landscape
#cat 4: Sports
#cat 5: Vehicle


eval_dataset = LaMemEvalDataset(csv_file=csv_file, root_dir=root_dir, cat=5, transform=transform)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)
model.eval()

#real_vals = np.array([0,0])
#predicted_vals = np.array([0,0])
#with torch.no_grad():
#    for idx, (img, target) in enumerate(eval_loader):
#        output = model(img)
#        #print(output.shape)
#        predicted_vals = np.vstack((predicted_vals,np.array(output)))
#        real_vals = np.vstack((real_vals,np.array(target)))
#
#real_vals = real_vals[1:,:]
#predicted_vals = predicted_vals[1:,:]
#from scipy.stats import spearmanr
#coef, p = spearmanr(real_vals, predicted_vals)
#print(coef)


#######
out = []
true_val = []
with torch.no_grad():
    for idx, (img, target) in enumerate(eval_loader):
        #img, target = img.to(device), target.to(device)
        target = torch.flatten(target)
        output = model(img)
        out.append(output)
        true_val.append(target)



y_pred = np.asarray(out[0])
y_test = np.asarray(true_val[0])
for i in range(len(out)-1):
    y_pred = np.vstack((y_pred, np.asarray(out[i+1])))
    y_test = np.vstack((y_test, np.asarray(true_val[i+1])))

y_test = y_test.reshape(2000,1)    
print(y_pred.shape)
print(y_test.shape)    

from scipy.stats import spearmanr
coef, p = spearmanr(y_test, y_pred)
print(coef)
