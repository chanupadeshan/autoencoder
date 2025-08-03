import torch, torch.nn as nn #  PyTorch and its neural network module
import torchvision #  torchvision for datasets , models and image tools
import torchvision.transforms as transforms # use to preproecss the images
from torch.utils.data import DataLoader # DataLoader to load the dataset in mini-batches
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 0.0005
num_epochs = 100

transform = transforms.ToTensor() # convert image to pytorch tensor
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) # Downloading the MNIST dataset

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) # DataLoader to load the dataset in mini-batches and shuffle the data



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 , 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8, 4)  # Bottleneck layer

        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28), # Output layer
            nn.Sigmoid() 
        )

    def forward(self,x):
        x = x.view(x.size(0),-1) # flatten the image : (batch_size, 1, 28, 28) -> (batch_size, 28*28)
        encoded = self.encoder(x) # compress the image
        decoded = self.decoder(encoded) # rebuild the image
        return decoded
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

class EarlyStopping:
    def __init__(self,patience=5,delta=0):
        # number of epochs with no improvement after which training will be stopped
        self.patience = patience 
        # minimum amount that counts as real improvement
        # if the loss is not improved by at least delta, it is not considered as improvement
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self,loss):
        # if this is the first epoch , set the current loss as the best loss
        if self.best_loss is None:
            self.best_loss = loss

        elif loss < self.best_loss - self.delta:
            self.counter +=1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = loss
            self.counter = 0
            self.early_stop = False

early_stopping = EarlyStopping(patience=5,delta=0.001)

for epoch in range(num_epochs):
    total_loss = 0
    for img,_ in train_loader:
        img = img.to(device) # move the image to the device (GPU or CPU)
        output = model(img)
        loss = criterion(output,img.view(img.size(0),-1)) # compare the output with the original imag
        
        
        optimizer.zero_grad() # remove the gradients from the previous step
        loss.backward() # calculate the gradients
        optimizer.step() # update the weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    early_stopping(avg_loss)

    # if early_stopping.early_stop:
    #     print("Early stopping triggered")
    #     break



# Visualization
model.eval()

dataiter = iter(train_loader)
'''
create an iterator from the DataLoader to get a batch of images

what is the iterator?
Python object that allows to loop over data one item (one batch) at a time
'''
images, _ = next(dataiter) # next : get the one batch
images = images.to(device)


with torch.no_grad():
    outputs = model(images)


num_images = 10
for i in range(num_images):
    #original image
    # 2 : number of rows
    # num_images : number of columns
    # 1 + i : place of the image in the first row
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(images[i].cpu().view(28, 28), cmap='gray')
    plt.axis('off')
    plt.title('Original')

    #reconstructed image
    # 2 : number of rows
    # num_images : number of columns
    # i + 1 + num_images : place of the image in the second row ; starts from position 11 to 20
    ax = plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(outputs[i].cpu().view(28, 28), cmap='gray')
    plt.axis('off')
    plt.title('Reconstructed')

plt.show()

