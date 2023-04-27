''' Student: Marisol Antunez '''

import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchviz import make_dot

# -------- Preparing the Dataset --------
n_epochs = 129
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.01
momentum = 0.5
log_interval = 14

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# DataLoader for the Greek data set
math_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder('processed_dataset',
                                    transform = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),]) ),
    batch_size = batch_size_train, shuffle = True)  

math_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder('processed_dataset',
                                    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),] ) ),
    batch_size = batch_size_test, shuffle = True)  

examples = enumerate(math_train)
batch_idx, (example_data, example_targets) = next(examples)

print(batch_idx)
print(example_data.shape)


# -------- Building Network --------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 14)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# Initialize the network and optimizer
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
print(network)


# -------- Training the Model --------
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(math_train.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  # iterate over all training data once per epoch
  for batch_idx, (data, target) in enumerate(math_train):
    optimizer.zero_grad()

    # produce the output of our network (forward pass)
    output = network(data) 
    loss = F.nll_loss(output, target)
    loss.backward() 
    optimizer.step() 

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(math_train.dataset),
        100. * batch_idx / len(math_train), loss.item()))

      # compute a negative log-likelihodd loss between the output and the ground truth label  
      train_losses.append(loss.item())
      train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(math_train.dataset)))

      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')


def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): 
    for data, target in math_test:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(math_test.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(math_test.dataset),
    100. * correct / len(math_test.dataset)))


# # Start traininng
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()


# # -------- Plot Prediction Outputs --------
examples = enumerate(math_test)
batch_idx, (example_data_test, example_targets_test) = next(examples)

with torch.no_grad():
  output = network(example_data_test)

# Plot predictions
labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9','add', 'div', 'mul', 'sub']
plt.figure(figsize=(13,6))
for i in range(27):
  plt.subplot(5,6,i+1)
  plt.tight_layout()
  plt.imshow(example_data_test[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(labels[
    output.data.max(1, keepdim=True)[1][i].item()]))   
  plt.xticks([])
  plt.yticks([])
plt.show()  