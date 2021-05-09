import json
from sklearn import model_selection
from chatbot_alpha import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
with open('data.json', 'r') as file:
    data = json.load(file)

## Important resource: https://www.youtube.com/watch?v=1lwddP0KUEg&t=1229s
## https://www.youtube.com/watch?v=k1SzvvFtl4w&t=471s
## the videos shows the way to convert 'text' to 'value', so that we can use the data in machine learning
### prepare the tranining dataset
all_words = []
tags = []
xy = []
## load the data and split sentences to individual words
for intent in data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = [',', '.', '?', '!', ';']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []
## convert this words to numerical valaue
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


x = np.array(x_train)
y = np.array(y_train)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20, random_state=1)


print("***********************")
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

class TestDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(x_test)
        self.x_data = x_test
        self.y_data = y_test

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


## initialize hyper-parameters
batch_size = 7
hidden_size = 9
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 200

dataset = ChatDataSet()
testdataset = TestDataSet()

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## build a CNN model 
model = nn.Sequential(
    nn.Linear(input_size, hidden_size), 
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size), 
)


## loss function I use 
criterion = nn.CrossEntropyLoss()

##using pytoch optimizer to optimize my hyper-prameter
### including input_size, hidden_size, output_size, learning rate(lr)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

## prepare pd for matlab plot
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
trainLoss_arr = []
testLoss_arr = []

### start training my CNN model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        #training set
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        for (words_test, labels_test) in test_loader:
            words_test = words_test.to(device)
            labels_test = labels_test.to(dtype=torch.long).to(device)
            outputs_test = model(words_test)
            loss_test = criterion(outputs_test, labels_test)
        print(f'epoch {epoch+1}/{num_epochs}, loss_trainingSet = {loss.item():.4f}, loss_testSet = {loss_test.item():.4f}')
        trainLoss_arr.append((f'{loss.item():.4f}'))
        testLoss_arr.append((f'{loss_test.item():.4f}'))
print(f'final loss: loss_trainingSet = {loss.item():.4f}, loss_testingSet = {loss_test.item():.4f}')


## data schema
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

## save the model to file
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

d = {'train': trainLoss_arr, 'test':testLoss_arr}
df = pd.DataFrame(data=d)
df=df.astype(float)
print(df)
df.plot()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
