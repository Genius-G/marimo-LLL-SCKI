from help_functions import daten
import random
import torch
import torch.nn as nn
from copy import deepcopy


daten, _ = daten()

print(f"Wir haben {len(daten)} Datenpunkte zur Verfügung.")

# Hier werden die Daten durchmischt.
daten = random.sample(daten, k=len(daten))

# Die Daten werden in Trainings- und Testdaten aufgeteilt (Verhältnis 8:2).
trainings_daten = daten[:400]
test_daten = daten[400:]
del daten

class Net(nn.Module):
    def __init__(self, num_in, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in, 2)
        self.fc2 = nn.Linear(2, 2)
        self.relu=torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
    
class Net2(nn.Module):
    def __init__(self, num_in, num_out):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(num_in, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, num_out)
        self.relu=torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

X_train = torch.tensor([])
Y_train = torch.tensor([])
X_test = torch.tensor([])
Y_test = torch.tensor([])
train_labels = []
test_labels = []

for datenpunkt in trainings_daten:
    koordinaten = torch.tensor([ [ [datenpunkt[0][0], datenpunkt[0][1]] ] ])
    X_train = torch.cat( ( X_train, koordinaten), dim=0 )
    Y_train = torch.cat( ( Y_train, torch.tensor( [[datenpunkt[1]]] ) ), dim=0)
    train_labels.append(datenpunkt[1])
    
for datenpunkt in test_daten:
    koordinaten = torch.tensor([[datenpunkt[0][0], datenpunkt[0][1]]])
    X_test = torch.cat( ( X_test, koordinaten), dim=0 )
    Y_test = torch.cat( ( Y_test, torch.tensor([[datenpunkt[1]]])), dim=0)
    test_labels.append(datenpunkt[1])
    
# Y_train = torch.tensor(train_labels)
# Y_test = torch.tensor(test_labels)
labels_train = torch.tensor([])
for label in Y_train:
    if label.item() == 0:
        labels_train = torch.concat( ( labels_train, torch.tensor([[1,0]]) ), dim=0)
    else:
        labels_train = torch.concat( ( labels_train, torch.tensor([[0,1]]) ), dim=0)
        
labels_test = torch.tensor([])
for label in Y_test:
    if label.item() == 0:
        labels_test = torch.concat( ( labels_test, torch.tensor([[1,0]]) ), dim=0)
    else:
        labels_test = torch.concat( ( labels_test, torch.tensor([[0,1]]) ), dim=0)
        
Y_train = labels_train
Y_test = labels_test

print(Y_train.shape)
    
del trainings_daten
del test_daten
del datenpunkt
del koordinaten

def train(model, train_data, train_labels, lr):
    model.train(True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = loss_fn(outputs, train_labels)
    print('#####', loss)
    loss.backward()
    optimizer.step()
    print()
        
def test(model, test_data, test_labels):
    model.train(False)
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    outputs = model(test_data)
    loss = loss_fn(outputs, test_labels)
    correctly_classified = 0
    for idx, x in enumerate(outputs):
        if torch.argmax(x) == test_labels[idx]:
            correctly_classified += 1
    print(f"Correctly classified: {correctly_classified} / {outputs.shape[0]}")
    return loss

def get_model(model):
      return deepcopy(model)

epochs = 50
net = Net2(2,2)
lr = 0.01
loss_min = 100000000
counter = 0
best_model = get_model(net)

print(f"\n\n#### X_train.shape={X_train.shape} \n\n")
print(f"\n#### Y_train.shape={Y_train.shape} \n\n")

for i in range(epochs):
    # Training
    train(net, X_train, Y_train, lr)
    # Test
    loss = test(net, X_test, Y_test)
    print(f"Loss={loss}")
    
    # Falls der Loss nach drei Durchläufen nicht wesentlich kleiner wird 
    # als der bisher minimaler Loss, dann wird die Lernrate um den Faktor 
    # 10 verkleinert.
    
    if loss - loss_min > 0.1:
        print("Diff=", loss - loss_min)
        counter += 1
        if counter == 3:
            lr /= 10
            counter = 0
            model = best_model
            print("#### New learning rate", lr)
            
    # Wenn der aktuelle Loss minimal ist, dann wird dieser in der Variablen 
    # 'loss_min' gespeichert und das beste Modell in der Variablen 'best_model'.
    
    else:
        loss_min = loss
        counter = 0
        best_model = get_model(net)