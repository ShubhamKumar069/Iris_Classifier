import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ReLU(x):
    return np.maximum(0, x)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

species_to_label = {
    "setosa" : 0,
    "versicolor" : 1,
    "virginica" : 2
}

data['species'] = data['species'].map(species_to_label)

X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = data[["species"]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long).squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 3) 
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.01)

epochs = 2000

for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"EPOCH : {epoch} | LOSS : {loss}")

with torch.no_grad():
    y_pred = model(X_test)
    predicted_classes = torch.argmax(y_pred, dim=1)
    accuracy = (predicted_classes == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")