import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Modelo GCN simples
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=7):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv3(x, edge_index)

# Carregar Cora
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Máscara de treino/validação simples (usamos train_mask padrão)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Non-IID partition: skew por classes (3 clients)
labels = data.y.numpy()
unique_classes = np.unique(labels)

# Client 1: domina classes 0-2
client1_mask = np.isin(labels, [0,1,2])
# Client 2: domina classes 3-4
client2_mask = np.isin(labels, [3,4])
# Client 3: domina classes 5-6 + alguns misto
client3_mask = np.isin(labels, [5,6]) | (np.random.rand(len(labels)) < 0.3)

# Garantir sobreposição mínima e cobertura
clients_masks = [client1_mask, client2_mask, client3_mask]

# Train local
def train_local(data, mask, epochs=200):
    model = GCN(dataset.num_features, output_dim=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    acc = accuracy_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy())
    return acc

# Federated
def federated_train(clients_masks, rounds=50):
    global_model = GCN(dataset.num_features, output_dim=dataset.num_classes)
    history = []
    for r in range(rounds):
        local_models = []
        for mask in clients_masks:
            local_model = GCN(dataset.num_features, output_dim=dataset.num_classes)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.01, weight_decay=5e-4)
            local_model.train()
            for _ in range(20):  # local epochs
                optimizer.zero_grad()
                out = local_model(data.x, data.edge_index)
                loss = F.cross_entropy(out[mask], data.y[mask])
                loss.backward()
                optimizer.step()
            local_models.append(local_model.state_dict())
        
        # FedAvg
        avg_state = {}
        for key in global_model.state_dict():
            avg_state[key] = torch.mean(torch.stack([m[key] for m in local_models]), dim=0)
        global_model.load_state_dict(avg_state)
        
        # Avg acc
        global_model.eval()
        out = global_model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        accs = [accuracy_score(data.y[m].cpu().numpy(), pred[m].cpu().numpy()) for m in clients_masks]
        avg_acc = np.mean(accs)
        history.append(avg_acc)
        print(f"Round {r+1}/50: Avg Acc = {avg_acc:.4f}")
    
    return history

# Main
if __name__ == "__main__":
    print("=== Local-only ===")
    local_accs = [train_local(data, mask) for mask in clients_masks]
    for i, acc in enumerate(local_accs):
        print(f"Cliente {i+1}: Acc {acc:.4f}")
    print(f"Média Local: {np.mean(local_accs):.4f}\n")
    
    print("=== Federated ===")
    history = federated_train(clients_masks)
    
    plt.plot(history)
    plt.title("Federated Learning Curve on Cora (Avg Accuracy)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.savefig("cora_fed_curve.png")
    plt.show()
    
    print(f"Final Federated Average: {history[-1]:.4f}")
