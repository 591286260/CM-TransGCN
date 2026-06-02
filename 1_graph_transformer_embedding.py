import os
import warnings

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TransformerConv

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
NUM_TRANS_LAYERS = 1
NUM_GCN_LAYERS = 1
NUM_HEADS = 4
HIDDEN_CHANNELS = 64
LEARNING_RATE = 0.001
EPOCHS = 300
DROPOUT = 0.0

# -----------------------------
# File paths
# -----------------------------
INTERACTION_FILE = "circAtlas 3.0/interaction.csv"
FEATURE_FILE = "circAtlas 3.0/Se_vector64.csv"
OUTPUT_FILE = "node_embeddings.csv"


# -----------------------------
# Load circRNA-miRNA interaction data
# -----------------------------
df_interactions = pd.read_csv(
    INTERACTION_FILE,
    header=None,
    names=["circRNA", "miRNA"]
)

# -----------------------------
# Construct graph
# -----------------------------
G = nx.Graph()

for _, row in df_interactions.iterrows():
    circRNA = row["circRNA"]
    miRNA = row["miRNA"]

    G.add_node(circRNA, type="circRNA")
    G.add_node(miRNA, type="miRNA")
    G.add_edge(circRNA, miRNA)

# -----------------------------
# Load node feature matrix
# -----------------------------
df_features = pd.read_csv(FEATURE_FILE, header=None)
df_features.columns = ["id"] + [
    f"feature_{i}" for i in range(1, df_features.shape[1])
]

# Convert node features to dictionary
node_features = df_features.set_index("id").T.to_dict("list")

# -----------------------------
# Convert NetworkX graph to PyG format
# -----------------------------
node_to_index = {node: i for i, node in enumerate(G.nodes)}

edge_index = torch.tensor(
    [[node_to_index[u], node_to_index[v]] for u, v in G.edges],
    dtype=torch.long
).t().contiguous()

# Explicit bidirectional edges for the undirected graph
edge_index_rev = edge_index[[1, 0], :]
edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

data = Data(edge_index=edge_index)

# -----------------------------
# Initialize node features
# -----------------------------
num_nodes = G.number_of_nodes()
feature_dim = len(next(iter(node_features.values())))

data.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)

for node, features in node_features.items():
    if node in node_to_index:
        data.x[node_to_index[node]] = torch.tensor(features, dtype=torch.float)

print("Node feature matrix shape:", data.x.shape)
print("Example node features:", data.x[:5])


# -----------------------------
# Transformer-first graph model
# TransformerConv × N -> GCNConv × M
# -----------------------------
class GraphTransformerBottom(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_trans_layers: int = 1,
        num_gcn_layers: int = 1,
        num_heads: int = 4,
        concat: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        assert num_trans_layers >= 1, "num_trans_layers must be >= 1"
        assert num_gcn_layers >= 1, "num_gcn_layers must be >= 1"
        assert num_heads >= 1, "num_heads must be >= 1"

        self.dropout = dropout

        # TransformerConv layers
        self.trans_layers = torch.nn.ModuleList()

        self.trans_layers.append(
            TransformerConv(
                in_channels,
                hidden_channels,
                heads=num_heads,
                concat=concat,
                dropout=dropout
            )
        )

        for _ in range(num_trans_layers - 1):
            self.trans_layers.append(
                TransformerConv(
                    hidden_channels,
                    hidden_channels,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout
                )
            )

        # GCNConv layers
        self.gcn_layers = torch.nn.ModuleList()

        if num_gcn_layers == 1:
            self.gcn_layers.append(
                GCNConv(hidden_channels, out_channels)
            )
        else:
            self.gcn_layers.append(
                GCNConv(hidden_channels, hidden_channels)
            )

            for _ in range(num_gcn_layers - 2):
                self.gcn_layers.append(
                    GCNConv(hidden_channels, hidden_channels)
                )

            self.gcn_layers.append(
                GCNConv(hidden_channels, out_channels)
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Transformer blocks
        for layer in self.trans_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)

            if self.dropout > 0:
                x = torch.nn.functional.dropout(
                    x,
                    p=self.dropout,
                    training=self.training
                )

        # GCN blocks
        for i, layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)

            # No activation on the final layer for feature reconstruction
            if i != len(self.gcn_layers) - 1:
                x = torch.relu(x)

        return x


# -----------------------------
# Build model
# -----------------------------
in_channels = feature_dim
out_channels = feature_dim

model = GraphTransformerBottom(
    in_channels=in_channels,
    out_channels=out_channels,
    hidden_channels=HIDDEN_CHANNELS,
    num_trans_layers=NUM_TRANS_LAYERS,
    num_gcn_layers=NUM_GCN_LAYERS,
    num_heads=NUM_HEADS,
    concat=False,
    dropout=DROPOUT
).to(device)

data = data.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = torch.nn.MSELoss()

# -----------------------------
# Model training
# -----------------------------
model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out, data.x)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

# -----------------------------
# Export node embeddings
# -----------------------------
model.eval()

with torch.no_grad():
    node_embeddings = model(data).detach().cpu().numpy()

node_embeddings_df = pd.DataFrame(
    node_embeddings,
    index=list(G.nodes)
)

node_embeddings_df.to_csv(
    OUTPUT_FILE,
    header=False
)

print(f"Node embeddings have been saved to: {OUTPUT_FILE}")