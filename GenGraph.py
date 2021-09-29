import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

fname = sys.argv[1]
df = pd.read_csv(fname, index_col=0)
headers = list(df.columns)
headers = [x.replace("Close_", "").upper() for x in headers]
mat = df.to_numpy()
n = df.shape[0]
edges = []
not_include = ["CAD2USD", "AUD2USD", "C"]

G = nx.DiGraph()

for i in range(n):
    for j in range(n):
        if mat[i, j] == 1 and headers[i] not in not_include and headers[j] not in not_include:
            edges.append((headers[i], headers[j]))

G.add_edges_from(edges)
pos = nx.spring_layout(G, k=3)
nx.draw(G, pos=pos, node_size=1000, with_labels=True)
plt.savefig(fname.replace(".csv", ".png"))
