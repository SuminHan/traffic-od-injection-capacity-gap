"""
Node2Vec spatial embedding (SE) for GMAN, ported from libcity_ref's
libcity/model/road_representation/Node2Vec.py (biased 2nd-order random walk + alias sampling,
Grover & Leskovec KDD 2016 -- the exact algorithm the original GMAN paper uses for its SE).
Runs over our own sensor kNN point graphs (volume_point_graph.npz / speed_point_graph.npz,
weighted, undirected) instead of a road-network graph, since that's the graph GMANBaseline's
spatial attention actually operates on. Produces a (N, emb_dim) embedding matrix per task to
REPLACE the plain nn.Embedding(N, emb_dim) node-identity lookup GMANBaseline currently learns
from scratch -- same shape, so it's a clean architecture-preserving swap.
"""
import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

GTS = "/home/ncrc/work/gts"
EMB_DIM = 16
P, Q = 2, 1  # return/in-out params (same defaults as libcity_ref)
NUM_WALKS, WALK_LENGTH, WINDOW = 100, 80, 10


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)
    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        (smaller if q[kk] < 1.0 else larger).append(kk)
    while smaller and larger:
        small, large = smaller.pop(), larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        (smaller if q[large] < 1.0 else larger).append(large)
    return J, q


def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    return kk if np.random.rand() < q[kk] else J[kk]


class Node2VecGraph:
    def __init__(self, G, p, q):
        self.G, self.p, self.q = G, p, q

    def walk(self, length, start):
        walk = [start]
        while len(walk) < length:
            cur = walk[-1]
            nbrs = sorted(self.G.neighbors(cur))
            if not nbrs:
                break
            if len(walk) == 1:
                walk.append(nbrs[alias_draw(*self.alias_nodes[cur])])
            else:
                prev = walk[-2]
                walk.append(nbrs[alias_draw(*self.alias_edges[(prev, cur)])])
        return walk

    def simulate_walks(self, num_walks, length):
        walks = []
        nodes = list(self.G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for n in nodes:
                walks.append(self.walk(length, n))
        return walks

    def get_alias_edge(self, src, dst):
        probs = []
        for nbr in sorted(self.G.neighbors(dst)):
            w = self.G[dst][nbr]["weight"]
            if nbr == src:
                probs.append(w / self.p)
            elif self.G.has_edge(nbr, src):
                probs.append(w)
            else:
                probs.append(w / self.q)
        s = sum(probs)
        return alias_setup([p / s for p in probs])

    def preprocess(self):
        self.alias_nodes = {}
        for n in self.G.nodes():
            probs = [self.G[n][nbr]["weight"] for nbr in sorted(self.G.neighbors(n))]
            s = sum(probs)
            self.alias_nodes[n] = alias_setup([p / s for p in probs])
        self.alias_edges = {}
        for e in self.G.edges():
            self.alias_edges[e] = self.get_alias_edge(e[0], e[1])
            self.alias_edges[(e[1], e[0])] = self.get_alias_edge(e[1], e[0])


def build(task, graph_path):
    g = np.load(graph_path, allow_pickle=True)
    A = g["A"]
    n = A.shape[0]
    nxg = nx.from_numpy_array(A, create_using=nx.Graph())
    for u, v, d in nxg.edges(data=True):
        if d.get("weight", 0) <= 0:
            d["weight"] = 1e-6
    n2v = Node2VecGraph(nxg, P, Q)
    n2v.preprocess()
    walks = n2v.simulate_walks(NUM_WALKS, WALK_LENGTH)
    walks = [[str(x) for x in w] for w in walks]
    model = Word2Vec(walks, vector_size=EMB_DIM, window=WINDOW, min_count=0, sg=1, hs=0,
                      workers=8, epochs=5)
    emb = np.zeros((n, EMB_DIM), dtype=np.float32)
    missing = 0
    for i in range(n):
        if str(i) in model.wv:
            emb[i] = model.wv[str(i)]
        else:
            missing += 1
    np.savez(f"{GTS}/node2vec_se_{task}.npz", emb=emb)
    print(f"{task}: SE {emb.shape}, {missing} isolated nodes fell back to zero vector")


build("volume", f"{GTS}/volume_point_graph.npz")
build("speed", f"{GTS}/speed_point_graph.npz")
