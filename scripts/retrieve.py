import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import json
import gensim
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from pcst_fast import pcst_fast
import re
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# Model configuration
pretrained_repo = 'sentence-transformers/all-roberta-large-v1'
batch_size = 1024  # Adjust the batch size as needed
model_name = 'sbert'

# Path configuration
path = '.'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

# Milvus configuration
CLUSTER_ENDPOINT = "https://in03-7a5f9d2a1aa84ef.serverless.gcp-us-west1.cloud.zilliz.com"
API_KEY = "a73c79fb1924d05aeb410abc0d5669293cc33be37a123953be640725aa42198ef5c1e499cc07f231977c742ad6e6977c6eddec05"
milvus_client = MilvusClient(uri=CLUSTER_ENDPOINT, token=API_KEY)
COLLECTION_NAME = "graph_embeddings"
VECTOR_DIM = 1024  # SBERT default
METRIC_TYPE = "COSINE"

# Dataset class for text embedding
class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data

# Sentence Transformer model
class Sentence_Transformer(nn.Module):
    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

def load_sbert():
    model = Sentence_Transformer(pretrained_repo)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

def sbert_text2embedding(model, tokenizer, device, text):
    if len(text) == 0:
        return torch.zeros((0, 1024))

    encoding = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])
            all_embeddings.append(embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    return all_embeddings

# Model loading functions
load_model = {
    'sbert': load_sbert,
}

load_text2embedding = {
    'sbert': sbert_text2embedding,
}

def retrieval_via_pcst(graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False)+'\n'+e.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))

    return data, desc




def retreival(question, k=3):
    load_model = {
        'sbert': load_sbert,
    }
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]
    # Encode question
    q_emb = text2embedding(model, tokenizer, device, [question])[0]

    # Ensure collection is loaded before search
    try:
        milvus_client.load_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error loading collection: {e}")
        return [], []

    # Perform similarity search in Milvus
    search_results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[q_emb.tolist()],
        limit=k,
        search_params={"metric_type": METRIC_TYPE, "params": {}},
        output_fields=["graph_idx"]
    )

    # Extract graph indices from results
    hits = search_results[0]
    graph_indices = [hit["entity"]["graph_idx"] for hit in hits]

    # Collect subgraphs and descriptions
    sub_graphs = []
    descriptions = []

    for index in tqdm(graph_indices, desc="Retrieving subgraphs"):
        nodes_path = f'{path_nodes}/{index}.csv'
        edges_path = f'{path_edges}/{index}.csv'
        graph_path = f'{path_graphs}/{index}.pt'

        if not (os.path.exists(nodes_path) and os.path.exists(edges_path) and os.path.exists(graph_path)):
            print(f"Missing data for graph {index}")
            continue

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        if len(nodes) == 0:
            print(f"Empty graph at index {index}")
            continue

        graph = torch.load(graph_path)

        # Apply your custom retrieval logic (must be defined elsewhere)
        subg, desc = retrieval_via_pcst(
            graph=graph,
            q_emb=q_emb,
            textual_nodes=nodes,
            textual_edges=edges,
            topk=3,
            topk_e=5,
            cost_e=0.5
        )

        sub_graphs.append(subg)
        descriptions.append(desc)

    return sub_graphs, descriptions

