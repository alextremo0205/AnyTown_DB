import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_sparse import matmul, SparseTensor

class G_filter_K(MessagePassing):
    def __init__(self, units, aggr = 'add', K = 1):
        self.aggr = aggr #add, mean, max
        self.K = K
        super(G_filter_K, self).__init__(aggr=self.aggr)  #"add", "mean" or "max"
        self.in_features, self.out_features = units
        #super(ParameterList, self).__init__()
#         self.H = nn.ParameterList([nn.Parameter(torch.randn(self.in_features, self.out_features)) for _ in range(K+1)])
        self.H = nn.ParameterList()
        for j in range(K+1):
            self.H.append(nn.Parameter(torch.randn(self.in_features, self.out_features)))
        
    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Adj):
        #First step. This is the deepest part of the MP framework.
        #Some preprocessing of the node or edge attributes may be 
        #made before the propagation
        
        #For activating message_and_aggregate() it is required that the edge index is a SparseTensor
        #edge_index = SparseTensor.from_edge_index(edge_index, edge_attr=edge_attr)
        
        accum_result = torch.matmul(x, self.H[0])
        for k in range(1,self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_attr,
                               size=None)
            accum_result += torch.matmul(x, self.H[k])
        
        return accum_result
        
        #ans = self.propagate(edge_index, x=x, edge_weight=edge_attr)
        #return ans
          
    def update(self, messages, x):
        #This is the UPDATE function, it receives the message from 
        #the nodes and the own information. 
        # Naming of the parameters impacts the function!! 
        ###   x_i != x
        # x_i has dimension (edges x 2) 
        # x   has dimension (nodes x node_features)
        
        return messages #+1*x #Dummy example of an update function involving the message and the self value

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
#         #This is the Message function. 
#         #It is useful to think about this one at a node level. Just one node i interacting with node j
#         #Notation of the parameters x_i: self information of the node, x_j: information of the neighbor
#         #edge_weight is the weight or value ei_j of the edge that connects the nodes i and j
#         msg= x_i + edge_weight
#         return msg
        #print('message')
        return edge_weight.view(-1, 1) * x_j
        #return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        #This function only is activated when the edge_index at propagate() is a SparseTensor.
        #It is equivalent to the message function, but a graph level. #adj_t is the Shift Operator in Sparse form.
        #print("message and aggregate")
        #print(adj_t)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(in_f={}, out_f={},aggr={}, K={})'.format(self.__class__.__name__,self.in_features,self.out_features, self.aggr, self.K)
        