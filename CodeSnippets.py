
def update(self, inputs: Tensor) -> Tensor:
	r"""Updates node embeddings in analogy to
	:math:`\gamma_{\mathbf{\Theta}}` for each node
	:math:`i \in \mathcal{V}`.
	Takes in the output of aggregation as first argument and any argument
	which was initially passed to :meth:`propagate`.
	"""
	return inputs

#########################################################################################################

class G_layer_Alex(MessagePassing):
    def __init__(self):
        self.aggr = 'add' #add, mean, max
        super(G_layer_Alex, self).__init__(aggr=self.aggr)  # "Add" aggregation
        #"add", "mean" or "max"
    
    def forward(self, x, edge_attr, edge_index):
#         x, edge_attr, edge_index = data.x, data.weight, data.edge_index
        
        #x = (x, x)
        
#         if isinstance(x, Tensor):
#             #x: OptPairTensor = (x, x)
#             x = (x, x)

        # Node and edge feature dimensionalites need to match.
#         if isinstance(edge_index, Tensor):
#             if edge_attr is not None:
#                 assert x[0].size(-1) == edge_attr.size(-1)
#         elif isinstance(edge_index, SparseTensor):
#             edge_attr = edge_index.storage.value()
#             if edge_attr is not None:
#                 assert x[0].size(-1) == edge_attr.size(-1)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
#         edge_attr, edge_index = data.weight, data.edge_index
#         return self.propagate(edge_index, edge_attr=edge_attr)
    
    def update(self, messages):
        return messages+1000
    
    def message(self, x_i, edge_attr): # x_j,
        #msg = x_j if edge_attr is None else x_j + edge_attr
        msg= x_i + edge_attr
        #print(msg.shape)
        #return self.lin(msg)
        return msg
    
    def __repr__(self):
        return '{}(aggr={})'.format(self.__class__.__name__, self.aggr)
#     def __init__(self, in_channels, out_channels):
#         super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation
#         self.lin = torch.nn.Linear(in_channels, out_channels)

    def __init__(self):
        self.aggr = 'add' #add, mean, max
        super(G_layer_Alex, self).__init__(aggr=self.aggr)  # "Add" aggregation
        #"add", "mean" or "max"
    
    def forward(self, x, edge_attr, edge_index):
#         x, edge_attr, edge_index = data.x, data.weight, data.edge_index
        
        #x = (x, x)
        
#         if isinstance(x, Tensor):
#             #x: OptPairTensor = (x, x)
#             x = (x, x)

        # Node and edge feature dimensionalites need to match.
#         if isinstance(edge_index, Tensor):
#             if edge_attr is not None:
#                 assert x[0].size(-1) == edge_attr.size(-1)
#         elif isinstance(edge_index, SparseTensor):
#             edge_attr = edge_index.storage.value()
#             if edge_attr is not None:
#                 assert x[0].size(-1) == edge_attr.size(-1)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
#         edge_attr, edge_index = data.weight, data.edge_index
#         return self.propagate(edge_index, edge_attr=edge_attr)
    
#     def update(self, messages)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = G_layer_Alex()
        #self.conv2 = GENConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.weight, data.edge_index
        
        x = self.conv1(x, edge_attr, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
        return x
###########################################################################################################    


#     def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
#         print(1)
#         return edge_attr.view(-1, 1) * x_j

#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         print(2)
#         return matmul(adj_t, x, reduce=self.aggr)

#edge_index = to_scipy_sparse_matrix(edge_index)




'Union type; Union[X, Y] means either X or Y.\n\n    To define a union, use e.g. Union[int, str].  
Details:\n    - The arguments must be types and there must be at least one.\n    - None as an argument 
is a special case and is replaced by\n      type(None).\n    - Unions of unions are flattened, 
e.g.::\n\n        Union[Union[int, str], float] == Union[int, str, float]\n\n    - Unions of a 
single argument vanish, e.g.::\n\n        Union[int] == int  # The constructor actually returns 
int\n\n    - Redundant arguments are skipped, e.g.::\n\n        Union[int, str, int] == Union[int, str]\n\n 
   - When comparing unions, the argument order is ignored, e.g.::\n\n        
   Union[int, str] == Union[str, int]\n\n    - You cannot subclass or instantiate a union.\n    
   - You can use Optional[X] as a shorthand for Union[X, None].\n






def message(self, x_i, edge_attr): # x_j,
        #msg = x_j if edge_attr is None else x_j + edge_attr
        msg= x_i + edge_attr
        #print(msg.shape)
        #return self.lin(msg)
        return msg
    
    def __repr__(self):
        return '{}(aggr={})'.format(self.__class__.__name__, self.aggr)
		
		
		
def message(self, x_j: Tensor) -> Tensor:
        r"""Constructs messages from node :math:`j` to node :math:`i`
        in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
        :obj:`edge_index`.
        This function can take any argument as input which was initially
        passed to :meth:`propagate`.
        Furthermore, tensors passed to :meth:`propagate` can be mapped to the
        respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
        :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
        """
        return x_j


[docs]    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)


[docs]    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        r"""Fuses computations of :func:`message` and :func:`aggregate` into a
        single function.
        If applicable, this saves both time and memory since messages do not
        explicitly need to be materialized.
        This function will only gets called in case it is implemented and
        propagation takes place based on a :obj:`torch_sparse.SparseTensor`.
        """
        raise NotImplementedError


[docs]    def update(self, inputs: Tensor) -> Tensor:
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`.
        """
        return inputs
		
		
		
		
		
		



#######################################################################################

class RunManager():
    self.epoch_count = 0
    self.epoch_loss = 0
    self.epoch_num_correct = 0
    self.epoch_start_time = None

    self.run_params = None
    self.run_count = 0
    self.run_data = []
    self.run_start_time = None

    self.network = None
    self.loader = None
    self.tb = None
    
    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
    def end_run(self):
        self.epoch_count = 0
    
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
		
	#time.sleep(1)
	clear_output(wait=True)
	
	## Batches -> To be implemented
	#Inside of run in RunBuilder
	data_loader_train = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=False,
    batch_size = run.batch_size
    )
		# Inside of epoch
		iter_X_train = iter(data_loader_X_train)

200 Epochs -> 270.9028 seconds

50 Epochs -> 63.5913 seconds

50 Epochs with Joblib(-1 processors) -> 206.75 seconds
