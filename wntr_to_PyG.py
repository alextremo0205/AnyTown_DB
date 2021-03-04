import wntr
import networkx as nx
from torch_geometric.utils import convert

def from_wntr_to_PyG(wn):
    wn_links = list(wn.links())
    wn_nodes = list(wn.nodes())
    
    G_WDS = wn.get_graph() # directed multigraph
    uG_WDS = G_WDS.to_undirected() # undirected
    sG_WDS = nx.Graph(uG_WDS) #Simple graph  
    
    i=0
    for (u, v, wt) in sG_WDS.edges.data():
        assert isinstance(wn_links[i][1], wntr.network.elements.Pipe), "The link is not a pipe"
        sG_WDS[u][v]['diameter'] =  wn_links[i][1].diameter
        sG_WDS[u][v]['length'] =    wn_links[i][1].length
        sG_WDS[u][v]['roughness'] = wn_links[i][1].roughness
        i+=1
    
    i=0
    for u in sG_WDS.nodes:
        
        if sG_WDS.nodes[u]['type'] == 'Junction': 
            sG_WDS.nodes[u]['ID'] =          wn_nodes[i][1].name
            sG_WDS.nodes[u]['type_1H'] =     0
            sG_WDS.nodes[u]['base_demand'] = list(wn_nodes[i][1].demand_timeseries_list)[0].base_value
            sG_WDS.nodes[u]['elevation'] =   wn_nodes[i][1].elevation
            sG_WDS.nodes[u]['base_head'] =   0
            
            
        elif sG_WDS.nodes[u]['type'] == 'Reservoir':
            sG_WDS.nodes[u]['ID'] =          wn_nodes[i][1].name
            sG_WDS.nodes[u]['type_1H'] =     1
            sG_WDS.nodes[u]['base_demand'] = 0
            sG_WDS.nodes[u]['elevation'] =   0
            sG_WDS.nodes[u]['base_head'] =   wn_nodes[i][1].base_head
        else:
            print(u)
            raise Exception('Only Junctions and Reservoirs so far')
            break
            
        i+=1

    return convert.from_networkx(sG_WDS) #df_nodes, df_links, sG_WDS


#data = convert.from_networkx(from_wntr_to_nx(wn_WDS))