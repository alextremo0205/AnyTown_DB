import pandas as pd
import numpy as np
import wntr
def res_index(head, pressure, demand, flowrate, wn, Pmins, indx):
    """
    Compute Todini and Prasad & Park indexes.

    The Prasad & Park index is related to the capability of a system to overcome
    failures while still meeting demands and pressures at the nodes. 
    
    The Todini index defines resilience at a specific time as a measure of surplus
    power at each node and measures relative energy redundancy.
    
    The modification that Prasad & Park include is the uniformity coefficient C. 

    Parameters
    ----------
    head : pandas DataFrame
        A pandas Dataframe containing node head 
        (index = times, columns = node names).
        
    pressure : pandas DataFrame
        A pandas Dataframe containing node pressure 
        (index = times, columns = node names).
        
    demand : pandas DataFrame
        A pandas Dataframe containing node demand 
        (index = times, columns = node names).
        
    flowrate : pandas DataFrame
        A pandas Dataframe containing pump flowrates 
        (index = times, columns = pump names).

    wn : wntr WaterNetworkModel
        Water network model.  The water network model is needed to 
        find the start and end node to each pump.

    Pstar : pandas DataFrame
        Minimum Pressure threshold.
		(index = name of the node, columns= min pressure)

    Returns
    -------
    A pandas Series that contains a time-series of Prasad & Park indexes
    """

    POut = {}
    PExp = {}
    PInRes = {}
    PInPump = {}

    time = head.index
    
    for name in wn.junction_name_list:

        #Begin---------------------- Modification-------------------------
        if indx == 'PP':
            diams = []
            adj_pipes = wn.get_links_for_node(name)
            for i in adj_pipes:
                try:
                    diams.append(wn.get_link(i).diameter)
                except:
                    pass
                    #print('Pump at link: ', i)
            c = sum(diams)/(len(diams)*max(diams))
            #print(c)
            #print(diams, c)
        elif indx == 'Todini':
            c = 1
        
        #End---------------------- Modification-------------------------
        
        h = np.array(head.loc[:,name]) # m
        p = np.array(pressure.loc[:,name])
        e = h - p # m
        q = np.array(demand.loc[:,name]) # m3/s
        #print(q, h, c, q*h*c)
        
        #Begin---------------------- Modification-------------------------
        #print(Pmins)
        Pstar = Pmins.loc[name]
        #print(Pstar)

        POut[name] = q*h*c
        PExp[name] = q*(Pstar+e)*c
        #End---------------------- Modification-------------------------

    for name, node in wn.nodes(wntr.network.Reservoir):
        H = np.array(head.loc[:,name]) # m
        Q = np.array(demand.loc[:,name]) # m3/s
        PInRes[name] = -Q*H # switch sign on Q.

    for name, link in wn.links(wntr.network.Pump):
        start_node = link.start_node_name
        end_node = link.end_node_name
        h_start = np.array(head.loc[:,start_node]) # (m)
        h_end = np.array(head.loc[:,end_node]) # (m)
        h = h_start - h_end # (m)
        q = np.array(flowrate.loc[:,name]) # (m^3/s)
        PInPump[name] = q*(abs(h)) # assumes that pumps always add energy to the system

    PPindx = (sum(POut.values()) - sum(PExp.values()))/  \
        (sum(PInRes.values()) + sum(PInPump.values()) - sum(PExp.values()))

    PPindx = pd.Series(data = PPindx.tolist(), index = time)

    return PPindx