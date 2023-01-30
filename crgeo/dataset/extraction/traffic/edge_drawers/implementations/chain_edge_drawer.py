import torch

from crgeo.dataset.extraction.traffic.edge_drawers.base_edge_drawer import BaseEdgeDrawer, BaseEdgeDrawingParams


class ChainEdgeDrawer(BaseEdgeDrawer):
    # TODO: Reimplement without 'indices'

    """
    Draws edges between vehicles located on the same lanelet in a chain-like fashion.
    """

    def _draw(self, options: BaseEdgeDrawingParams) -> torch.Tensor:
        raise NotImplementedError()
        """
            Creates edge links between vehicles belonging to the same lanelet
            Args:
                options (BaseEdgeDrawingParams): Containing vehicle, lanelet and vehicle to lanelet data
            Returns:
                np.ndarray: Edge index array.
        """
        data = {i: [] for i in range(len(options.l_data['lanelet_id'].numpy()))}
        edge_indexes = None

        # Segment data into a dictionary with lanelets as keys and vehicle indices as values
        for j in options.v_data['indices']:
            # Here we get the vehicle index from v2l and we get the index of it in options.v_data['indices'] in case the nodes have been 
            # sorted 
            data[int(options.v2l_data['edge_index'][:,j][1])].append(int((options.v_data['indices'] == options.v2l_data['edge_index'][:,j][0]).nonzero(as_tuple=True)[0]))

        for k in data.keys():
            # Ignore for empty tensors
            if len(data[k]) > 0:
                if edge_indexes is None:
                    # We have to connect the preceding vehicle with the succeeding one. Since the list of vehicle indices is flattened, we 
                    # have to vstack the tensors i.e. prior representation -> [0, 1, 2, 3, 4, 5] -> After transformation -> 
                    # tensor([[0, 1, 2, 3, 4],
                    #        [1, 2, 3, 4, 5]]) 
                    edge_indexes = torch.vstack((torch.tensor(data[k][0:(len(data[k]) - 1):], dtype=int), torch.tensor(data[k][1::], dtype=int)))
                else:
                    # Here, we just need to concatenate the stacks obtained for the previous lanelet with edge_indexes computed for this
                    # lanelet
                    edge_index = torch.vstack((torch.tensor(data[k][0:(len(data[k]) - 1):], dtype=int), torch.tensor(data[k][1::], dtype=int)))
                    edge_indexes = torch.cat((edge_indexes, edge_index), 1)

        return edge_indexes
