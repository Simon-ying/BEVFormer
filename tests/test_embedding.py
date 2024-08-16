import torch
from torch import nn

# query_embedding = nn.Embedding(2, 4*2)
# query_pos, query = torch.split(query_embedding.weight, 4, dim=1)
# print(f"query_pos.shape={query_pos.shape}, query.shape={query.shape}")
# query = query.unsqueeze(0).expand(2,-1,-1)
# query_pos = query_pos.unsqueeze(0).expand(2,-1,-1)
# print(f"query_pos.shape={query_pos.shape}, query.shape={query.shape}")
# reference_points = nn.Linear(4, 1)
# rp = reference_points(query_pos)
# print(rp.shape)

# print(query_embedding.weight.shape)
# col = torch.arange(50)
# print(f"col.shape: {col.shape}")
# col = query_embedding(col)
# print(col.shape)
h, w = 1024, 1024
spatial_shapes = [(h/8, w/8), (h/16, w/16), (h/32, w/32), (h/64, w/64)]
spatial_shapes = spatial_shapes[::-1]
spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
import pdb; pdb.set_trace()