import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class SmartContractGAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=2):
        super(SmartContractGAT, self).__init__()
        # استفاده از GraphSAGE برای پایداری بیشتر روی دیتاست‌های کوچک
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)