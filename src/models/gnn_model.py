import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GlobalAttention

class SmartContractGAT(torch.nn.Module):
    """
    Graph Attention Network برای تشخیص آسیب‌پذیری قراردادهای هوشمند.
    از چند لایه GAT با هدهای مختلف استفاده می‌کند.
    """
    def __init__(self, num_node_features, hidden_channels, num_classes=2, heads=4):
        super(SmartContractGAT, self).__init__()
        
        # لایه اول GAT
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads, dropout=0.2)
        
        # لایه دوم GAT (ورودی آن hidden_channels * heads است)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.2)
        
        # لایه خطی برای طبقه‌بندی نهایی
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        
        # مکانیزم توجه سراسری برای تجمیع ویژگی‌های گراف
        self.pool = GlobalAttention(gate_nn=torch.nn.Linear(hidden_channels, 1))

    def forward(self, x, edge_index, batch):
        # ۱. اعمال لایه اول و تابع فعال‌ساز
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # تابع فعال‌ساز ELU
        
        # ۲. اعمال لایه دوم
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # ۳. تجمیع نودها به یک بردار برای کل گراف
        x = self.pool(x, batch)
        
        # ۴. خروجی نهایی (احتمال هر کلاس)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

# --- تست سریع مدل (برای اطمینان از درستی ابعاد) ---
if __name__ == "__main__":
    from torch_geometric.data import Data
    
    # یک گراف تستی با ۵ نود و ۱ ویژگی برای هر نود
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn((3, 1))  # ۳ نود، ۱ ویژگی
    batch = torch.tensor([0, 0, 0], dtype=torch.long)  # همه در یک گراف
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    model = SmartContractGAT(num_node_features=1, hidden_channels=16)
    out = model(data.x, data.edge_index, data.batch)
    
    print("ابعاد خروجی مدل:", out.shape)
    print("خروجی (احتمال‌ها):", out)