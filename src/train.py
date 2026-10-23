import torch
import torch.nn.functional as F
from src.utils.graph_builder import SolidityGraphBuilder
from src.models.gnn_model import SmartContractGAT
import random

def generate_synthetic_dataset(num_samples=200):
    builder = SolidityGraphBuilder()
    dataset = []

    vuln_code = """
    contract Vulnerable {
        mapping(address => uint) public balances;
        function withdraw() public {
            uint amount = balances[msg.sender];
            (bool success, ) = msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
    
    safe_code = """
    contract Safe {
        mapping(address => uint) public balances;
        function withdraw() public {
            uint amount = balances[msg.sender];
            balances[msg.sender] = 0;
            (bool success, ) = msg.sender.transfer(amount);
        }
    }
    """

    for _ in range(num_samples):
        if random.random() > 0.5:
            graph = builder.parse_contract(vuln_code)
            label = torch.tensor([1], dtype=torch.long)
        else:
            graph = builder.parse_contract(safe_code)
            label = torch.tensor([0], dtype=torch.long)
            
        data = builder.graph_to_pyg_data(graph)
        data.y = label
        dataset.append(data)
        
    return dataset

def train_model():
    print("در حال ساخت دیتاست...")
    dataset = generate_synthetic_dataset(200)
    
    # Shuffle برای جلوگیری از بایاس
    random.shuffle(dataset)
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    
    # حالا ویژگی‌ها ۴ بعدی هستند
    model = SmartContractGAT(num_node_features=4, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    print("شروع آموزش مدل...")
    for epoch in range(1, 51):
        total_loss = 0
        for data in train_dataset:
            optimizer.zero_grad()
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            out = model(data.x, data.edge_index, batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {total_loss/len(train_dataset):.4f}")

    model.eval()
    correct = 0
    for data in test_dataset:
        batch = torch.zeros(data.num_nodes, dtype=torch.long)
        with torch.no_grad():
            out = model(data.x, data.edge_index, batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    
    acc = correct / len(test_dataset)
    print(f"دقت مدل روی داده تست: {acc:.2%}")
    
    torch.save(model.state_dict(), "src/models/smart_contract_gnn.pth")
    print("مدل با موفقیت ذخیره شد.")

if __name__ == "__main__":
    train_model()