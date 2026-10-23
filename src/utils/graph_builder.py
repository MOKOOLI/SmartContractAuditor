import re
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List

class SolidityGraphBuilder:
    """
    یک کلاس برای تبدیل کد سالیدیتی به گراف (AST-like).
    """
    def __init__(self):
        self.function_pattern = re.compile(r'function\s+(\w+)\s*\(([^)]*)\)')
        self.call_pattern = re.compile(r'(\w+)\s*\.\s*(call|delegatecall|send|transfer)\s*\(')
        self.variable_pattern = re.compile(r'(uint|int|address|bool|string)\s+(?:public|private|internal)?\s*(\w+)')

    def parse_contract(self, source_code: str) -> nx.DiGraph:
        """کد سالیدیتی رو می‌گیره و یک گراف جهت‌دار برمی‌گردونه."""
        G = nx.DiGraph()
        
        # ۱. پیدا کردن توابع
        functions = self.function_pattern.findall(source_code)
        for func_name, args in functions:
            G.add_node(func_name, type="function", args=args)
            if args.strip():
                for arg in args.split(','):
                    arg_name = arg.strip().split()[-1] if arg.strip() else "unknown"
                    G.add_node(arg_name, type="variable")
                    G.add_edge(func_name, arg_name, relation="has_param")

        # ۲. پیدا کردن فراخوانی‌های خطرناک
        for match in self.call_pattern.finditer(source_code):
            target, call_type = match.groups()
            G.add_node(call_type, type="dangerous_call")
            G.add_edge("Contract", call_type, relation="contains_call")

        # ۳. پیدا کردن متغیرهای حالت
        variables = self.variable_pattern.findall(source_code)
        for var_type, var_name in variables:
            G.add_node(var_name, type=var_type)
            G.add_edge("Contract", var_name, relation="has_state_var")

        return G

    def graph_to_pyg_data(self, G: nx.DiGraph) -> Data:
        """تبدیل گراف NetworkX به فرمت PyTorch Geometric با ویژگی‌های غنی‌تر."""
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
        
        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # ویژگی‌های ۴ بعدی: [نوع نود, درجه ورودی, درجه خروجی, آیا نود Contract است]
        x = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "other")
            type_val = {"function": 0, "variable": 1, "dangerous_call": 2}.get(node_type, 3)
            
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            is_contract = 1.0 if node == "Contract" else 0.0
            
            x.append([float(type_val), float(in_deg), float(out_deg), is_contract])
        
        x = torch.tensor(x, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)

if __name__ == "__main__":
    sample_code = """
    contract Vulnerable {
        uint public balance;
        function withdraw() public {
            uint amount = balance;
            (bool success, ) = msg.sender.call{value: amount}("");
            balance = 0;
        }
    }
    """
    builder = SolidityGraphBuilder()
    graph = builder.parse_contract(sample_code)
    data = builder.graph_to_pyg_data(graph)
    
    print(f"تعداد نودها: {data.num_nodes}")
    print(f"تعداد یال‌ها: {data.num_edges}")
    print(f"ابعاد ویژگی نودها (x): {data.x.shape}")