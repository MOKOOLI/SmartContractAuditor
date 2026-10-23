import re
import networkx as nx
from typing import Dict, List, Tuple

class SolidityGraphBuilder:
    """
    یک کلاس برای تبدیل کد سالیدیتی به گراف (AST-like).
    این نسخه از Regex برای استخراج موجودیت‌ها استفاده می‌کنه.
    در نسخه‌های صنعتی از solc (کامپایلر) استفاده میشه، اما برای دموی سبک این کافیه.
    """
    def __init__(self):
        # الگوهای ساده برای پیدا کردن ساختارها
        self.function_pattern = re.compile(r'function\s+(\w+)\s*\(([^)]*)\)')
        self.call_pattern = re.compile(r'(\w+)\s*\.\s*(call|delegatecall|send|transfer)\s*\(')
        self.variable_pattern = re.compile(r'(uint|int|address|bool|string)\s+(?:public|private|internal)?\s*(\w+)')

    def parse_contract(self, source_code: str) -> nx.DiGraph:
        """کد سالیدیتی رو می‌گیره و یک گراف جهت‌دار برمی‌گردونه."""
        G = nx.DiGraph()
        
        # ۱. پیدا کردن توابع
        functions = self.function_pattern.findall(source_code)
        for i, (func_name, args) in enumerate(functions):
            G.add_node(func_name, type="function", args=args)
            
            # اگر تابع پارامتر داشت، به عنوان نود متغیر اضافه کن
            if args.strip():
                for arg in args.split(','):
                    arg_name = arg.strip().split()[-1] if arg.strip() else "unknown"
                    G.add_node(arg_name, type="variable")
                    G.add_edge(func_name, arg_name, relation="has_param")

        # ۲. پیدا کردن فراخوانی‌های خطرناک (مثل call, delegatecall)
        for match in self.call_pattern.finditer(source_code):
            target, call_type = match.groups()
            # پیدا کردن تابعی که این فراخوانی داخلش هست (ساده‌سازی شده)
            G.add_node(call_type, type="dangerous_call")
            # برای سادگی به یک نود مجازی وصلش می‌کنیم
            G.add_edge("Contract", call_type, relation="contains_call")

        # ۳. پیدا کردن متغیرهای حالت
        variables = self.variable_pattern.findall(source_code)
        for var_type, var_name in variables:
            G.add_node(var_name, type=var_type)
            G.add_edge("Contract", var_name, relation="has_state_var")

        return G

    def graph_to_pyg_data(self, G: nx.DiGraph):
        """
        تبدیل گراف NetworkX به فرمت PyTorch Geometric.
        این تابع برای زمانی است که بخواهیم مدل GNN را آموزش دهیم.
        """
        # در اینجا یک تبدیل ساده انجام می‌دهیم. در پروژه واقعی باید ویژگی‌ها (Features) و لیبل‌ها رو دقیق تعریف کرد.
        import torch
        from torch_geometric.data import Data
        
        # نقشه‌برداری از نام نودها به ایندکس
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # ساخت لیست یال‌ها
        edge_index = []
        for u, v in G.edges():
            edge_index.append([node_mapping[u], node_mapping[v]])
        
        # تبدیل به تنسور
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # ویژگی‌های ساده برای هر نود (مثلاً نوع نود رو به عدد تبدیل می‌کنیم)
        # 0: function, 1: variable, 2: dangerous_call, 3: other
        type_map = {"function": 0, "variable": 1, "dangerous_call": 2}
        x = []
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "other")
            x.append([type_map.get(node_type, 3)])
        
        x = torch.tensor(x, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index)

# --- تست سریع (اختیاری) ---
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
    
    print(f"تعداد نودها: {graph.number_of_nodes()}")
    print(f"تعداد یال‌ها: {graph.number_of_edges()}")
    print("نودها:", list(graph.nodes(data=True)))