import streamlit as st
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import torch

# اضافه کردن مسیر src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.graph_builder import SolidityGraphBuilder
from src.models.gnn_model import SmartContractGAT

st.set_page_config(page_title="Smart-Contract Auditor", page_icon="🛡️", layout="wide")

st.title("🛡️ Smart-Contract Vulnerability Auditor")
st.markdown("""
**تشخیص آسیب‌پذیری قراردادهای هوشمند با شبکه عصبی گرافی (GNN)**

این ابزار کد سالیدیتی (Solidity) را به یک گراف (AST) تبدیل می‌کند و با استفاده از مدل آموزش‌دیده GraphSAGE، الگوهای خطرناک (مثل Reentrancy) را شناسایی می‌کند.
""")

# ==========================================
# بارگذاری مدل (فقط یک بار)
# ==========================================
@st.cache_resource
def load_model():
    model = SmartContractGAT(num_node_features=4, hidden_channels=16)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'models', 'smart_contract_gnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        return None

model = load_model()

if model is None:
    st.error("⚠️ فایل مدل آموزش‌دیده پیدا نشد. لطفاً ابتدا `python -m src.train` را اجرا کنید.")
    st.stop()

# ==========================================
# ورودی
# ==========================================
st.subheader("۱. کد سالیدیتی خود را وارد کنید")
solidity_code = st.text_area(
    "کد قرارداد (Solidity):",
    height=300,
    placeholder="contract MyContract {\n    // کد شما...\n}"
)

# نمونه‌ها
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    if st.button("بارگذاری نمونه آسیب‌پذیر (Reentrancy)"):
        st.session_state['code'] = """
    contract Vulnerable {
        mapping(address => uint) public balances;
        function withdraw() public {
            uint amount = balances[msg.sender];
            (bool success, ) = msg.sender.call{value: amount}("");
            balances[msg.sender] = 0;
        }
    }
    """
with col_btn2:
    if st.button("بارگذاری نمونه سالم"):
        st.session_state['code'] = """
    contract Safe {
        mapping(address => uint) public balances;
        function withdraw() public {
            uint amount = balances[msg.sender];
            balances[msg.sender] = 0;
            (bool success, ) = msg.sender.transfer(amount);
        }
    }
    """

# استفاده از کد نمونه اگر دکمه زده شده
if 'code' in st.session_state:
    solidity_code = st.session_state['code']

# ==========================================
# تحلیل
# ==========================================
if st.button("تحلیل امنیتی", type="primary"):
    if not solidity_code.strip():
        st.warning("لطفاً کد سالیدیتی را وارد کنید.")
    else:
        with st.spinner("در حال تحلیل گراف و اجرای مدل GNN..."):
            try:
                builder = SolidityGraphBuilder()
                graph = builder.parse_contract(solidity_code)
                data = builder.graph_to_pyg_data(graph)
                
                if data.num_nodes == 0:
                    st.error("هیچ ساختار قابل تشخیصی در کد پیدا نشد.")
                else:
                    batch = torch.zeros(data.num_nodes, dtype=torch.long)
                    with torch.no_grad():
                        out = model(data.x, data.edge_index, batch)
                        prob = torch.exp(out).squeeze().tolist()
                    
                    if isinstance(prob, float):
                        prob = [1-prob, prob]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("نتیجه تحلیل")
                        vuln_prob = prob[1]
                        st.metric("احتمال آسیب‌پذیری", f"{vuln_prob:.1%}")
                        
                        if vuln_prob > 0.5:
                            st.error("🚨 **آسیب‌پذیر**: الگوهای خطرناکی مانند Reentrancy شناسایی شد.")
                        else:
                            st.success("✅ **سالم**: هیچ الگوی خطرناکی شناسایی نشد.")
                            
                    with col2:
                        st.subheader("گراف ساختاری (AST)")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pos = nx.spring_layout(graph, seed=42)
                        
                        node_colors = []
                        for node in graph.nodes(data=True):
                            ntype = node[1].get('type', 'other')
                            if ntype == 'dangerous_call':
                                node_colors.append('red')
                            elif ntype == 'function':
                                node_colors.append('lightblue')
                            elif ntype == 'variable':
                                node_colors.append('lightgreen')
                            else:
                                node_colors.append('gray')
                        
                        nx.draw(graph, pos, ax=ax, with_labels=True, node_color=node_colors, 
                                font_size=8, node_size=800, edge_color='gray')
                        st.pyplot(fig)

                with st.expander("مشاهده جزئیات فنی"):
                    st.write(f"تعداد نودها: {graph.number_of_nodes()}")
                    st.write(f"تعداد یال‌ها: {graph.number_of_edges()}")
                    st.json({"nodes": list(graph.nodes(data=True)), "edges": list(graph.edges())})

            except Exception as e:
                st.error(f"خطا در تحلیل: {e}")

st.divider()
st.caption("ساخته شده با ❤️ با استفاده از PyTorch Geometric و Streamlit")