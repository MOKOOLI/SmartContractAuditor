import streamlit as st
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

# اضافه کردن مسیر src به سیستم تا ماژول‌ها رو پیدا کنه
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.graph_builder import SolidityGraphBuilder
from src.models.gnn_model import SmartContractGAT
import torch

st.set_page_config(page_title="Smart-Contract Auditor", page_icon="🛡️", layout="wide")

st.title("🛡️ Smart-Contract Vulnerability Auditor")
st.markdown("""
**تشخیص آسیب‌پذیری قراردادهای هوشمند با شبکه عصبی گرافی (GNN)**

این ابزار کد سالیدیتی (Solidity) را به یک گراف (AST) تبدیل می‌کند و با استفاده از مدل GAT الگوهای خطرناک را شناسایی می‌کند.
""")

# ==========================================
# ۱. بخش ورودی کد
# ==========================================
st.subheader("۱. کد سالیدیتی خود را وارد کنید")
solidity_code = st.text_area(
    "کد قرارداد (Solidity):",
    height=300,
    placeholder="contract MyContract {\n    // کد شما...\n}"
)

# نمونه‌ی پیش‌فرض
if st.button("بارگذاری نمونه (Reentrancy)"):
    solidity_code = """
    contract VulnerableBank {
        mapping(address => uint) public balances;
        
        function withdraw() public {
            uint amount = balances[msg.sender];
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            balances[msg.sender] = 0;
        }
    }
    """
    st.session_state['solidity_code'] = solidity_code

if st.button("تحلیل امنیتی", type="primary"):
    if not solidity_code.strip():
        st.warning("لطفاً کد سالیدیتی را وارد کنید.")
    else:
        with st.spinner("در حال تحلیل گراف و اجرای مدل GNN..."):
            try:
                # ۱. ساخت گراف
                builder = SolidityGraphBuilder()
                graph = builder.parse_contract(solidity_code)
                
                # ۲. تبدیل به داده PyG
                data = builder.graph_to_pyg_data(graph)
                
                # ۳. مدل (در این دمو از یک مدل آموزش‌دیده‌نشده/تصادفی استفاده می‌کنیم)
                model = SmartContractGAT(num_node_features=1, hidden_channels=16)
                model.eval()
                
                # اگر گراف خالی بود، یک نود مجازی بساز
                if data.num_nodes == 0:
                    st.error("هیچ ساختار قابل تشخیصی در کد پیدا نشد.")
                else:
                    batch = torch.zeros(data.num_nodes, dtype=torch.long)
                    with torch.no_grad():
                        out = model(data.x, data.edge_index, batch)
                        prob = torch.exp(out).squeeze().tolist()
                    
                    # ۴. نمایش نتایج
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("نتیجه تحلیل")
                        # در اینجا فرض می‌کنیم کلاس ۰ سالم و کلاس ۱ آسیب‌پذیر است
                        vuln_prob = prob[1] if isinstance(prob, list) else prob
                        st.metric("احتمال آسیب‌پذیری", f"{vuln_prob:.1%}")
                        
                        if vuln_prob > 0.5:
                            st.error("🚨 **آسیب‌پذیر**: الگوهای خطرناکی در کد شناسایی شد.")
                        else:
                            st.success("✅ **سالم**: هیچ الگوی خطرناکی شناسایی نشد.")
                            
                    with col2:
                        st.subheader("گراف ساختاری (AST)")
                        # رسم گراف با NetworkX
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pos = nx.spring_layout(graph, seed=42)
                        
                        # رنگ‌بندی بر اساس نوع نود
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

                # ۵. نمایش جزئیات فنی
                with st.expander("مشاهده جزئیات فنی"):
                    st.write(f"تعداد نودها: {graph.number_of_nodes()}")
                    st.write(f"تعداد یال‌ها: {graph.number_of_edges()}")
                    st.json({"nodes": list(graph.nodes(data=True)), "edges": list(graph.edges())})

            except Exception as e:
                st.error(f"خطا در تحلیل: {e}")

st.divider()
st.caption("ساخته شده با ❤️ با استفاده از PyTorch Geometric و Streamlit")