# Smart-Contract Auditor 🛡️

A Graph Neural Network (GNN) based vulnerability scanner for Solidity smart contracts.

## Features
- Converts Solidity smart contracts into Abstract Syntax Trees (AST) and then into Graph representations.
- Uses Graph Attention Networks (GAT) to detect vulnerabilities like Reentrancy, Integer Overflow, and Unchecked Call Return Values.
- Provides an interactive Streamlit dashboard for real-time analysis.

## Tech Stack
- Python, PyTorch, PyTorch Geometric (PyG)
- Streamlit
- Solc-select (Solidity compiler)

## Architecture
1. **Parsing:** `solc` compiles the smart contract and generates the AST.
2. **Graph Construction:** The AST is converted into a heterogeneous graph (Nodes: functions, variables, calls; Edges: control flow, data flow).
3. **GNN Model:** A custom GAT model is trained on the graph to predict vulnerability types.
4. **UI:** Streamlit app to upload `.sol` files and visualize the graph & results.