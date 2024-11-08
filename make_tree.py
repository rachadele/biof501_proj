import yaml
import networkx as nx
projPath= "."
import os

# Load the YAML data
with open(os.path.join(projPath,"meta","master_hierarchy.yaml"))as file:
    tree_data = yaml.safe_load(file)


# Create a networkx graph from the parsed data
tree = nx.DiGraph()

def add_nodes_from_dict(data, parent=None):
    name = data['name']
    colname = data.get('colname')
    tree.add_node(name, colname=colname)
    if parent:
        tree.add_edge(parent, name)
    for child in data.get('children', []):
        add_nodes_from_dict(child, parent=name)

add_nodes_from_dict(tree_data)

# Accessing node attributes
for node in tree.nodes:
    print(f"Node: {node}, colname: {tree.nodes[node].get('colname')}")
