from anytree import Node, RenderTree, find_by_attr


def draw_tree(explored_nodes):
    if len(explored_nodes) > 500:
        print(f'too many nodes: {len(explored_nodes)}')
        return
    
    # find root
    for node_id, node in enumerate(explored_nodes):
        if node.depth == 0:
            root = node
            break
    
    root_node = Node(node_id)
    root_node.h, root_node.V, root_node.cost = root.h, root.V, root.cost
    nodes = {}
    nodes[node_id] = root_node

    for node_id, node in enumerate(explored_nodes):
        if len(node.children):
            for child in node.children:
                child_id = explored_nodes.index(child)
                if child_id not in nodes:
                    nodes[child_id] = child_node = Node(child_id, parent=nodes[node_id])
                    child_node.h, child_node.V, child_node.cost = child.h, child.V, child.cost

    for pre, _, node in RenderTree(root_node):
        print("%s%s, h: %.2f, V: %s, cost: %.2f" % (pre, node.name, node.h, str(node.V), node.cost))