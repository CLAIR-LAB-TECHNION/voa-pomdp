
def get_tree_max_depth(tree):
    """
    Calculate the maximum depth (number of actions) in a POUCT/POMCP tree.
    """
    if tree is None:
        return 0

    def _get_depth(node, is_vnode=True):
        if node is None or not node.children:
            return 0

        max_child_depth = 0
        for child in node.children.values():
            # For VNodes (action nodes), add 1 to depth
            # For QNodes (observation nodes), don't increment depth
            depth = _get_depth(child, not is_vnode)
            if is_vnode:  # only increment depth when transitioning from VNode->QNode (action)
                depth += 1
            max_child_depth = max(max_child_depth, depth)

        return max_child_depth

    return _get_depth(tree)