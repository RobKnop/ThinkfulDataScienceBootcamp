class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

# Establish the initial root node and children
root = Node(22)
root.left = Node(25)
root.right = Node(12)

# Add second layer
root.left.left = Node(234)
root.left.right = Node(56)
root.right.left = Node(6)
root.right.right = Node(4)
# Add third layer
root.left.left.left = Node(9)
root.left.left.right = Node(32)
root.left.right.left = Node(98)
root.left.right.right = Node(235)
root.right.left.left = Node(93)
root.right.left.right = Node(38)
root.right.right.left = Node(76)
root.right.right.right = Node(3)


def breadth_first_search(root):
    arr = []
    to_visit = []
    to_visit.append(root)
    while to_visit:
        current = to_visit.pop(0)
        arr.append(current.val)
        if current.left:
            to_visit.append(current.left)
        if current.right:
            to_visit.append(current.right)
    return arr

print(breadth_first_search(root))