
class Node():
    def __init__(self, value, height=None):
        self.value = int(value)
        self.parent = None
        self.left = None
        self.right = None
        self.height = height

class binaryTree():
    def __init__(self, height):
        self.root = Node(2**height - 1, height)
        self.current_node = self.root
        self.nodes = {2**height - 1 : self.root}
        self.left(self.root)
        self.right(self.root)

    def left(self, parent_node):
        if(parent_node.height != 1):
            left_node = Node(parent_node.value - 2 ** (parent_node.height - 1))
            left_node.height = parent_node.height - 1
            left_node.parent = parent_node
            parent_node.left = left_node
            self.nodes[left_node.value] = left_node
            self.left(left_node)
            self.right(left_node)
        else:
            return

    def right(self, parent_node):
        if(parent_node.height != 1):
            right_node = Node(parent_node.value - 1)
            right_node.height = parent_node.height - 1
            right_node.parent = parent_node
            parent_node.right = right_node
            self.nodes[right_node.value] = right_node
            self.left(right_node)
            self.right(right_node)
        else:
            return

    def tree_solution(self, q):
        answer = []
        for node_value in q:
            parent_node = self.nodes[node_value].parent
            if(parent_node):
                answer.append(parent_node.value)
            else:
                answer.append(-1)

        return (str(answer)[1:-1])        

def solution(h, q):
    '''
    Google foobar challenge.

    Problem 02. ion-flux-relabeling Solution
    '''
    tree = binaryTree(h)
    tree.tree_solution(q)

    return tree.tree_solution(q)


solution(3, [7, 3, 5, 1])
solution(5, [19, 14, 28])
