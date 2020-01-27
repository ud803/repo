
import queue
import copy

class Node():
    def __init__(self, coordinate, is_blocked):
        self.coordinate = coordinate
        self.is_blocked = bool(is_blocked)
        self.is_visited = False
        self.is_visited_2 = False
        self.is_reversed = False
        self.adjacents = {}
        self.level = 1

class NodeList():
    def __init__(self):
        self.node_list = {}

    def add_node(self, maps):
        for r_idx, row in enumerate(maps):
            for c_idx, col in enumerate(row):   
                self.node_list[(r_idx, c_idx)] = Node((r_idx, c_idx), col)

    def get_node(self, coor):
        return self.node_list[coor]

    def connect_nodes(self, coor_a, coor_b):
        self.node_list[coor_a].adjacents[coor_b] = self.node_list[coor_b]
        self.node_list[coor_b].adjacents[coor_a] = self.node_list[coor_a]

    def make_relationship(self, maps):
        for r_idx, row in enumerate(maps):
            for c_idx, col in enumerate(row):
                coor_now = (r_idx, c_idx)
                if(self.node_list[coor_now].is_blocked):
                    continue
                coors = [
                    (r_idx, c_idx - 1),
                    (r_idx, c_idx + 1),
                    (r_idx - 1, c_idx),
                    (r_idx + 1, c_idx)
                    ]
                
                for coor in coors:
                    try:
                        if(self.node_list[coor].is_blocked == False):
                            self.connect_nodes(coor_now, coor)
                    except:
                        pass

    def list_node(self):
        for node in self.node_list:
            print(self.node_list[node].coordinate, self.node_list[node].is_reversed)

def find_ones(maps):
    one_list = []
    for r_idx, row in enumerate(maps):
            for c_idx, col in enumerate(row): 
                if(col == 1):
                    one_list.append((r_idx, c_idx))
    return one_list

def solution(maps):    
    '''
    Google foobar challenge.

    Problem 03-2. Prepare the Bunnies' Escape
    '''

    for row in maps:
        print(row)
    print('\n')
    new_map = [[0 for _ in range(len(maps[0]))] for _ in range(len(maps))]

    node_list = NodeList()
    node_list.add_node(new_map)
    node_list.make_relationship(new_map)
    node_queue = queue.Queue()

    node_queue.put(node_list.get_node((0, 0)))
    path = {}
    path_list = []
    count = 0
    level = 0
    while(not node_queue.empty()):
        popped_element = node_queue.get()
        popped_element.is_visited = True
        level = popped_element.level
        count += 1

        for adj_node in popped_element.adjacents:
            if popped_element.adjacents[adj_node].is_visited: continue
            popped_element.adjacents[adj_node].level = level + 1
            node_queue.put(popped_element.adjacents[adj_node])
            path[adj_node] = popped_element.coordinate

    cur_pos = (len(new_map)-1, len(new_map[0])-1)

    while(True):
        path_list.append(cur_pos)
        cur_pos = path.get(cur_pos)
        if(cur_pos == (0,0)):
            path_list.append(cur_pos)
            break

    contact_list = []
    for coor_now in find_ones(maps):
        print(coor_now)
        r_idx, c_idx = coor_now
        coors = [
                    (r_idx, c_idx - 1),
                    (r_idx, c_idx + 1),
                    (r_idx - 1, c_idx),
                    (r_idx + 1, c_idx)
                ]
        for coor in coors:
            try:
                if(not node_list.get_node(coor).is_blocked):
                    node_list.connect_nodes(coor_now, coor)
            except:
                pass

        for adj in node_list.get_node(coor_now).adjacents:
            print(node_list.get_node(coor_now).adjacents[adj].coordinate)
        break
        node_queue = queue.Queue()
        node_list.get_node(coor)
        node_queue.put(node_list.get_node(coor))

        while(not node_queue.empty()):
            popped_element = node_queue.get()
            popped_element.is_visited_2 = True

            for adj_node in popped_element.adjacents:
                if popped_element.adjacents[adj_node].is_visited_2: continue
                if popped_element.adjacents[adj_node].coordinate in path_list:
                    contact_list.append(popped_element.adjacents[adj_node].coordinate)
                    continue
                node_queue.put(popped_element.adjacents[adj_node])
        print(contact_list)
        break
# solution([[0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])

# solution([[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
# solution([[0, 1], [0, 0]])
# solution([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
# solution([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
solution(
    [
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
)
# solution(
#     [
#         [0, 1, 1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 1, 1, 1, 1, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 1, 1, 0, 1, 0, 1, 1, 1],
#         [0, 1, 1, 0, 1, 0, 1, 1, 1],
#         [0, 1, 1, 0, 1, 0, 1, 1, 1],
#         [0, 1, 0, 0, 1, 0, 1, 1, 1],
#         [0, 1, 0, 0, 1, 0, 0, 1, 1],
#         [0, 0, 0, 0, 1, 1, 0, 0, 0]
#     ]
# )