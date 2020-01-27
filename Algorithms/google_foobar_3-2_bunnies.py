
import queue
import copy

class Node():
    def __init__(self, coordinate, is_blocked):
        self.coordinate = coordinate
        self.is_blocked = bool(is_blocked)
        self.is_visited = False
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

def make_map_candidate(maps):
    map_list = [maps]

    for r_idx, row in enumerate(maps):
            for c_idx, col in enumerate(row):   
                if(c_idx < len(maps[0]) - 2):
                    if(maps[r_idx][c_idx + 1] == 1):
                        if(maps[r_idx][c_idx + 2] != 1):
                            maps_copy = copy.deepcopy(maps)
                            maps_copy[r_idx][c_idx + 1] = 0
                            map_list.append(maps_copy)

                if(c_idx > 1):
                    if(maps[r_idx][c_idx - 1] == 1):
                        if(maps[r_idx][c_idx - 2] != 1):
                            maps_copy = copy.deepcopy(maps)
                            maps_copy[r_idx][c_idx - 1] = 0
                            map_list.append(maps_copy)

                if(r_idx < len(maps) - 2):
                    if(maps[r_idx + 1][c_idx] == 1):
                        if(maps[r_idx + 2][c_idx] != 1):
                            maps_copy = copy.deepcopy(maps)
                            maps_copy[r_idx + 1][c_idx] = 0
                            map_list.append(maps_copy)

                if(r_idx > 1):
                    if(maps[r_idx - 1][c_idx] == 1):
                        if(maps[r_idx - 2][c_idx] != 1):
                            maps_copy = copy.deepcopy(maps)
                            maps_copy[r_idx - 1][c_idx] = 0
                            map_list.append(maps_copy)
                
    return map_list

def make_between_candidate(maps, coor):
    x, y = coor
    connect_list = []

    if(y < len(maps[0]) - 2):
        if(maps[x][y + 1] == 1):
            if(maps[x][y + 2] != 1):
                connect_list.append([(x, y), (x, y+2)])

    if(y > 1):
        if(maps[x][y - 1] == 1):
            if(maps[x][y - 2] != 1):
                connect_list.append([(x, y), (x, y-2)])

    if(x < len(maps) - 2):
        if(maps[x + 1][y] == 1):
            if(maps[x + 2][y] != 1):
                connect_list.append([(x, y), (x+2, y)])

    if(x > 1):
        if(maps[x - 1][y] == 1):
            if(maps[x - 2][y] != 1):
                connect_list.append([(x, y), (x-2, y)])
                
    return connect_list

def solve_map_2(maps):    
    '''
    Google foobar challenge.

    Problem 03-2. Prepare the Bunnies' Escape
    '''

    # for row in maps:
    #     print(row)
    node_list = NodeList()
    node_list.add_node(maps)
    node_list.make_relationship(maps)
    # node_list.list_node()
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

    return node_list.get_node((len(maps)-1, len(maps[0])-1)).level

def solve_map(maps):    
    '''
    Google foobar challenge.

    Problem 03-2. Prepare the Bunnies' Escape
    '''

    for row in maps:
        print(row)
    node_list = NodeList()
    node_list.add_node(maps)
    node_list.make_relationship(maps)
    # node_list.list_node()
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

    cur_pos = (len(maps)-1, len(maps[0])-1)

    while(True):
        path_list.append(cur_pos)
        cur_pos = path.get(cur_pos)
        if(cur_pos == (0,0)):
            path_list.append(cur_pos)
            break

    distance_dict = {}
    for path_ in path_list:
        for candidate in make_between_candidate(maps, path_):
            coor_a, coor_b = candidate
            node_a = node_list.get_node(coor_a)
            node_b = node_list.get_node(coor_b)

            if(node_a.is_visited and node_b.is_visited):
                val = abs(node_list.get_node(coor_b).level - node_list.get_node(coor_a).level)
                if(distance_dict.get(path_)):
                    if(val > distance_dict.get(path_)):
                        distance_dict[path_] = val
                else:
                    distance_dict[path_] = val

    print(distance_dict)
    if(distance_dict):
        print(node_list.get_node((len(maps)-1, len(maps[0])-1)).level)

        print(node_list.get_node((len(maps)-1, len(maps[0])-1)).level - max(distance_dict.values()) + 2)
        return node_list.get_node((len(maps)-1, len(maps[0])-1)).level - max(distance_dict.values()) + 2
    else:
        print(node_list.get_node((len(maps)-1, len(maps[0])-1)).level)
        return node_list.get_node((len(maps)-1, len(maps[0])-1)).level


def solution(maps):
    map_list = make_map_candidate(maps)
    answers = []
    for map_ in map_list:
        answers.append(solve_map_2(map_))
    return min(answers)

solve_map([[0, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]])

solve_map([[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
solve_map([[0, 1], [0, 0]])
solve_map([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
solve_map([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]])
solve_map(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0]
    ]
)
solve_map(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0]
    ]
)