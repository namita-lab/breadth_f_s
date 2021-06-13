def bfs_maxdepth(graph, start, goal, maxdepth):
    explored = set()
    queue = deque([(start, 0)])
    depths = {start: 0}

    if start == goal:
        return "That was easy! Start = goal"

    while queue:
        vertex, depths = queue.popleft()
        print(vertex)
        print(depths)
#        depths=str(depths)
        if depths == maxdepth:
            break
        if vertex not in explored:
            neighbours = graph[vertex]

            for neighbour in neighbours:
                if neighbour in str(depths):
                    continue
                queue.append((neighbour, depths+1))

            explored.add(vertex)
                # if neighbour == goal:
                #     return new_path

            #depths[neighbour] = depths[vertex] + 1
    return depths


bfs = bfs_maxdepth(d, g, h, 12)
