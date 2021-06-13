def bfs_maxdepth(graph, start, maxdepth=10):
    queue = deque([ (start, 0) ])
    depths = {start: 0}
    while queue:
        vertex, depth = queue.popleft()
        if depth == maxdepth:
            break
        for neighbour in graph[vertex]:
            if neighbour in depths:
                continue
            queue.append( (neighbour, depth+1) )
            #depths[neighbour] = depths[vertex] + 1
    return len(depths)
