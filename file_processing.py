import os
import pickle
import numpy as np
from collections import defaultdict

from plotter import plot_static_path, plot_dyn_path

# Functions to load files ###########################

def load_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if parts[0] == 'v':
                # Convert strings to floats and append as tuple
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':
                # Convert vertex indices to integers and adjust for 0-indexing
                faces.append([int(index.split('/')[0]) - 1 for index in parts[1:]])

    # Convert list of vertices to a NumPy array all at once for efficiency
    vertices_array = np.array(vertices)

    return vertices_array, faces

####################################################

def normalize_geometry(vertices):
    return vertices / np.linalg.norm(vertices, axis=1).max()

def compute_normal(face_vertices):
    # Assuming face_vertices is an array of vertices [A, B, C, ...]
    # Compute the normal vector using the first three vertices (A, B, C)
    A, B, C = face_vertices[0], face_vertices[1], face_vertices[2]
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    normal = normal / np.linalg.norm(normal)  # Normalize the vector
    return normal

def generate_edges(vertices, faces):
    edge_set = set()
    edge_faces = {}  # Maps edges to a list of face indices that share the edge
    
    # Compute normal vectors for all faces first
    face_normals = [compute_normal(i) for i in vertices[faces]]
    
    for face_idx, face in enumerate(faces):
        num_vertices = len(face)
        for i in range(num_vertices):
            # Normalize edge representation
            normalized_edge = tuple(sorted((face[i], face[(i + 1) % num_vertices])))
            
            if normalized_edge not in edge_faces:
                edge_faces[normalized_edge] = [face_idx]
            else:
                edge_faces[normalized_edge].append(face_idx)
    
    # Filter edges that are shared by faces with the same normal vector
    for edge, face_indices in edge_faces.items():
        if len(face_indices) < 2 or len(np.unique(np.array(face_normals)[face_indices], axis = 0))==2:
            edge_set.add(edge)
    
    return np.array(sorted(list(edge_set)))

def construct_graph(edges):
    """Constructs a graph from edges."""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def find_odd_degree_vertices(graph):
    """Finds vertices of odd degree in the graph."""
    return np.array([v for v, adj in graph.items() if len(adj) % 2 != 0])

def ensure_connectivity(edges):
    """Ensure graph connectivity by identifying and connecting disconnected components."""
    graph = construct_graph(edges)
    visited = set()
    
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
    
    components = []
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
            components.append(vertex)
    
    # Connect disconnected components by adding an edge between them
    new_edges_to_add = np.array([(components[i-1], components[i]) for i in range(1, len(components))], dtype=int)
    if new_edges_to_add.size > 0:
        edges = np.vstack([edges, new_edges_to_add])
    
    return edges

def find_eulerian_path(edges):
    edges = ensure_connectivity(edges)
    graph = construct_graph(edges)
    odd_degree_vertices = find_odd_degree_vertices(graph)

    # You can try to skip this
    if len(odd_degree_vertices) not in [0, 2]:
        print('An Eulerian path does not exist. Solved by duplicating all edges.')
        edges = np.vstack([edges, edges])
        graph = construct_graph(edges)
        # return None

    if len(odd_degree_vertices) == 2:
        start_vertex = odd_degree_vertices[0]
    else:
        start_vertex = next(iter(graph))

    # Iterative DFS to find an Eulerian path or cycle
    def dfs_iterative(start_vertex):
        stack = [start_vertex]
        path = []
        while stack:
            vertex = stack[-1]
            if graph[vertex]:
                next_vertex = graph[vertex].pop()
                graph[next_vertex].remove(vertex)
                stack.append(next_vertex)
            else:
                path.append(stack.pop())
        return path[::-1]  # Reverse the path to get the correct order

    path = np.array(dfs_iterative(start_vertex))
    return path

def path_length(vertices, path):
    vartices = vertices[path]
    # Convert the list of vertices to a NumPy array
    vertices_array = np.array(vertices)
    # Calculate the differences between consecutive vertices
    diffs = np.diff(vertices_array, axis=0)
    # Calculate the Euclidean distance for each segment and sum them up
    total_length = np.sum(np.linalg.norm(diffs, axis=1))
    return total_length

def interpolate_path(vertices, path, t):

    # Calculate total number of segments and scale t to the total number of segments
    total_segments = len(path) - 1
    scaled_t = t * total_segments
    segment_index = np.floor(scaled_t).astype(int)
    
    # Calculate local t within the current segments
    local_t = scaled_t - segment_index
    
    # Clip segment_index to ensure it's within bounds
    segment_index = np.clip(segment_index, 0, total_segments - 1)
    
    # Linearly interpolate between the start and end vertices for each t
    start_vertices = vertices[path[segment_index]]
    end_vertices = vertices[path[segment_index + 1]]
    interpolated_vertices = start_vertices + (end_vertices - start_vertices) * local_t[:, None]
    
    return interpolated_vertices

def save_data(vertices, path, filename):
    # Prepare the data as a dictionary
    data = {
        'vertices': vertices,
        'path': path
    }
    # Replace or append a custom extension for the data file
    data_filename = filename.replace('.obj', '.data')
    # Serialize and save the data to a file
    with open(data_filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(directory_path, filename):
    # Replace or append the custom extension to match the save format
    data_filename = os.path.join(directory_path, filename)
    data_filename = data_filename.replace('.obj', '.data')
    # Deserialize the data from the file
    with open(data_filename, 'rb') as file:
        data = pickle.load(file)
    # Return the loaded vertices and path
    return data['vertices'], data['path']

def process_files(directory_path):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        # Check if it is a file
        if os.path.isfile(file_path):

            if filename.endswith('.obj'):
                print(5*'*'+f'{filename}'+5*'*')
                vertices, faces = load_obj(file_path)
                # vertices = normalize_geometry(vertices)
                print(f'Vertices count: {len(vertices)}')
                print(f'Face types: {set([len(face) for face in faces])}')
                edges = generate_edges(vertices, faces)
                print(f'Unique edges count: {len(edges)}')

                # edges = np.vstack([edges, edges]) # Duplicate edges to ensure eulerian path
                path = find_eulerian_path(edges)
                print(f'Eulerian path vertices: {len(path)}')
                length = path_length(vertices, path)
                print(f'Eulerian path distance: {int(length)}')

                save_data(vertices, path, file_path)

                sampling_rate = 100
                time = np.linspace(0, 1, sampling_rate*int(length), endpoint = False)
                interpath = np.array(interpolate_path(vertices, path, time))

                # plot_dyn_path(interpath, time, 1)








