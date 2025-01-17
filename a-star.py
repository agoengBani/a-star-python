import pandas as pd
import heapq
import matplotlib.pyplot as plt
from PIL import Image

# Baca dataset dari Excel
graph_data = pd.read_excel('dataset.xlsx', sheet_name='Graph')
heuristic_data = pd.read_excel('dataset.xlsx', sheet_name='Heuristic')
node_data = pd.read_excel('dataset.xlsx', sheet_name='Heuristic')

# Konversi graph menjadi adjacency list
graph = {}
for _, row in graph_data.iterrows():
    node, neighbor, distance = row['Node'], row['Neighbor'], row['Graph (g)']
    if node not in graph:
        graph[node] = []
    graph[node].append((neighbor, distance))

# Konversi heuristic menjadi dictionary
heuristic = {row['Node']: row['Heuristic (h) ke Masjid Dumai Islamic Center'] for _, row in heuristic_data.iterrows()}

# Konversi data koordinat menjadi dictionary
node_positions = {
    row['Node']: tuple(map(float, row['Coordinate'].split(',')))
    for _, row in node_data.iterrows()
}

# Fungsi konversi latitude/longitude ke piksel untuk visualisasi pada peta
def lat_lon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, img_width, img_height):
    x = img_width * (lon - lon_min) / (lon_max - lon_min)
    y = img_height * (lat - lat_min) / (lat_max - lat_min)
    return x, y

# Dimensi gambar peta dan rentang koordinat (ubah sesuai ukuran peta Anda)
img_width, img_height = 1500, 800
lat_min, lat_max = 1.655, 1.685  # Rentang latitude
lon_min, lon_max = 101.428, 101.469  # Rentang longitude

# Konversi latitude/longitude ke piksel
pixel_positions = {
    node: lat_lon_to_pixel(lat, lon, lat_min, lat_max, lon_min, lon_max, img_width, img_height)
    for node, (lat, lon) in node_positions.items()
}

# Validasi apakah semua node dan neighbor dalam graph memiliki koordinat
missing_nodes = []
for node in graph:
    if node not in pixel_positions:
        missing_nodes.append(node)
    for neighbor, _ in graph[node]:
        if neighbor not in pixel_positions:
            missing_nodes.append(neighbor)

if missing_nodes:
    print("Error: Koordinat untuk node berikut tidak ditemukan:")
    for missing_node in set(missing_nodes):
        print(f" - {missing_node}")
    raise ValueError("Harap tambahkan koordinat untuk node yang hilang ke node_positions.")

# Implementasi Algoritma A*
def a_star_algorithm(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_cost = {node: float('inf') for node in graph}
    g_cost[start] = 0

    while open_set:
        _, current_node = heapq.heappop(open_set)
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            return path[::-1], g_cost[goal]

        for neighbor, distance in graph[current_node]:
            tentative_g_cost = g_cost[current_node] + distance
            if tentative_g_cost < g_cost.get(neighbor, float('inf')):
                came_from[neighbor] = current_node
                g_cost[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic.get(neighbor, float('inf'))
                heapq.heappush(open_set, (f_cost, neighbor))

    return None, float('inf')

# Jalankan algoritma
start_node = 'Sonaview Hotel'
goal_node = 'Masjid Habiburrahman Dumai Islamic Center'
path, cost = a_star_algorithm(start_node, goal_node)

# Output hasil
print(f"Jalur tercepat dari {start_node} ke {goal_node}:")
print(f"({' → '.join(path)})")
print(f"Total jarak tempuh: {cost} km")

# Visualisasi jalur pada peta
map_image_path = 'image.png'  # Path ke gambar peta
map_img = Image.open(map_image_path)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(map_img, extent=[0, img_width, 0, img_height])  # Extent sesuai dimensi gambar

# Gambar semua simpul
for node, (x, y) in pixel_positions.items():
    ax.plot(x, y, '.', color='blue', markersize=8)
    ax.text(x, y + 10, node, fontsize=10, ha='center')

# Gambar semua jalur
for node, neighbors in graph.items():
    x1, y1 = pixel_positions[node]
    for neighbor, _ in neighbors:
        x2, y2 = pixel_positions[neighbor]
        ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', alpha=0.7)

# Gambar jalur terpendek
if path:
    for i in range(len(path) - 1):
        x1, y1 = pixel_positions[path[i]]
        x2, y2 = pixel_positions[path[i + 1]]
        ax.plot([x1, x2], [y1, y2], 'red', linewidth=2)

# Judul dan tampilkan
plt.title("Visualisasi Jalur A* di Atas Peta \n Mencari Jalan Tercepat Dari Hotel Sonaview Menuju Masjid Habiburrahman Dumai Islamic Center \n")
plt.axis('off')
plt.show()
