import tkinter as tk
import random
import time
from collections import deque
import heapq

class Maze:
    def __init__(self, cols, rows, cell_size):
        self.cols = cols
        self.rows = rows
        self.cell_size = cell_size
        self.grid = [[{'top': True, 'right': True, 'bottom': True, 'left': True, 'visited': False}
                      for _ in range(rows)] for _ in range(cols)]
        self.stack = []
        self.generate_maze()

    def generate_maze(self):
        current = (0, 0)
        self.grid[current[0]][current[1]]['visited'] = True
        self.stack.append(current)
        while self.stack:
            x, y = current
            neighbors = []
            directions = [('top', (x, y-1)),
                          ('right', (x+1, y)),
                          ('bottom', (x, y+1)),
                          ('left', (x-1, y))]
            for direction, (nx, ny) in directions:
                if 0 <= nx < self.cols and 0 <= ny < self.rows and not self.grid[nx][ny]['visited']:
                    neighbors.append((direction, (nx, ny)))
            if neighbors:
                direction, next_cell = random.choice(neighbors)
                nx, ny = next_cell
                if direction == 'top':
                    self.grid[x][y]['top'] = False
                    self.grid[nx][ny]['bottom'] = False
                elif direction == 'right':
                    self.grid[x][y]['right'] = False
                    self.grid[nx][ny]['left'] = False
                elif direction == 'bottom':
                    self.grid[x][y]['bottom'] = False
                    self.grid[nx][ny]['top'] = False
                elif direction == 'left':
                    self.grid[x][y]['left'] = False
                    self.grid[nx][ny]['right'] = False
                self.grid[nx][ny]['visited'] = True
                self.stack.append(current)
                current = next_cell
            else:
                current = self.stack.pop()
        # Reset visited flags for search algorithms.
        for x in range(self.cols):
            for y in range(self.rows):
                self.grid[x][y]['visited'] = False

    def get_cell_walls(self, x, y):
        return self.grid[x][y]

class MazeGUI:
    def __init__(self, master, default_cols=15, default_rows=15):
        self.master = master
        self.max_canvas_width = 800
        self.max_canvas_height = 600
        self.cols = default_cols
        self.rows = default_rows
        self.cell_size = int(min(self.max_canvas_width / self.cols, self.max_canvas_height / self.rows))
        self.maze = None

        # Dimension inputs and Generate Maze button.
        self.dimension_frame = tk.Frame(master)
        self.dimension_frame.pack(pady=5)
        tk.Label(self.dimension_frame, text="Columns:").grid(row=0, column=0)
        self.cols_entry = tk.Entry(self.dimension_frame, width=4)
        self.cols_entry.insert(0, str(default_cols))
        self.cols_entry.grid(row=0, column=1, padx=5)
        self.cols_entry.bind("<Return>", lambda event: self.generate_new_maze())
        tk.Label(self.dimension_frame, text="Rows:").grid(row=0, column=2)
        self.rows_entry = tk.Entry(self.dimension_frame, width=4)
        self.rows_entry.insert(0, str(default_rows))
        self.rows_entry.grid(row=0, column=3, padx=5)
        self.rows_entry.bind("<Return>", lambda event: self.generate_new_maze())
        self.generate_button = tk.Button(self.dimension_frame, text="Generate Maze", command=self.generate_new_maze)
        self.generate_button.grid(row=0, column=4, padx=10)

        # Maze canvas.
        self.canvas = tk.Canvas(master, width=self.cols * self.cell_size + 2,
                                height=self.rows * self.cell_size + 2, bg="white")
        self.canvas.pack(pady=5)

        # Message label.
        self.message_label = tk.Label(master, text="", font=("Helvetica", 14))
        self.message_label.pack(pady=5)

        # AI Controls.
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=5)
        self.algo_choice = tk.StringVar()
        algo_options = ["DFS", "BFS", "A*", "Greedy Best-First", "Dijkstra", "Bidirectional", "JPS", "Fringe Search"]
        self.algo_choice.set("DFS")
        self.algo_menu = tk.OptionMenu(self.control_frame, self.algo_choice, *algo_options)
        self.algo_menu.grid(row=0, column=0, padx=10)
        self.start_button = tk.Button(self.control_frame, text="Start AI", command=self.start_ai)
        self.start_button.grid(row=0, column=1, padx=10)
        self.reset_button = tk.Button(self.control_frame, text="Reset AI", command=self.reset_ai)
        self.reset_button.grid(row=0, column=2, padx=10)

        # Visualization.
        self.ai_line_id = None

        # Search state variables.
        self.dfs_stack = []
        self.bfs_queue = None
        self.bfs_parent = {}
        self.bfs_visited = set()   # New: BFS uses its own visited set.
        self.astar_open = []
        self.astar_parent = {}
        self.astar_g = {}
        self.greedy_open = []
        self.greedy_parent = {}
        self.dijkstra_open = []
        self.dijkstra_parent = {}
        self.dijkstra_dist = {}
        self.bi_forward_queue = deque()
        self.bi_backward_queue = deque()
        self.bi_forward_parent = {}
        self.bi_backward_parent = {}
        self.bi_forward_visited = set()
        self.bi_backward_visited = set()
        # For JPS:
        self.jps_open = []
        self.jps_parent = {}
        self.jps_g = {}
        # For Fringe Search:
        self.fringe_open = []
        self.fringe_next = []
        self.fringe_threshold = None
        self.fringe_min = float('inf')
        self.fringe_parent = {}
        self.fringe_g = {}

        self.start_cell = None
        self.goal_cell = None

        self.generate_new_maze()

    def generate_new_maze(self):
        self.message_label.config(text="")
        try:
            self.cols = int(self.cols_entry.get())
            self.rows = int(self.rows_entry.get())
        except ValueError:
            pass
        self.cell_size = int(min(self.max_canvas_width / self.cols, self.max_canvas_height / self.rows))
        self.canvas.config(width=self.cols * self.cell_size + 2,
                           height=self.rows * self.cell_size + 2)
        self.maze = Maze(self.cols, self.rows, self.cell_size)
        self.draw_maze()
        self.start_cell = (0, 0)
        self.goal_cell = (self.cols - 1, self.rows - 1)
        self.draw_cell(self.start_cell[0], self.start_cell[1], "green")
        self.draw_cell(self.goal_cell[0], self.goal_cell[1], "red")
        self.reset_search_state()

    def reset_search_state(self):
        # Remove bidirectional markers.
        self.canvas.delete("bi")
        for x in range(self.cols):
            for y in range(self.rows):
                self.maze.grid[x][y]['visited'] = False
        self.dfs_stack = [self.start_cell]
        self.bfs_queue = deque([self.start_cell])
        self.bfs_parent = {self.start_cell: None}
        self.bfs_visited = {self.start_cell}  # Initialize BFS visited set.
        self.astar_open = []
        self.astar_parent = {self.start_cell: None}
        self.astar_g = {self.start_cell: 0}
        heapq.heappush(self.astar_open, (self.manhattan(self.start_cell, self.goal_cell), self.start_cell))
        self.greedy_open = []
        self.greedy_parent = {self.start_cell: None}
        heapq.heappush(self.greedy_open, (self.manhattan(self.start_cell, self.goal_cell), self.start_cell))
        self.dijkstra_open = []
        self.dijkstra_parent = {self.start_cell: None}
        self.dijkstra_dist = {self.start_cell: 0}
        heapq.heappush(self.dijkstra_open, (0, self.start_cell))
        self.bi_forward_queue = deque([self.start_cell])
        self.bi_backward_queue = deque([self.goal_cell])
        self.bi_forward_parent = {self.start_cell: None}
        self.bi_backward_parent = {self.goal_cell: None}
        self.bi_forward_visited = {self.start_cell}
        self.bi_backward_visited = {self.goal_cell}
        # For JPS:
        self.jps_open = []
        self.jps_parent = {self.start_cell: None}
        self.jps_g = {self.start_cell: 0}
        heapq.heappush(self.jps_open, (self.manhattan(self.start_cell, self.goal_cell), self.start_cell))
        # For Fringe Search:
        self.fringe_open = [self.start_cell]
        self.fringe_next = []
        self.fringe_threshold = self.manhattan(self.start_cell, self.goal_cell)
        self.fringe_min = float('inf')
        self.fringe_parent = {self.start_cell: None}
        self.fringe_g = {self.start_cell: 0}
        if self.ai_line_id:
            self.canvas.delete(self.ai_line_id)
            self.ai_line_id = None

    def draw_maze(self):
        self.canvas.delete("all")
        for x in range(self.cols):
            for y in range(self.rows):
                cell = self.maze.get_cell_walls(x, y)
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if cell['top']:
                    self.canvas.create_line(x1, y1, x2, y1)
                if cell['right']:
                    self.canvas.create_line(x2, y1, x2, y2)
                if cell['bottom']:
                    self.canvas.create_line(x1, y2, x2, y2)
                if cell['left']:
                    self.canvas.create_line(x1, y1, x1, y2)

    def draw_cell(self, col, row, color):
        x1 = col * self.cell_size + 2
        y1 = row * self.cell_size + 2
        x2 = x1 + self.cell_size - 4
        y2 = y1 + self.cell_size - 4
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def update_path_line(self, path):
        if self.ai_line_id:
            self.canvas.delete(self.ai_line_id)
        points = []
        for cell in path:
            x, y = cell
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            points.extend([cx, cy])
        if len(points) >= 4:
            self.ai_line_id = self.canvas.create_line(points, fill="blue", width=4)

    def manhattan(self, cell, goal):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    # --- Bidirectional Visualization ---
    def update_bidirectional_visualization(self):
        self.canvas.delete("bi")
        r = self.cell_size // 4
        for cell in self.bi_forward_visited:
            x, y = cell
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="blue", tags="bi")
        for cell in self.bi_backward_visited:
            x, y = cell
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline="red", tags="bi")

    # --- AI Dispatch ---
    def start_ai(self):
        self.message_label.config(text="")
        self.reset_search_state()
        algo = self.algo_choice.get()
        if algo == "DFS":
            self.start_dfs()
        elif algo == "BFS":
            self.start_bfs()
        elif algo == "A*":
            self.start_astar()
        elif algo == "Greedy Best-First":
            self.start_greedy()
        elif algo == "Dijkstra":
            self.start_dijkstra()
        elif algo == "Bidirectional":
            self.start_bidirectional()
        elif algo == "JPS":
            self.start_jps()
        elif algo == "Fringe Search":
            self.start_fringe()

    # ------------------- DFS -------------------
    def start_dfs(self):
        self.start_time = time.time()
        self.dfs_step()

    def dfs_step(self):
        if not self.dfs_stack:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        current = self.dfs_stack[-1]
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (DFS) in {elapsed:.2f}s", fg="green")
            self.update_path_line(self.dfs_stack)
            return
        x, y = current
        cell = self.maze.get_cell_walls(x, y)
        neighbors = []
        if not cell['top'] and y > 0 and not self.maze.grid[x][y-1]['visited']:
            neighbors.append((x, y-1))
        if not cell['right'] and x < self.cols - 1 and not self.maze.grid[x+1][y]['visited']:
            neighbors.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1 and not self.maze.grid[x][y+1]['visited']:
            neighbors.append((x, y+1))
        if not cell['left'] and x > 0 and not self.maze.grid[x-1][y]['visited']:
            neighbors.append((x-1, y))
        if neighbors:
            next_cell = random.choice(neighbors)
            self.maze.grid[next_cell[0]][next_cell[1]]['visited'] = True
            self.dfs_stack.append(next_cell)
            self.update_path_line(self.dfs_stack)
        else:
            self.dfs_stack.pop()
            self.update_path_line(self.dfs_stack)
        self.master.after(50, self.dfs_step)

    # ------------------- BFS -------------------
    def start_bfs(self):
        self.start_time = time.time()
        self.bfs_step()

    def bfs_step(self):
        if not self.bfs_queue:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        current = self.bfs_queue.popleft()
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.bfs_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (BFS) in {elapsed:.2f}s", fg="green")
            return
        x, y = current
        cell = self.maze.get_cell_walls(x, y)
        neighbors = []
        if not cell['top'] and y > 0:
            neighbors.append((x, y-1))
        if not cell['right'] and x < self.cols - 1:
            neighbors.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1:
            neighbors.append((x, y+1))
        if not cell['left'] and x > 0:
            neighbors.append((x-1, y))
        for n in neighbors:
            if n not in self.bfs_visited:
                self.bfs_visited.add(n)
                self.bfs_parent[n] = current
                self.bfs_queue.append(n)
        self.master.after(50, self.bfs_step)

    # ------------------- A* -------------------
    def start_astar(self):
        self.start_time = time.time()
        self.astar_step()

    def astar_step(self):
        if not self.astar_open:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        f, current = heapq.heappop(self.astar_open)
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.astar_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (A*) in {elapsed:.2f}s", fg="green")
            return
        x, y = current
        cell = self.maze.get_cell_walls(x, y)
        neighbors = []
        if not cell['top'] and y > 0:
            neighbors.append((x, y-1))
        if not cell['right'] and x < self.cols - 1:
            neighbors.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1:
            neighbors.append((x, y+1))
        if not cell['left'] and x > 0:
            neighbors.append((x-1, y))
        for n in neighbors:
            tentative_g = self.astar_g[current] + 1
            if n not in self.astar_g or tentative_g < self.astar_g[n]:
                self.astar_g[n] = tentative_g
                self.astar_parent[n] = current
                f_score = tentative_g + self.manhattan(n, self.goal_cell)
                heapq.heappush(self.astar_open, (f_score, n))
        self.master.after(50, self.astar_step)

    # ------------------- Greedy Best-First -------------------
    def start_greedy(self):
        self.start_time = time.time()
        self.greedy_step()

    def greedy_step(self):
        if not self.greedy_open:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        f, current = heapq.heappop(self.greedy_open)
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.greedy_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (Greedy) in {elapsed:.2f}s", fg="green")
            return
        x, y = current
        cell = self.maze.get_cell_walls(x, y)
        neighbors = []
        if not cell['top'] and y > 0:
            neighbors.append((x, y-1))
        if not cell['right'] and x < self.cols - 1:
            neighbors.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1:
            neighbors.append((x, y+1))
        if not cell['left'] and x > 0:
            neighbors.append((x-1, y))
        for n in neighbors:
            h = self.manhattan(n, self.goal_cell)
            if n not in self.greedy_parent:
                self.greedy_parent[n] = current
                heapq.heappush(self.greedy_open, (h, n))
        self.master.after(50, self.greedy_step)

    # ------------------- Dijkstra -------------------
    def start_dijkstra(self):
        self.start_time = time.time()
        self.dijkstra_step()

    def dijkstra_step(self):
        if not self.dijkstra_open:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        dist, current = heapq.heappop(self.dijkstra_open)
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.dijkstra_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (Dijkstra) in {elapsed:.2f}s", fg="green")
            return
        x, y = current
        cell = self.maze.get_cell_walls(x, y)
        neighbors = []
        if not cell['top'] and y > 0:
            neighbors.append((x, y-1))
        if not cell['right'] and x < self.cols - 1:
            neighbors.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1:
            neighbors.append((x, y+1))
        if not cell['left'] and x > 0:
            neighbors.append((x-1, y))
        for n in neighbors:
            tentative_dist = self.dijkstra_dist[current] + 1
            if n not in self.dijkstra_dist or tentative_dist < self.dijkstra_dist[n]:
                self.dijkstra_dist[n] = tentative_dist
                self.dijkstra_parent[n] = current
                heapq.heappush(self.dijkstra_open, (tentative_dist, n))
        self.master.after(50, self.dijkstra_step)

    # ------------------- Bidirectional -------------------
    def start_bidirectional(self):
        self.start_time = time.time()
        self.bidirectional_step()

    def bidirectional_step(self):
        meeting_point = None
        for cell in self.bi_forward_visited:
            if cell in self.bi_backward_visited:
                meeting_point = cell
                break
        if meeting_point is not None:
            path_forward = []
            node = meeting_point
            while node is not None:
                path_forward.append(node)
                node = self.bi_forward_parent.get(node)
            path_forward.reverse()
            path_backward = []
            node = self.bi_backward_parent.get(meeting_point)
            while node is not None:
                path_backward.append(node)
                node = self.bi_backward_parent.get(node)
            full_path = path_forward + path_backward
            self.update_path_line(full_path)
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (Bidirectional) in {elapsed:.2f}s", fg="green")
            return

        if self.bi_forward_queue:
            current_f = self.bi_forward_queue.popleft()
            x, y = current_f
            cell = self.maze.get_cell_walls(x, y)
            neighbors = []
            if not cell['top'] and y > 0:
                neighbors.append((x, y-1))
            if not cell['right'] and x < self.cols - 1:
                neighbors.append((x+1, y))
            if not cell['bottom'] and y < self.rows - 1:
                neighbors.append((x, y+1))
            if not cell['left'] and x > 0:
                neighbors.append((x-1, y))
            for n in neighbors:
                if n not in self.bi_forward_visited:
                    self.bi_forward_visited.add(n)
                    self.bi_forward_parent[n] = current_f
                    self.bi_forward_queue.append(n)
        if self.bi_backward_queue:
            current_b = self.bi_backward_queue.popleft()
            x, y = current_b
            cell = self.maze.get_cell_walls(x, y)
            neighbors = []
            if not cell['top'] and y > 0:
                neighbors.append((x, y-1))
            if not cell['right'] and x < self.cols - 1:
                neighbors.append((x+1, y))
            if not cell['bottom'] and y < self.rows - 1:
                neighbors.append((x, y+1))
            if not cell['left'] and x > 0:
                neighbors.append((x-1, y))
            for n in neighbors:
                if n not in self.bi_backward_visited:
                    self.bi_backward_visited.add(n)
                    self.bi_backward_parent[n] = current_b
                    self.bi_backward_queue.append(n)
        self.update_bidirectional_visualization()
        self.master.after(50, self.bidirectional_step)

    # ------------------- JPS (Jump Point Search) -------------------
    def jps_forced(self, x, y, dx, dy, prev):
        neighbors = self.get_open_neighbors(x, y)
        if prev in neighbors:
            neighbors.remove(prev)
        return len(neighbors) >= 1

    def get_open_neighbors(self, x, y):
        res = []
        cell = self.maze.get_cell_walls(x, y)
        if not cell['top'] and y > 0:
            res.append((x, y-1))
        if not cell['right'] and x < self.cols - 1:
            res.append((x+1, y))
        if not cell['bottom'] and y < self.rows - 1:
            res.append((x, y+1))
        if not cell['left'] and x > 0:
            res.append((x-1, y))
        return res

    def jps_jump(self, x, y, dx, dy, prev):
        nx = x + dx
        ny = y + dy
        if nx < 0 or ny < 0 or nx >= self.cols or ny >= self.rows:
            return None
        cell = self.maze.get_cell_walls(x, y)
        if dx == 1 and cell['right']:
            return None
        if dx == -1 and cell['left']:
            return None
        if dy == 1 and cell['bottom']:
            return None
        if dy == -1 and cell['top']:
            return None
        new_cell = (nx, ny)
        if new_cell == self.goal_cell:
            return new_cell
        if prev is None:
            prev = (x, y)
        if self.jps_forced(nx, ny, dx, dy, prev):
            return new_cell
        return self.jps_jump(nx, ny, dx, dy, new_cell)

    def start_jps(self):
        self.start_time = time.time()
        self.jps_step()

    def jps_step(self):
        if not self.jps_open:
            self.message_label.config(text="Invalid Maze", fg="red")
            return
        f, current = heapq.heappop(self.jps_open)
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.jps_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (JPS) in {elapsed:.2f}s", fg="green")
            return
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        for dx, dy in directions:
            jp = self.jps_jump(current[0], current[1], dx, dy, current)
            if jp is not None:
                cost = abs(jp[0]-current[0]) + abs(jp[1]-current[1])
                tentative_g = self.jps_g[current] + cost
                if jp not in self.jps_g or tentative_g < self.jps_g[jp]:
                    self.jps_g[jp] = tentative_g
                    self.jps_parent[jp] = current
                    f_score = tentative_g + self.manhattan(jp, self.goal_cell)
                    heapq.heappush(self.jps_open, (f_score, jp))
        self.master.after(50, self.jps_step)

    # ------------------- Fringe Search -------------------
    def start_fringe(self):
        self.start_time = time.time()
        self.fringe_step()

    def fringe_step(self):
        if not self.fringe_open:
            if not self.fringe_next:
                self.message_label.config(text="Invalid Maze", fg="red")
                return
            self.fringe_open = self.fringe_next
            self.fringe_next = []
            self.fringe_threshold = self.fringe_min
            self.fringe_min = float('inf')
        current = self.fringe_open.pop(0)
        f = self.fringe_g[current] + self.manhattan(current, self.goal_cell)
        path = []
        node = current
        while node is not None:
            path.append(node)
            node = self.fringe_parent.get(node)
        path.reverse()
        self.update_path_line(path)
        if current == self.goal_cell:
            elapsed = time.time() - self.start_time
            self.message_label.config(text=f"Goal Found! (Fringe) in {elapsed:.2f}s", fg="green")
            return
        if f > self.fringe_threshold:
            if f < self.fringe_min:
                self.fringe_min = f
        else:
            x, y = current
            cell = self.maze.get_cell_walls(x, y)
            neighbors = []
            if not cell['top'] and y > 0:
                neighbors.append((x, y-1))
            if not cell['right'] and x < self.cols - 1:
                neighbors.append((x+1, y))
            if not cell['bottom'] and y < self.rows - 1:
                neighbors.append((x, y+1))
            if not cell['left'] and x > 0:
                neighbors.append((x-1, y))
            for n in neighbors:
                if n not in self.fringe_g or self.fringe_g[current] + 1 < self.fringe_g[n]:
                    self.fringe_g[n] = self.fringe_g[current] + 1
                    self.fringe_parent[n] = current
                    fn = self.fringe_g[n] + self.manhattan(n, self.goal_cell)
                    if fn > self.fringe_threshold:
                        if fn < self.fringe_min:
                            self.fringe_min = fn
                        self.fringe_next.append(n)
                    else:
                        self.fringe_open.append(n)
        self.master.after(50, self.fringe_step)

    def reset_ai(self):
        self.message_label.config(text="")
        self.reset_search_state()
        self.draw_maze()
        self.draw_cell(self.start_cell[0], self.start_cell[1], "green")
        self.draw_cell(self.goal_cell[0], self.goal_cell[1], "red")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Maze Generator and Multi-AI Solver")
    gui = MazeGUI(root, default_cols=15, default_rows=15)
    root.mainloop()
