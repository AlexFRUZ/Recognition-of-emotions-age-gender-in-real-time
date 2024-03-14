from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QPushButton, QVBoxLayout, \
    QWidget, QLabel, QTextEdit, QSpinBox, QLineEdit
from PyQt5.QtGui import QPixmap
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy
import math

population_size = 50
generations = 100
mutation_rate = 0.1
elite_size = 5

class GeneticAlgorithmApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Genetic Algorithm App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.layout.addWidget(self.view)

        self.vertex_count_spin_box = QSpinBox(self)
        self.vertex_count_spin_box.setMinimum(2)
        self.vertex_count_spin_box.setValue(12)
        self.layout.addWidget(self.vertex_count_spin_box)

        self.setup_button = QPushButton("Setup Graph", self)
        self.setup_button.clicked.connect(self.setup_graph)
        self.layout.addWidget(self.setup_button)

        self.start_vertex_line_edit = QLineEdit(self)
        self.layout.addWidget(self.start_vertex_line_edit)

        self.end_vertex_line_edit = QLineEdit(self)
        self.layout.addWidget(self.end_vertex_line_edit)

        self.label = QLabel("Select start and end vertices", self)
        self.layout.addWidget(self.label)

        self.run_button = QPushButton("Run Genetic Algorithm", self)
        self.run_button.clicked.connect(self.run_genetic_algorithm)
        self.layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear Selection", self)
        self.clear_button.clicked.connect(self.clear_selection)
        self.layout.addWidget(self.clear_button)

        self.log_text_edit = QTextEdit(self)
        self.layout.addWidget(self.log_text_edit)

        self.best_solution = None
        self.best_fitness_label = QLabel(self)
        self.layout.addWidget(self.best_fitness_label)

        self.golf_course_graph = None

    def setup_graph(self):
        vertex_count = self.vertex_count_spin_box.value()
        self.golf_course_graph = self.create_graph(vertex_count)
        self.display_graph()

    def create_graph(self, vertex_count):
        G = nx.Graph()

        for i in range(vertex_count):
            G.add_node(i)

        for i in range(vertex_count - 1):
            for j in range(i + 1, vertex_count):
                x1, y1 = random.randint(0, 50), random.randint(0, 50)
                x2, y2 = random.randint(0, 50), random.randint(0, 50)
                weight = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 1)
                G.add_edge(i, j, weight=weight)

        return G

    def display_graph(self):
        plt.clf()
        self.graph_pos = nx.spring_layout(self.golf_course_graph)
        nx.draw(self.golf_course_graph, self.graph_pos, with_labels=True, node_size=700, node_color='lightblue',
                font_size=8, font_color='black')

        edge_labels = nx.get_edge_attributes(self.golf_course_graph, 'weight')
        nx.draw_networkx_edge_labels(self.golf_course_graph, self.graph_pos, edge_labels=edge_labels, font_color='red')

        plt.savefig('graph.png')
        pixmap = QPixmap('graph.png')
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def run_genetic_algorithm(self):
        try:
            self.start_vertex = int(self.start_vertex_line_edit.text())
            self.end_vertex = int(self.end_vertex_line_edit.text())

            if self.start_vertex not in self.golf_course_graph.nodes or self.end_vertex not in self.golf_course_graph.nodes:
                raise ValueError("Start or end vertex does not exist in the graph.")

            self.run_genetic_algorithm_internal()
        except ValueError as e:
            self.label.setText(str(e))
        except Exception as ex:
            self.label.setText(f"An unexpected error occurred: {str(ex)}")

    def clear_selection(self):
        self.start_vertex_line_edit.clear()
        self.end_vertex_line_edit.clear()
        self.label.setText("Select start and end vertices")

    def run_genetic_algorithm_internal(self):
        population = generate_population(self.golf_course_graph, self.start_vertex, self.end_vertex)

        log = []
        best_fitness_values = []

        for generation in range(generations):
            fitnesses = [calculate_fitness(individual, self.golf_course_graph) for individual in population]

            log.append(f"Покоління {generation}: Найкраща придатність = {-min(fitnesses)}")

            best_fitness_values.append(-min(fitnesses))

            elite_indices = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
            elite_population = [population[i] for i in elite_indices]

            new_population = copy.deepcopy(elite_population)
            while len(new_population) < population_size:
                parent1 = random.choice(elite_population)
                parent2 = random.choice(elite_population)
                child = crossover(parent1, parent2)
                if random.uniform(0, 1) < mutation_rate:
                    child = mutate(child)
                new_population.append(child)

            population = new_population

        best_solutions = sorted(population, key=lambda x: calculate_fitness(x, self.golf_course_graph), reverse=True)[:4]


        current_best_solution = best_solutions[0]
        current_best_fitness = -min(best_fitness_values)

        if self.best_solution is None or current_best_fitness < -calculate_fitness(self.best_solution,
                                                                                   self.golf_course_graph):
            self.best_solution = current_best_solution


        for i, solution in enumerate(best_solutions):
            path_fitness = calculate_fitness(solution, self.golf_course_graph)
            path_text = f"Шлях {i + 1}: {solution}\nДовжина шляху: {-round(path_fitness, 1)}"
            self.label.setText(path_text)

            self.log_text_edit.append(path_text)

            self.display_path(solution)

    def display_path(self, path):
        plt.clf()
        nx.draw(self.golf_course_graph, self.graph_pos, with_labels=True, node_size=700, node_color='lightblue',
                font_size=8, font_color='black')

        edge_labels = nx.get_edge_attributes(self.golf_course_graph, 'weight')
        nx.draw_networkx_edge_labels(self.golf_course_graph, self.graph_pos, edge_labels=edge_labels, font_color='red')

        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw(self.golf_course_graph, self.graph_pos, edgelist=path_edges, edge_color='green', width=2)

        plt.savefig('graph_with_path.png')
        pixmap = QPixmap('graph_with_path.png')
        self.scene.clear()
        self.scene.addPixmap(pixmap)


def generate_population(graph, start, end):
    return [generate_individual(graph, start, end) for _ in range(population_size)]


def generate_individual(graph, start, end):
    path = [start] + list(graph.nodes - {start, end}) + [end]
    return path


def calculate_fitness(individual, graph):
    path_length = 0
    for i in range(len(individual) - 1):
        edge_data = graph.get_edge_data(individual[i], individual[i + 1])
        if edge_data is not None and 'weight' in edge_data:
            path_length += edge_data['weight']
    return -path_length


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child


def mutate(individual):
    mutation_point1, mutation_point2 = random.sample(range(1, len(individual) - 1), 2)
    individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual


def main():
    app = QApplication(sys.argv)
    window = GeneticAlgorithmApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    import sys
    main()
