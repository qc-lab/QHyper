# VIEW - things from PyQt
# MODEL - the DAG
# CONTROLLER - things that take the input and provide it to the Model


from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from custom_receipe import CustomRecipe
from wfcommons.common import Workflow


class CustomDialog(qtw.QDialog):
    def __init__(self, parent=None, window_title="Hello", line_edit_text="Line edit text"):

        super().__init__(parent)

        self.setWindowTitle(window_title)

        buttons_ok_cancel = qtw.QDialogButtonBox.Ok | qtw.QDialogButtonBox.Cancel
        self.buttonBox = qtw.QDialogButtonBox(buttons_ok_cancel)
        self.buttonBox.accepted.connect(self.read_input)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = qtw.QFormLayout()
        self.line_edit = qtw.QLineEdit()
        self.layout.addRow(line_edit_text, self.line_edit)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.input = None

    def read_input(self):
        self.input = self.line_edit.text()
        self.close()


class MplWidget(qtw.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = qtw.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.show()


class MyQtApp(qtw.QWidget):
    def __init__(self):
        super(MyQtApp, self).__init__()

        self.matplolib_canvas = MplWidget(self)
        self.configure_buttons()
        self.configure_layout()
        self.G = Workflow("default_name")

    def configure_buttons(self):
        self.button_add_node = qtw.QPushButton('Add node', self)
        self.button_add_edge = qtw.QPushButton('Add edge', self)
        self.button_remove_node = qtw.QPushButton('Remove node', self)
        self.button_remove_edge = qtw.QPushButton('Remove edge', self)
        self.button_generate_wffile = qtw.QPushButton('Generate WfFile', self)
        self.button_add_node.clicked.connect(self.add_node)
        self.button_add_edge.clicked.connect(self.add_edge)
        self.button_remove_node.clicked.connect(self.remove_node)
        self.button_remove_edge.clicked.connect(self.remove_edge)
        self.button_generate_wffile.clicked.connect(self.generate_wffile)

    def configure_layout(self):
        self.layout = qtw.QVBoxLayout(self)
        self.layout.addWidget(self.matplolib_canvas)
        self.layout.addWidget(self.button_add_node)
        self.layout.addWidget(self.button_add_edge)
        self.layout.addWidget(self.button_remove_node)
        self.layout.addWidget(self.button_remove_edge)
        self.layout.addWidget(self.button_generate_wffile)
        self.setLayout(self.layout)

    def draw_graph(self):
        self.matplolib_canvas.figure.clf()
        pos = graphviz_layout(self.G, prog='dot')
        nx.draw(self.G, pos=pos, with_labels=True)
        self.matplolib_canvas.canvas.draw()

    def add_node(self):
        dialog_add_node = CustomDialog(window_title="Add tasks", line_edit_text="Task id(s)")
        dialog_add_node.exec()
        input = dialog_add_node.input
        print(input)
        try:
            nodes = eval(input)
            if type(nodes) == int:
                nodes = [nodes]
            self.G.add_nodes_from(nodes)
            self.draw_graph()
        except NameError:
            message = "If you want to add tasks named using letters use quotes."
            qtw.QMessageBox.about(self, "Error", message)
        # except:
        #     print("Error when adding node")  # eval on cancel button

    def add_edge(self):
        dialog_add_node = CustomDialog(window_title="Add edges", line_edit_text="Task ids")
        dialog_add_node.exec()
        input = dialog_add_node.input
        print("input ", input)
        try:
            nodes = eval(input)
            print("nodes ", nodes)
            if not isinstance(nodes[0], tuple):  # if we have a single tuple we need to enclose it in a list
                nodes = [nodes]
            self.G.add_edges_from(nodes)
            self.draw_graph()
        except NameError:
            message = "If you want to add edges between tasks named using letters use quotes."
            qtw.QMessageBox.about(self, "Error", message)
            print(message)
        # except:
        #     print("error when adding edge")  # eval on cancel button

    def remove_node(self):
        dialog_add_node = CustomDialog(window_title="Remove tasks", line_edit_text="Task ids")
        dialog_add_node.exec()
        input = dialog_add_node.input
        try:
            nodes = eval(input)
            if type(nodes) == int:
                nodes = [nodes]
            self.G.remove_nodes_from(nodes)
            self.draw_graph()
        except NameError:
            message = "If you want to remove tasks named using letters use quotes."
            qtw.QMessageBox.about(self, "Error", message)
            print(message)
        except:
            print("error when removing node")  # eval on cancel button

    def remove_edge(self):
        dialog_add_node = CustomDialog(window_title="Remove edges", line_edit_text="Task ids")
        dialog_add_node.exec()
        input = dialog_add_node.input
        try:
            nodes = eval(input)
            print((nodes), len(nodes), type(nodes))

            if isinstance(nodes[0], int):  # if we have a single tuple we need to enclose it in a list
                nodes = [nodes]
            self.G.remove_edges_from(nodes)
            self.draw_graph()
        except NameError:
            message = "If you want to remove edges between tasks named using letters use quotes."
            qtw.QMessageBox.about(self, "Error", message)
            print(message)
        except:
            print("error when removing edges")  # eval on cancel button

    def generate_wffile(self):
        recipe = CustomRecipe()
        workflow = recipe.build_workflow(self.G)
        print(workflow.nodes())
        workflow.write_json("./data/workflows/generated_new.json")


if __name__ == '__main__':
    app = qtw.QApplication([])
    qt_app = MyQtApp()
    qt_app.show()
    app.exec_()
