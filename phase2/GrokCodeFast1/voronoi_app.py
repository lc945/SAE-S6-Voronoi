import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import Voronoi
import tkinter as tk
from tkinter import filedialog, messagebox

## Code GrockCodeFast1

def load_points(filename):
    """
    Charge les points depuis un fichier texte.

    Le fichier doit contenir des paires x,y séparées par des virgules,
    une par ligne. Les valeurs peuvent être des entiers ou des flottants.

    Args:
        filename (str): Chemin vers le fichier texte.

    Returns:
        np.ndarray: Tableau numpy des points (shape: (n, 2)).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le format est incorrect ou s'il y a moins de 2 points.
    """
    try:
        points = []
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Ignore les lignes vides
                    continue
                parts = line.split(',')
                if len(parts) != 2:
                    raise ValueError(f"Ligne {line_num}: Doit contenir exactement deux valeurs séparées par une virgule.")
                try:
                    x, y = float(parts[0].strip()), float(parts[1].strip())
                    points.append((x, y))
                except ValueError:
                    raise ValueError(f"Ligne {line_num}: Valeurs non numériques.")
        if len(points) < 2:
            raise ValueError("Au moins 2 points sont requis pour générer un diagramme de Voronoï.")
        return np.array(points)
    except FileNotFoundError:
        raise FileNotFoundError("Le fichier spécifié est introuvable.")
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement : {str(e)}")


def generate_voronoi(points):
    """
    Génère le diagramme de Voronoï à partir des points.

    Args:
        points (np.ndarray): Tableau des points (shape: (n, 2)).

    Returns:
        scipy.spatial.Voronoi: Objet Voronoï contenant les régions et arêtes.
    """
    return Voronoi(points)


def plot_voronoi(vor, points, ax=None):
    """
    Trace le diagramme de Voronoï avec les points superposés.

    Args:
        vor (scipy.spatial.Voronoi): Objet Voronoï.
        points (np.ndarray): Tableau des points.
        ax (matplotlib.axes.Axes, optional): Axe pour tracer. Si None, utilise plt.gca().
    """
    if ax is None:
        ax = plt.gca()
    # Tracer les arêtes du diagramme
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
    # Tracer les points
    ax.plot(points[:, 0], points[:, 1], 'ro')  # Points en rouge
    ax.set_aspect('equal')


def export_voronoi(vor, points, filename, format_type):
    """
    Exporte le diagramme de Voronoï en SVG ou PNG.

    Args:
        vor (scipy.spatial.Voronoi): Objet Voronoï.
        points (np.ndarray): Tableau des points.
        filename (str): Nom du fichier de sortie (sans extension).
        format_type (str): 'svg' ou 'png'.
    """
    fig, ax = plt.subplots()
    plot_voronoi(vor, points, ax)
    plt.title("Diagramme de Voronoï")
    plt.savefig(f"{filename}.{format_type}", format=format_type, bbox_inches='tight')
    plt.close(fig)


class VoronoiApp:
    """
    Application Tkinter pour charger, générer et visualiser un diagramme de Voronoï.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de Diagramme de Voronoï")
        self.points = None
        self.vor = None

        # Widgets
        self.load_button = tk.Button(root, text="Charger Fichier", command=self.load_file)
        self.load_button.pack(pady=10)

        self.generate_button = tk.Button(root, text="Générer et Visualiser", command=self.generate_and_plot, state=tk.DISABLED)
        self.generate_button.pack(pady=10)

        self.export_svg_button = tk.Button(root, text="Exporter en SVG", command=lambda: self.export('svg'), state=tk.DISABLED)
        self.export_svg_button.pack(pady=5)

        self.export_png_button = tk.Button(root, text="Exporter en PNG", command=lambda: self.export('png'), state=tk.DISABLED)
        self.export_png_button.pack(pady=5)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.error_label = tk.Label(root, text="", fg="red")
        self.error_label.pack(pady=10)

    def load_file(self):
        """Ouvre un dialogue pour charger le fichier et active les boutons si réussi."""
        filename = filedialog.askopenfilename(filetypes=[("Fichiers texte", "*.txt")])
        if filename:
            try:
                self.points = load_points(filename)
                self.error_label.config(text="")
                self.generate_button.config(state=tk.NORMAL)
                self.export_svg_button.config(state=tk.DISABLED)
                self.export_png_button.config(state=tk.DISABLED)
            except Exception as e:
                self.error_label.config(text=str(e))
                self.generate_button.config(state=tk.DISABLED)

    def generate_and_plot(self):
        """Génère le Voronoï et l'affiche dans le canvas Tkinter."""
        if self.points is not None:
            self.vor = generate_voronoi(self.points)
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_voronoi(self.vor, self.points, ax)
            ax.set_title("Diagramme de Voronoï")

            # Intégrer dans Tkinter
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.export_svg_button.config(state=tk.NORMAL)
            self.export_png_button.config(state=tk.NORMAL)
            self.error_label.config(text="")

    def export(self, format_type):
        """Exporte le diagramme dans le format spécifié."""
        if self.vor is not None and self.points is not None:
            filename = filedialog.asksaveasfilename(defaultextension=f".{format_type}", filetypes=[(f"Fichiers {format_type.upper()}", f"*.{format_type}")])
            if filename:
                try:
                    export_voronoi(self.vor, self.points, filename, format_type)
                    messagebox.showinfo("Succès", f"Exporté en {format_type.upper()} : {filename}.{format_type}")
                except Exception as e:
                    messagebox.showerror("Erreur", f"Échec de l'export : {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiApp(root)
    root.mainloop()