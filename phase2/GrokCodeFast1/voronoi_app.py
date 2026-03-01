import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox


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
    Placeholder pour compatibilité – le calcul est fait dans plot_voronoi.

    Args:
        points (np.ndarray): Tableau des points.

    Returns:
        None
    """
    return None


def line_intersection(p1, d1, p2, d2):
    """
    Calcule l'intersection de deux lignes définies par un point et une direction.

    Args:
        p1 (np.ndarray): Point sur la première ligne.
        d1 (np.ndarray): Direction de la première ligne.
        p2 (np.ndarray): Point sur la deuxième ligne.
        d2 (np.ndarray): Direction de la deuxième ligne.

    Returns:
        np.ndarray or None: Point d'intersection, ou None si parallèle.
    """
    A = np.array([d1, -d2]).T
    b = p2 - p1
    try:
        ts = np.linalg.solve(A, b)
        return p1 + ts[0] * d1
    except np.linalg.LinAlgError:
        return None


def compute_region(point, points):
    """
    Calcule les sommets approximatifs de la région de Voronoï pour un point.

    Utilise les intersections des bissectrices avec les voisins pour former un polygone.

    Args:
        point (np.ndarray): Le point central.
        points (np.ndarray): Tous les points.

    Returns:
        np.ndarray or None: Sommets du polygone, triés autour du point, ou None.
    """
    neighbors = [p for p in points if not np.allclose(p, point)]
    vertices = []
    for i in range(len(neighbors)):
        for j in range(i+1, len(neighbors)):
            p1 = neighbors[i]
            p2 = neighbors[j]
            mid1 = (point + p1) / 2
            dir1 = p1 - point
            perp1 = np.array([-dir1[1], dir1[0]])
            perp1 = perp1 / np.linalg.norm(perp1)
            mid2 = (point + p2) / 2
            dir2 = p2 - point
            perp2 = np.array([-dir2[1], dir2[0]])
            perp2 = perp2 / np.linalg.norm(perp2)
            inter = line_intersection(mid1, perp1, mid2, perp2)
            if inter is not None:
                vertices.append(inter)
    if vertices:
        # Trier les vertices autour du point par angle
        angles = [np.angle((v - point)[0] + 1j * (v - point)[1]) for v in vertices]
        sorted_indices = np.argsort(angles)
        sorted_vertices = [vertices[i] for i in sorted_indices]
        return np.array(sorted_vertices)
    return None


def plot_voronoi(points, ax=None):
    """
    Trace une approximation du diagramme de Voronoï avec zones colorées et points.

    Calcule et remplit les régions approximatives pour chaque point avec une couleur unique.

    Args:
        points (np.ndarray): Tableau des points.
        ax (matplotlib.axes.Axes, optional): Axe pour tracer.
    """
    if ax is None:
        ax = plt.gca()

    colors = cm.rainbow(np.linspace(0, 1, len(points)))
    for i, point in enumerate(points):
        region = compute_region(point, points)
        if region is not None and len(region) > 2:
            # Fermer le polygone pour fill
            region_closed = np.vstack([region, region[0]])
            ax.fill(region_closed[:, 0], region_closed[:, 1], color=colors[i], alpha=0.5)
            # Tracer le contour noir
            ax.plot(region_closed[:, 0], region_closed[:, 1], 'k-')

    # Tracer les points rouges
    ax.plot(points[:, 0], points[:, 1], 'ro')
    ax.set_aspect('equal')


def export_voronoi(points, filename, format_type):
    """
    Exporte l'approximation du diagramme de Voronoï en SVG ou PNG.

    Args:
        points (np.ndarray): Tableau des points.
        filename (str): Nom du fichier de sortie (sans extension).
        format_type (str): 'svg' ou 'png'.
    """
    fig, ax = plt.subplots()
    plot_voronoi(points, ax)
    plt.title("Approximation du Diagramme de Voronoï")
    plt.savefig(f"{filename}.{format_type}", format=format_type, bbox_inches='tight')
    plt.close(fig)


class VoronoiApp:
    """
    Application Tkinter pour charger, générer et visualiser une approximation du diagramme de Voronoï.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Approximation du Diagramme de Voronoï")
        self.points = None

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
        """Génère l'approximation et l'affiche dans le canvas Tkinter."""
        if self.points is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_voronoi(self.points, ax)
            ax.set_title("Approximation du Diagramme de Voronoï")

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
        """Exporte l'approximation dans le format spécifié."""
        if self.points is not None:
            filename = filedialog.asksaveasfilename(defaultextension=f".{format_type}", filetypes=[(f"Fichiers {format_type.upper()}", f"*.{format_type}")])
            if filename:
                try:
                    export_voronoi(self.points, filename, format_type)
                    messagebox.showinfo("Succès", f"Exporté en {format_type.upper()} : {filename}.{format_type}")
                except Exception as e:
                    messagebox.showerror("Erreur", f"Échec de l'export : {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiApp(root)
    root.mainloop()
