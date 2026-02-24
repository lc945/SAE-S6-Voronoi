"""
Application de génération et visualisation de diagramme de Voronoï.

Fonctionnalités :
- Lecture d'un fichier texte contenant des points (format : x,y)
- Génération du diagramme de Voronoï (approche géométrique naïve)
- Visualisation avec matplotlib
- Export en PNG et SVG
- Interface graphique avec tkinter
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt


# ==========================
# LOGIQUE MÉTIER
# ==========================

def read_points(file_path):
    """
    Lit un fichier texte contenant des points au format "x,y".

    Parameters
    ----------
    file_path : str
        Chemin vers le fichier texte.

    Returns
    -------
    np.ndarray
        Tableau numpy de shape (n, 2) contenant les points.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le format est incorrect ou moins de 2 points.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Le fichier spécifié est introuvable.")

    points = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Format incorrect à la ligne {line_number}."
                )

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Valeur non numérique à la ligne {line_number}."
                ) from exc

            points.append((x, y))

    if len(points) < 2:
        raise ValueError("Au moins deux points sont requis.")

    return np.array(points)


def compute_perpendicular_bisector(p1, p2):
    """
    Calcule la médiatrice du segment [p1, p2].

    Returns
    -------
    tuple
        (a, b, c) coefficients de la droite ax + by + c = 0
    """
    midpoint = (p1 + p2) / 2
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Droite perpendiculaire
    a = dx
    b = dy
    c = -(a * midpoint[0] + b * midpoint[1])

    return a, b, c


def generate_voronoi_edges(points, bounds):
    """
    Génère les médiatrices pour chaque paire de points.

    Parameters
    ----------
    points : np.ndarray
    bounds : tuple
        (xmin, xmax, ymin, ymax)

    Returns
    -------
    list
        Liste de segments (x_vals, y_vals)
    """
    xmin, xmax, ymin, ymax = bounds
    edges = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b, c = compute_perpendicular_bisector(
                points[i], points[j]
            )

            if abs(b) > 1e-10:
                x_vals = np.linspace(xmin, xmax, 500)
                y_vals = (-a * x_vals - c) / b
            else:
                x_vals = np.full(500, -c / a)
                y_vals = np.linspace(ymin, ymax, 500)

            edges.append((x_vals, y_vals))

    return edges


def plot_voronoi(points, save_path=None):
    """
    Génère et affiche le diagramme de Voronoï.

    Parameters
    ----------
    points : np.ndarray
    save_path : str, optional
        Chemin de sauvegarde sans extension.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()

    xmin, xmax = points[:, 0].min() - 5, points[:, 0].max() + 5
    ymin, ymax = points[:, 1].min() - 5, points[:, 1].max() + 5

    edges = generate_voronoi_edges(
        points, (xmin, xmax, ymin, ymax)
    )

    for x_vals, y_vals in edges:
        ax.plot(x_vals, y_vals)

    ax.scatter(points[:, 0], points[:, 1], zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Diagramme de Voronoï")
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(f"{save_path}.png")
        fig.savefig(f"{save_path}.svg")

    return fig


# ==========================
# INTERFACE UTILISATEUR
# ==========================

class VoronoiApp:
    """
    Interface graphique principale.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de diagramme de Voronoï")

        self.label = tk.Label(
            root,
            text="Charger un fichier contenant des points (x,y)",
        )
        self.label.pack(pady=10)

        self.button = tk.Button(
            root,
            text="Charger un fichier",
            command=self.load_file,
        )
        self.button.pack(pady=5)

    def load_file(self):
        """
        Ouvre une boîte de dialogue pour charger un fichier.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Fichiers texte", "*.txt")]
        )

        if not file_path:
            return

        try:
            points = read_points(file_path)
            plot_voronoi(points, save_path="voronoi_output")
            plt.show()

        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))


def main():
    """
    Point d'entrée principal de l'application.
    """
    root = tk.Tk()
    app = VoronoiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()