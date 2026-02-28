import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def parse_points_file(filepath):
    """
    Lit un fichier texte et extrait les coordonnées des points.
    
    Format attendu : une paire de coordonnées (x,y) par ligne, séparée par une virgule.
    
    Args:
        filepath (str): Le chemin vers le fichier texte.
        
    Returns:
        np.ndarray: Un tableau numpy de forme (N, 2) contenant les points.
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le format du fichier est invalide ou s'il y a moins de 2 points.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier '{filepath}' est introuvable.")

    points = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Format incorrect ligne {line_idx + 1}. Attendu : x,y")
            
            try:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
            except ValueError:
                raise ValueError(f"Coordonnées non numériques ligne {line_idx + 1}.")

    if len(points) < 2:
        raise ValueError("Le fichier doit contenir au moins 2 points pour générer un diagramme.")

    return np.array(points)


def generate_voronoi_grid(points, resolution=800, padding=0.2):
    """
    Génère une grille discrète représentant le diagramme de Voronoï.
    Utilise une approche de champ de distance minimisée pour éviter la dépendance à scipy.
    
    Args:
        points (np.ndarray): Tableau des coordonnées des points (N, 2).
        resolution (int): Nombre de pixels pour la largeur et la hauteur de la grille.
        padding (float): Marge autour des points extrêmes (en pourcentage de la taille).
        
    Returns:
        tuple: (X, Y, Z) où X et Y sont les grilles de coordonnées, 
               et Z est la matrice des indices du point le plus proche.
    """
    # Calcul de la boîte englobante (bounding box)
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    x_start, x_end = min_x - padding * range_x, max_x + padding * range_x
    y_start, y_end = min_y - padding * range_y, max_y + padding * range_y

    # Création de la grille
    x = np.linspace(x_start, x_end, resolution)
    y = np.linspace(y_start, y_end, resolution)
    X, Y = np.meshgrid(x, y)

    # Initialisation de la carte des zones (Z) et des distances minimales
    Z = np.zeros((resolution, resolution), dtype=int)
    min_dists = np.full((resolution, resolution), np.inf)

    # Optimisation mémoire et CPU : calcul itératif des distances au carré
    for i, point in enumerate(points):
        dist_sq = (X - point[0])**2 + (Y - point[1])**2
        mask = dist_sq < min_dists
        min_dists[mask] = dist_sq[mask]
        Z[mask] = i

    return X, Y, Z


class VoronoiApp:
    """Interface utilisateur principale pour l'application Voronoï."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de Diagramme de Voronoï")
        self.root.geometry("800x700")
        
        self.points = None
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        
        self.create_widgets()

    def create_widgets(self):
        """Initialise les composants de l'interface graphique Tkinter."""
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=5, padx=5)

        tk.Button(toolbar, text="Charger Fichier (.txt)", command=self.load_file).pack(side=tk.LEFT, padx=5)
        
        self.btn_png = tk.Button(toolbar, text="Exporter PNG", command=lambda: self.export_image('png'), state=tk.DISABLED)
        self.btn_png.pack(side=tk.LEFT, padx=5)
        
        self.btn_svg = tk.Button(toolbar, text="Exporter SVG", command=lambda: self.export_image('svg'), state=tk.DISABLED)
        self.btn_svg.pack(side=tk.LEFT, padx=5)

        # Intégration de Matplotlib dans Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.ax.set_title("Veuillez charger un fichier de points.")
        self.ax.axis('off')
        self.canvas.draw()

    def load_file(self):
        """Ouvre un dialogue pour sélectionner le fichier et lance la génération."""
        filepath = filedialog.askopenfilename(
            title="Sélectionnez le fichier de points",
            filetypes=(("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*"))
        )
        
        if not filepath:
            return

        try:
            self.points = parse_points_file(filepath)
            self.plot_voronoi()
            self.btn_png.config(state=tk.NORMAL)
            self.btn_svg.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Erreur de chargement", str(e))

    def plot_voronoi(self):
        """Génère la grille et affiche le diagramme sur le canevas Matplotlib."""
        self.ax.clear()
        self.ax.set_title("Génération en cours...")
        self.canvas.draw()
        
        # Génération
        X, Y, Z = generate_voronoi_grid(self.points)
        
        # Affichage
        self.ax.clear()
        self.ax.imshow(
            Z, 
            extent=(X.min(), X.max(), Y.min(), Y.max()), 
            origin='lower',
            cmap='tab20', 
            alpha=0.6, 
            aspect='auto'
        )
        self.ax.plot(self.points[:, 0], self.points[:, 1], 'ko', markersize=5, label='Points')
        
        self.ax.set_title("Diagramme de Voronoï")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend(loc='upper right')
        
        self.fig.tight_layout()
        self.canvas.draw()

    def export_image(self, fmt):
        """Exporte le graphique affiché dans le format souhaité (png ou svg)."""
        if self.points is None:
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=((f"Fichier {fmt.upper()}", f"*.{fmt}"),)
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Succès", f"Fichier exporté avec succès sous :\n{filepath}")
            except Exception as e:
                messagebox.showerror("Erreur d'exportation", f"Impossible de sauvegarder : {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoronoiApp(root)
    # Gestion de la fermeture propre de matplotlib
    root.protocol("WM_DELETE_WINDOW", lambda: (plt.close('all'), root.destroy()))
    root.mainloop()