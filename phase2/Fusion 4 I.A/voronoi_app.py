"""
Application de génération et visualisation de diagrammes de Voronoï.

Fonctionnalités :
- Lecture robuste d'un fichier texte de coordonnées (x,y).
- Génération de diagramme de Voronoï par approche de champ de distance vectorisé (Numpy).
- Visualisation interactive avec Matplotlib intégrée dans Tkinter.
- Exportations en PNG et SVG.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================================
# LOGIQUE MÉTIER (Traitement et Algorithme)
# ==========================================

def read_points_file(filepath: str) -> np.ndarray:
    """
    Lit un fichier texte et extrait les coordonnées des points.

    Format attendu : une paire de coordonnées (x,y) par ligne, séparée par une virgule.

    Args:
        filepath (str): Le chemin vers le fichier texte.

    Returns:
        np.ndarray: Un tableau numpy de forme (N, 2) contenant les points.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le format est incorrect ou s'il y a moins de 2 points.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier spécifié est introuvable : {filepath}")

    points = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Ignorer les lignes vides
            
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Format incorrect à la ligne {line_idx}. Attendu : x,y")
            
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                points.append([x, y])
            except ValueError:
                raise ValueError(f"Valeurs non numériques à la ligne {line_idx}.")

    if len(points) < 2:
        raise ValueError("Le fichier doit contenir au moins 2 points pour générer un diagramme.")

    return np.array(points)


def generate_voronoi_grid(points: np.ndarray, resolution: int = 800, padding: float = 0.1) -> tuple:
    """
    Génère une grille discrète représentant le diagramme de Voronoï.
    
    Complexité optimisée : Utilise des opérations vectorisées Numpy pour 
    calculer les distances au carré, évitant les lentes boucles Python imbriquées.

    Args:
        points (np.ndarray): Tableau des coordonnées des points (N, 2).
        resolution (int): Nombre de pixels pour la largeur et la hauteur.
        padding (float): Marge autour des points extrêmes.

    Returns:
        tuple: (X, Y, Z) où X et Y sont les grilles de coordonnées (meshgrid), 
               et Z est la matrice des indices du point le plus proche.
    """
    # Calcul de la boîte englobante (bounding box)
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    range_x = max_x - min_x if max_x > min_x else 1.0
    range_y = max_y - min_y if max_y > min_y else 1.0

    x_start, x_end = min_x - padding * range_x, max_x + padding * range_x
    y_start, y_end = min_y - padding * range_y, max_y + padding * range_y

    # Création de la grille (Meshgrid)
    x = np.linspace(x_start, x_end, resolution)
    y = np.linspace(y_start, y_end, resolution)
    X, Y = np.meshgrid(x, y)

    # Initialisation de la carte des zones (Z) et des distances minimales
    Z = np.zeros((resolution, resolution), dtype=int)
    min_dists = np.full((resolution, resolution), np.inf)

    # Opération vectorisée pour trouver le point le plus proche de chaque pixel
    for i, point in enumerate(points):
        # Distance euclidienne au carré (plus rapide car sans racine carrée)
        dist_sq = (X - point[0])**2 + (Y - point[1])**2
        mask = dist_sq < min_dists
        min_dists[mask] = dist_sq[mask]
        Z[mask] = i

    return X, Y, Z


# ==========================================
# INTERFACE UTILISATEUR (Tkinter + Matplotlib)
# ==========================================

class VoronoiApp:
    """Interface graphique principale pour l'application Voronoï."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Générateur de Diagramme de Voronoï")
        self.root.geometry("900x650")
        self.root.configure(bg="#2b2b2b")
        
        self.points = None
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self._build_ui()

    def _build_ui(self):
        """Construit les éléments de l'interface graphique."""
        # Panneau de contrôle à gauche
        control_panel = tk.Frame(self.root, width=250, bg="#2b2b2b", padx=15, pady=20)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_panel, text="Diagramme de Voronoï", font=("Arial", 14, "bold"), 
                 fg="white", bg="#2b2b2b").pack(pady=(0, 20))

        # Bouton Charger
        tk.Button(control_panel, text="📂 Charger un fichier", command=self.load_file,
                  bg="#4a90d9", fg="white", relief=tk.FLAT, width=20, pady=5).pack(pady=10)

        self.file_label = tk.Label(control_panel, text="Aucun fichier chargé", 
                                   fg="#aaaaaa", bg="#2b2b2b", wraplength=200)
        self.file_label.pack(pady=5)

        tk.Label(control_panel, text="─" * 25, fg="#555555", bg="#2b2b2b").pack(pady=15)

        # Boutons d'exportation
        tk.Label(control_panel, text="Exportation", font=("Arial", 11, "bold"), 
                 fg="white", bg="#2b2b2b").pack(pady=5)

        self.btn_png = tk.Button(control_panel, text="💾 Exporter en PNG", command=lambda: self.export_image('png'),
                                 bg="#27ae60", fg="white", relief=tk.FLAT, width=20, state=tk.DISABLED)
        self.btn_png.pack(pady=5)

        self.btn_svg = tk.Button(control_panel, text="💾 Exporter en SVG", command=lambda: self.export_image('svg'),
                                 bg="#27ae60", fg="white", relief=tk.FLAT, width=20, state=tk.DISABLED)
        self.btn_svg.pack(pady=5)

        # Zone d'affichage du graphique à droite
        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.set_title("Veuillez charger un fichier de points.")
        self.ax.axis('off')
        self.canvas.draw()

    def load_file(self):
        """Gère la sélection du fichier et lance le calcul."""
        filepath = filedialog.askopenfilename(
            title="Sélectionnez le fichier de points",
            filetypes=(("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*"))
        )
        
        if not filepath:
            return

        try:
            self.points = read_points_file(filepath)
            filename = os.path.basename(filepath)
            self.file_label.config(text=f"✔ {filename}\n({len(self.points)} points)", fg="#27ae60")
            
            self.plot_voronoi()
            
            # Activation des boutons d'export
            self.btn_png.config(state=tk.NORMAL)
            self.btn_svg.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            self.file_label.config(text="Erreur de chargement", fg="#e74c3c")

    def plot_voronoi(self):
        """Affiche le diagramme sur le canevas Matplotlib."""
        self.ax.clear()
        
        # Génération
        X, Y, Z = generate_voronoi_grid(self.points)
        
        # Affichage de la carte des couleurs
        self.ax.imshow(
            Z, 
            extent=(X.min(), X.max(), Y.min(), Y.max()), 
            origin='lower',
            cmap='tab20', 
            alpha=0.6, 
            aspect='equal'
        )
        
        # Affichage des points par-dessus
        self.ax.plot(self.points[:, 0], self.points[:, 1], 'ko', markersize=4, label='Points germes')
        
        # Annotations des coordonnées
        for (px, py) in self.points:
            self.ax.annotate(f"({px}, {py})", (px, py), xytext=(4, 4), 
                             textcoords='offset points', fontsize=8)

        self.ax.set_title("Diagramme de Voronoï")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.axis('on')
        self.fig.tight_layout()
        self.canvas.draw()

    def export_image(self, fmt: str):
        """Exporte le graphique au format demandé (png ou svg)."""
        if self.points is None:
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{fmt}",
            filetypes=((f"Fichier {fmt.upper()}", f"*.{fmt}"),)
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Succès", f"Fichier exporté avec succès :\n{filepath}")
            except Exception as e:
                messagebox.showerror("Erreur d'exportation", f"Impossible de sauvegarder : {str(e)}")


def main():
    """Point d'entrée principal de l'application."""
    root = tk.Tk()
    app = VoronoiApp(root)
    # Fermeture propre pour libérer la mémoire Matplotlib
    root.protocol("WM_DELETE_WINDOW", lambda: (plt.close('all'), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
