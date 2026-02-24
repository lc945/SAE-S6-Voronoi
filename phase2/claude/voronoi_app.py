"""
Application de génération et visualisation de diagrammes de Voronoï.
Phase 2 - SAÉ S6
"""

import math
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg


# ─────────────────────────────────────────────
# 1. Lecture du fichier
# ─────────────────────────────────────────────

def lire_coordonnees(nom_fichier: str) -> list[tuple[float, float]]:
    """
    Lit un fichier texte et retourne une liste de points (x, y).

    Le fichier doit contenir une paire de coordonnées par ligne,
    séparées par une virgule. Exemple :
        2,4
        5.3,4.5

    Args:
        nom_fichier: Chemin vers le fichier à lire.

    Returns:
        Liste de tuples (x, y) en float.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier contient moins de 2 points valides.
    """
    points = []

    with open(nom_fichier, 'r', encoding='utf-8') as fichier:
        for numero_ligne, ligne in enumerate(fichier, start=1):
            ligne = ligne.strip()
            if not ligne:
                continue  # Ignorer les lignes vides
            valeurs = ligne.split(',')
            if len(valeurs) != 2:
                raise ValueError(
                    f"Ligne {numero_ligne} mal formatée : '{ligne}'. "
                    "Format attendu : x,y"
                )
            try:
                x = float(valeurs[0])
                y = float(valeurs[1])
            except ValueError:
                raise ValueError(
                    f"Ligne {numero_ligne} : impossible de convertir "
                    f"'{valeurs[0]}' ou '{valeurs[1]}' en nombre."
                )
            points.append((x, y))

    if len(points) < 2:
        raise ValueError(
            f"Le fichier doit contenir au moins 2 points. "
            f"Seulement {len(points)} point(s) trouvé(s)."
        )

    return points


# ─────────────────────────────────────────────
# 2. Calcul des distances
# ─────────────────────────────────────────────

def calculer_distance(point1: tuple[float, float],
                      point2: tuple[float, float]) -> float:
    """
    Calcule la distance euclidienne entre deux points.

    Args:
        point1: Tuple (x1, y1).
        point2: Tuple (x2, y2).

    Returns:
        Distance euclidienne (float).
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def trouver_site_le_plus_proche(pixel_x: float,
                                pixel_y: float,
                                liste_points: list[tuple[float, float]]) -> int:
    """
    Retourne l'index du site le plus proche d'un pixel donné.

    Args:
        pixel_x: Coordonnée x du pixel.
        pixel_y: Coordonnée y du pixel.
        liste_points: Liste des sites (points germes).

    Returns:
        Index (int) du site le plus proche dans liste_points.
    """
    distance_min = float('inf')
    index_plus_proche = 0

    for index, point in enumerate(liste_points):
        distance = calculer_distance((pixel_x, pixel_y), point)
        if distance < distance_min:
            distance_min = distance
            index_plus_proche = index

    return index_plus_proche


# ─────────────────────────────────────────────
# 3. Normalisation des points
# ─────────────────────────────────────────────

def normaliser_points(points: list[tuple[float, float]],
                      largeur: int = 500,
                      hauteur: int = 500,
                      marge: int = 30) -> list[tuple[float, float]]:
    """
    Normalise les points pour les adapter à la taille de la grille.

    Recentre les points en soustrayant les minimums, puis les étire
    pour occuper toute la surface disponible en respectant une marge.

    Args:
        points: Liste de points bruts.
        largeur: Largeur de la grille en pixels.
        hauteur: Hauteur de la grille en pixels.
        marge: Marge en pixels autour des bords.

    Returns:
        Liste de points normalisés.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Éviter la division par zéro si tous les points ont la même coordonnée
    plage_x = x_max - x_min if x_max != x_min else 1.0
    plage_y = y_max - y_min if y_max != y_min else 1.0

    points_normalises = []
    for px, py in points:
        nx = marge + (px - x_min) / plage_x * (largeur - 2 * marge)
        ny = marge + (py - y_min) / plage_y * (hauteur - 2 * marge)
        points_normalises.append((nx, ny))

    return points_normalises


# ─────────────────────────────────────────────
# 4. Génération de la grille Voronoï
# ─────────────────────────────────────────────

def generer_grille_voronoi(points: list[tuple[float, float]],
                           largeur: int = 500,
                           hauteur: int = 500) -> np.ndarray:
    """
    Génère la grille de Voronoï par force brute.

    Pour chaque pixel de la grille, détermine quel site est le plus
    proche et stocke son index. Complexité : O(largeur × hauteur × n).

    Args:
        points: Liste des sites (déjà normalisés).
        largeur: Largeur de la grille.
        hauteur: Hauteur de la grille.

    Returns:
        Tableau numpy 2D contenant l'index du site le plus proche
        pour chaque pixel.
    """
    grille = np.zeros((hauteur, largeur), dtype=int)

    for y in range(hauteur):
        for x in range(largeur):
            grille[y, x] = trouver_site_le_plus_proche(x, y, points)

    return grille


# ─────────────────────────────────────────────
# 5. Affichage matplotlib
# ─────────────────────────────────────────────

def creer_figure_voronoi(points_originaux: list[tuple[float, float]],
                         points_normalises: list[tuple[float, float]],
                         grille: np.ndarray) -> plt.Figure:
    """
    Crée et retourne une figure matplotlib du diagramme de Voronoï.

    Args:
        points_originaux: Points d'origine (pour les labels).
        points_normalises: Points normalisés (pour l'affichage sur la grille).
        grille: Grille Voronoï générée.

    Returns:
        Objet Figure matplotlib.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(grille, origin='lower', cmap='tab20')

    for i, (px, py) in enumerate(points_normalises):
        ax.scatter(px, py, color='red', s=60, zorder=5)
        ox, oy = points_originaux[i]
        ax.annotate(
            f"({ox}, {oy})",
            xy=(px, py),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            color='white'
        )

    ax.set_title("Diagramme de Voronoï", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    return fig


# ─────────────────────────────────────────────
# 6. Export
# ─────────────────────────────────────────────

def exporter_image(fig: plt.Figure, chemin: str) -> None:
    """
    Exporte la figure matplotlib en image PNG.

    Args:
        fig: Figure matplotlib à exporter.
        chemin: Chemin de destination du fichier PNG.
    """
    fig.savefig(chemin, format='png', dpi=150, bbox_inches='tight')


def exporter_svg(fig: plt.Figure, chemin: str) -> None:
    """
    Exporte la figure matplotlib en fichier SVG.

    Args:
        fig: Figure matplotlib à exporter.
        chemin: Chemin de destination du fichier SVG.
    """
    fig.savefig(chemin, format='svg', bbox_inches='tight')


# ─────────────────────────────────────────────
# 7. Interface graphique Tkinter
# ─────────────────────────────────────────────

class ApplicationVoronoi(tk.Tk):
    """Interface graphique principale pour l'application Voronoï."""

    LARGEUR_GRILLE = 400
    HAUTEUR_GRILLE = 400

    def __init__(self):
        """Initialise la fenêtre principale et ses composants."""
        super().__init__()

        self.title("Diagramme de Voronoï - SAÉ S6")
        self.resizable(False, False)

        self._points_originaux = []
        self._points_normalises = []
        self._grille = None
        self._figure = None

        self._construire_interface()

        # Charger automatiquement points.txt
        try:
            chemin_points = "../../phase1/points.txt"
            self._points_originaux = lire_coordonnees(chemin_points)
            nom_court = chemin_points.split("/")[-1]
            self._label_fichier.config(
                text=f"✔ {nom_court}\n({len(self._points_originaux)} points)",
                fg="#27ae60"
            )
            self._label_statut.config(text="Fichier points.txt chargé automatiquement.")
        except (FileNotFoundError, ValueError) as e:
            self._label_fichier.config(text="Erreur de chargement points.txt", fg="red")
            self._label_statut.config(text=str(e))

    def _construire_interface(self) -> None:
        """Construit tous les widgets de l'interface."""
        # ── Panneau de contrôle (gauche) ──
        panneau = tk.Frame(self, padx=10, pady=10, bg="#2b2b2b")
        panneau.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(
            panneau, text="Voronoï", font=("Arial", 16, "bold"),
            fg="white", bg="#2b2b2b"
        ).pack(pady=(0, 20))

        tk.Button(
            panneau, text="📂 Charger un fichier",
            command=self._charger_fichier,
            width=22, bg="#4a90d9", fg="white", relief=tk.FLAT
        ).pack(pady=5)

        self._label_fichier = tk.Label(
            panneau, text="Aucun fichier chargé",
            fg="#aaaaaa", bg="#2b2b2b", wraplength=160
        )
        self._label_fichier.pack(pady=5)

        tk.Button(
            panneau, text="▶ Générer le diagramme",
            command=self._generer,
            width=22, bg="#27ae60", fg="white", relief=tk.FLAT
        ).pack(pady=10)

        tk.Label(panneau, text="─" * 20, fg="#555555", bg="#2b2b2b").pack(pady=5)

        tk.Label(
            panneau, text="Export", font=("Arial", 11, "bold"),
            fg="white", bg="#2b2b2b"
        ).pack()

        tk.Button(
            panneau, text="💾 Exporter PNG",
            command=self._exporter_png,
            width=22, bg="#8e44ad", fg="white", relief=tk.FLAT
        ).pack(pady=5)

        tk.Button(
            panneau, text="💾 Exporter SVG",
            command=self._exporter_svg,
            width=22, bg="#8e44ad", fg="white", relief=tk.FLAT
        ).pack(pady=5)

        self._label_statut = tk.Label(
            panneau, text="", fg="#f0c040", bg="#2b2b2b", wraplength=160
        )
        self._label_statut.pack(pady=10)

        # ── Zone d'affichage (droite) ──
        self._cadre_canvas = tk.Frame(self, bg="#1e1e1e")
        self._cadre_canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self._canvas_tk = None

    def _charger_fichier(self) -> None:
        """Ouvre une boîte de dialogue pour sélectionner le fichier de points."""
        chemin = filedialog.askopenfilename(
            title="Choisir un fichier de points",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not chemin:
            return

        try:
            self._points_originaux = lire_coordonnees(chemin)
            nom_court = chemin.split("/")[-1]
            self._label_fichier.config(
                text=f"✔ {nom_court}\n({len(self._points_originaux)} points)",
                fg="#27ae60"
            )
            self._label_statut.config(text="Fichier chargé. Cliquez sur Générer.")
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Erreur de lecture", str(e))
            self._label_fichier.config(text="Erreur de chargement", fg="red")

    def _generer(self) -> None:
        """Génère et affiche le diagramme de Voronoï."""
        if not self._points_originaux:
            messagebox.showwarning("Aucun fichier", "Veuillez d'abord charger un fichier.")
            return

        self._label_statut.config(text="⏳ Calcul en cours...")
        self.update()

        self._points_normalises = normaliser_points(
            self._points_originaux,
            largeur=self.LARGEUR_GRILLE,
            hauteur=self.HAUTEUR_GRILLE
        )
        self._grille = generer_grille_voronoi(
            self._points_normalises,
            largeur=self.LARGEUR_GRILLE,
            hauteur=self.HAUTEUR_GRILLE
        )
        self._figure = creer_figure_voronoi(
            self._points_originaux,
            self._points_normalises,
            self._grille
        )

        self._afficher_figure()
        self._label_statut.config(text="✔ Diagramme généré !")

    def _afficher_figure(self) -> None:
        """Intègre la figure matplotlib dans la fenêtre Tkinter."""
        if self._canvas_tk:
            self._canvas_tk.get_tk_widget().destroy()

        self._canvas_tk = tkagg.FigureCanvasTkAgg(
            self._figure, master=self._cadre_canvas
        )
        self._canvas_tk.draw()
        self._canvas_tk.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def _exporter_png(self) -> None:
        """Exporte le diagramme au format PNG."""
        if not self._figure:
            messagebox.showwarning("Rien à exporter", "Générez d'abord le diagramme.")
            return
        chemin = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if chemin:
            exporter_image(self._figure, chemin)
            messagebox.showinfo("Export réussi", f"Image sauvegardée :\n{chemin}")

    def _exporter_svg(self) -> None:
        """Exporte le diagramme au format SVG."""
        if not self._figure:
            messagebox.showwarning("Rien à exporter", "Générez d'abord le diagramme.")
            return
        chemin = filedialog.asksaveasfilename(
            defaultextension=".svg",
            filetypes=[("SVG", "*.svg")]
        )
        if chemin:
            exporter_svg(self._figure, chemin)
            messagebox.showinfo("Export réussi", f"SVG sauvegardé :\n{chemin}")


# ─────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = ApplicationVoronoi()
    app.mainloop()
