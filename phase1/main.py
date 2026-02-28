import numpy as np      
import matplotlib.pyplot as plt
import math 
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def lire_coordonnees(nom_fichier):
    """Lit un fichier texte et retourne une liste de points."""
    points = []
    try:
        with open(nom_fichier, 'r') as fichier:
            for ligne in fichier:
                valeurs = ligne.strip().split(',') # On sépare les coordonnées quand il y a une virgule 
                if len(valeurs) == 2: 
                    points.append((float(valeurs[0]), float(valeurs[1]))) # On met toutes les valeurs en FLOAT.
    except FileNotFoundError: # Si le fichier est pas trouvé on envoie un message d'erreur à l'utilisateur
        print(f"Attention: Ton fichier {nom_fichier} n'a pas été trouvé !")
    return points

def calculer_distance(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # On calcule la distance Euclidienne point 1 et point 2 
    
    return distance

def trouver_point_plus_proche(pixel_x, pixel_y, liste_points):
    distance_min = float('inf')
    index_du_plus_proche = -1
    for index in range(len(liste_points)):
        point = liste_points[index]
        distance = calculer_distance((pixel_x, pixel_y), point)

        if distance < distance_min:
            distance_min = distance
            index_du_plus_proche = index
            
    return index_du_plus_proche

def generer_grille(mes_points):
    """Génère la grille de Voronoi"""
    if not mes_points:
        print("Erreur: pas de points!")
        return None, None
    
    taille_max = 0
    for point in mes_points:
        if point[0] > taille_max:
            taille_max = int(point[0])
        if point[1] > taille_max:
            taille_max = int(point[1])
    
    taille_max = taille_max + 10
    
    grille = np.zeros((taille_max, taille_max))
    
    for y in range(taille_max):
        for x in range(taille_max):
            grille[y, x] = trouver_point_plus_proche(x, y, mes_points)
    
    return grille, taille_max


def exporter_png(grille, mes_points, taille_max, nom_fichier):
    """Exporte en PNG"""
    plt.figure(figsize=(10, 10))
    plt.imshow(grille, origin="lower")
    
    for point in mes_points:
        plt.scatter(point[0], point[1], color='red', s=100)
    
    plt.title(f"Voronoi ({len(mes_points)} points)")
    plt.savefig(nom_fichier, dpi=150)
    plt.close()
    print(f"✓ PNG exporté: {nom_fichier}")

def exporter_svg(grille, mes_points, taille_max, nom_fichier):
    """Exporte en SVG"""
    plt.figure(figsize=(10, 10))
    plt.imshow(grille, origin="lower")
    
    for point in mes_points:
        plt.scatter(point[0], point[1], color='red', s=100)
    
    plt.title(f"Voronoi ({len(mes_points)} points)")
    plt.savefig(nom_fichier, format='svg')
    plt.close()
    print(f"✓ SVG exporté: {nom_fichier}")

# ============ INTERFACE ============

class AppVoronoi:
    def __init__(self):
        # On initialise les variables dont on aura besoin dans toute l'appli
        self.mes_points = []   # contiendra les points lus depuis le fichier
        self.grille = None     # la grille voronoi generee
        self.taille_max = None # taille de la grille
        
        # Creation de la fenetre principale
        self.fenetre = tk.Tk()
        self.fenetre.title("Voronoi")
        self.fenetre.geometry("900x700")
        
        # On cree une zone en haut pour mettre les boutons
        frame_boutons = tk.Frame(self.fenetre)
        frame_boutons.pack(fill=tk.X, padx=10, pady=10)
        
        # Les 3 boutons principaux, chacun appelle une methode differente
        tk.Button(frame_boutons, text="Charger fichier", command=self.charger, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_boutons, text="Exporter PNG", command=self.export_png, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_boutons, text="Exporter SVG", command=self.export_svg, width=20).pack(side=tk.LEFT, padx=5)
        
        # Label pour informer l'utilisateur de ce qui se passe
        self.label = tk.Label(self.fenetre, text="Prêt - Charger un fichier", fg="blue")
        self.label.pack(pady=5)
        
        # Zone ou on va afficher le diagramme
        self.frame_canvas = tk.Frame(self.fenetre)
        self.frame_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def charger(self):
        # On ouvre une boite de dialogue pour choisir le fichier txt
        fichier = filedialog.askopenfilename(filetypes=[("Texte", "*.txt")])
        if fichier:
            self.mes_points = lire_coordonnees(fichier)
            if self.mes_points:
                # On previent l'utilisateur que ca calcule
                self.label.config(text="en cours...", fg="orange")
                self.fenetre.update() # important sinon le label se met pas a jour visuellement
                
                # On genere la grille puis on affiche
                self.grille, self.taille_max = generer_grille(self.mes_points)
                self.afficher()
                
                self.label.config(text=f"✓ {len(self.mes_points)} points chargés", fg="green")
    
    def afficher(self):
        # Si on charge un nouveau fichier, on efface d'abord l'ancien diagramme
        for widget in self.frame_canvas.winfo_children():
            widget.destroy()
        
        # On cree la figure matplotlib avec la grille voronoi
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        ax.imshow(self.grille, origin="lower" ) 
        
        # On affiche les points par dessus en rouge pour qu on les voit bien
        for point in self.mes_points:
            ax.scatter(point[0], point[1], color='red', s=100, edgecolor='black', linewidth=1)
        
        ax.set_title(f"Voronoi ({len(self.mes_points)} points)")
        
        # On integre la figure matplotlib dans la fenetre tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def export_png(self):
        # On verifie qu'un diagramme a bien ete genere avant d'exporter
        if self.grille is None:
            self.label.config(text="Charger un fichier d'abord!", fg="red")
            return
        # Boite de dialogue pour choisir ou sauvegarder
        fichier = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if fichier:
            exporter_png(self.grille, self.mes_points, self.taille_max, fichier)
            self.label.config(text="✓ PNG exporté", fg="green")
    
    def export_svg(self):
        # Pareil que pour le PNG mais en SVG
        if self.grille is None:
            self.label.config(text="Charger un fichier d'abord!", fg="red")
            return
        fichier = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if fichier:
            exporter_svg(self.grille, self.mes_points, self.taille_max, fichier)
            self.label.config(text="✓ SVG exporté", fg="green")
    
    def run(self):
        self.fenetre.mainloop()

app = AppVoronoi()
app.run()