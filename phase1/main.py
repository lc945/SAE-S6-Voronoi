import pandas as pd
import numpy as np      
import matplotlib as  mpl 

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

print(lire_coordonnees("phase1/points.txt"))