import pandas as pd
import numpy as np      
import matplotlib as  mpl 
import math 

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

mes_points = lire_coordonnees("phase1/points.txt") 

def calculer_distance(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # On calcule la distance Euclidienne point 1 et point 2 
    
    return distance

print(calculer_distance((0, 0), (3, 4)))

point_A = mes_points[0]
point_B = mes_points[1]

test_avec_coordonnee_fichier = calculer_distance(point_A, point_B)
print("Le point A est :", point_A)
print("Le point B est :", point_B)
print("La distance entre les deux est :", round(test_avec_coordonnee_fichier,2) # round c'est pour arrondir le résultat à 2 chiffre apres la virgule.