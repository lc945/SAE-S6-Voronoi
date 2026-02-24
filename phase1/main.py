import numpy as np      
import matplotlib.pyplot as plt
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

print(lire_coordonnees("points.txt"))

mes_points = lire_coordonnees("points.txt")

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

print(calculer_distance((0, 0), (3, 4)))

point_A = mes_points[0]
point_B = mes_points[1]

test_avec_coordonnee_fichier = calculer_distance(point_A, point_B)
print("Le point A est : ", point_A)
print("Le point B est : ", point_B)
print("La distance entre les deux est : ", round(test_avec_coordonnee_fichier,2)) # round c'est pour arrondir le résultat à 2 chiffre apres la virgule.

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

plt.figure(figsize=(8, 8))

plt.imshow(grille, origin = "lower")

for point in mes_points:
    plt.scatter(point[0], point[1], color='red', s=50)

plt.show()