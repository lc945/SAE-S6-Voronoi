import pytest
import numpy as np
from main import parse_points_file, generate_voronoi_grid

def test_parse_points_file_valid(tmp_path):
    """Teste la lecture d'un fichier valide avec des entiers et des flottants."""
    file = tmp_path / "points.txt"
    file.write_text("2,4\n5.3,4.5\n18,29\n12.5,23.7")
    
    points = parse_points_file(str(file))
    
    assert points.shape == (4, 2)
    assert np.allclose(points[0], [2.0, 4.0])
    assert np.allclose(points[1], [5.3, 4.5])

def test_parse_points_file_not_found():
    """Teste le déclenchement de l'erreur si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError, match="introuvable"):
        parse_points_file("chemin_inexistant.txt")

def test_parse_points_file_invalid_format(tmp_path):
    """Teste la détection de lignes qui ne sont pas des coordonnées (x,y)."""
    file = tmp_path / "bad_points.txt"
    file.write_text("2,4\ntexte,invalide\n18,29")
    
    with pytest.raises(ValueError, match="Coordonnées non numériques"):
        parse_points_file(str(file))

def test_parse_points_file_too_few_elements(tmp_path):
    """Teste la détection de lignes ne contenant qu'une seule valeur."""
    file = tmp_path / "bad_points_2.txt"
    file.write_text("2,4\n5.3\n18,29")
    
    with pytest.raises(ValueError, match="Format incorrect"):
        parse_points_file(str(file))

def test_parse_points_file_not_enough_points(tmp_path):
    """Teste la gestion des fichiers contenant moins de 2 points."""
    file = tmp_path / "short_points.txt"
    file.write_text("2,4\n")
    
    with pytest.raises(ValueError, match="au moins 2 points"):
        parse_points_file(str(file))

def test_generate_voronoi_grid():
    """Teste la génération de la matrice de Voronoï sur des points simples."""
    points = np.array([[0, 0], [10, 10]])
    res = 10
    X, Y, Z = generate_voronoi_grid(points, resolution=res)
    
    # Vérification des formes
    assert X.shape == (res, res)
    assert Y.shape == (res, res)
    assert Z.shape == (res, res)
    
    # Le point [0,0] est à l'index 0, le point [10,10] est à l'index 1.
    # En bas à gauche (première coordonnée), la valeur doit appartenir au point 0.
    assert Z[0, 0] == 0
    # En haut à droite, la valeur doit appartenir au point 1.
    assert Z[-1, -1] == 1

def test_generate_voronoi_grid_aligned():
    """Teste que l'algorithme ne crashe pas sur des points alignés."""
    points = np.array([[1, 1], [2, 1], [3, 1]])
    X, Y, Z = generate_voronoi_grid(points, resolution=50)
    assert Z.shape == (50, 50)
    # On s'assure que les 3 zones ont bien été générées
    assert len(np.unique(Z)) == 3