"""
Tests unitaires pour l'application Voronoï.
Exécuter avec la commande : pytest test_voronoi.py -v
"""

import os
import numpy as np
import pytest
from voronoi_app import read_points_file, generate_voronoi_grid

# ==========================================
# FIXTURES (Données de test)
# ==========================================

@pytest.fixture
def valid_file(tmp_path):
    """Crée un fichier temporaire avec des points valides."""
    content = "2,4\n5.3,4.5\n18,29\n12.5,23.7\n"
    file = tmp_path / "points.txt"
    file.write_text(content, encoding="utf-8")
    return str(file)

@pytest.fixture
def invalid_format_file(tmp_path):
    """Crée un fichier temporaire avec un format incorrect."""
    content = "2,4\nabc,def\n18,29\n"
    file = tmp_path / "bad_format.txt"
    file.write_text(content, encoding="utf-8")
    return str(file)

@pytest.fixture
def insufficient_points_file(tmp_path):
    """Crée un fichier avec un seul point."""
    content = "5.5,10.0\n"
    file = tmp_path / "one_point.txt"
    file.write_text(content, encoding="utf-8")
    return str(file)

# ==========================================
# TESTS : read_points_file
# ==========================================

class TestReadPointsFile:
    
    def test_lecture_fichier_valide(self, valid_file):
        """Vérifie qu'un fichier valide est correctement lu en array numpy."""
        points = read_points_file(valid_file)
        assert isinstance(points, np.ndarray)
        assert points.shape == (4, 2)
        assert np.allclose(points[0], [2.0, 4.0])
        assert np.allclose(points[1], [5.3, 4.5])

    def test_fichier_inexistant(self):
        """Vérifie l'erreur si le fichier n'existe pas."""
        with pytest.raises(FileNotFoundError, match="introuvable"):
            read_points_file("chemin_fictif_inexistant.txt")

    def test_format_incorrect(self, invalid_format_file):
        """Vérifie l'erreur levée si des lettres sont présentes au lieu de nombres."""
        with pytest.raises(ValueError, match="Valeurs non numériques"):
            read_points_file(invalid_format_file)

    def test_points_insuffisants(self, insufficient_points_file):
        """Vérifie l'erreur levée s'il y a strictement moins de 2 points."""
        with pytest.raises(ValueError, match="au moins 2 points"):
            read_points_file(insufficient_points_file)

    def test_lignes_vides_ignorees(self, tmp_path):
        """Vérifie que les lignes vides ou contenant des espaces sont bien ignorées."""
        content = "1,2\n\n  \n3,4\n\n"
        file = tmp_path / "avec_vides.txt"
        file.write_text(content, encoding="utf-8")
        points = read_points_file(str(file))
        assert points.shape == (2, 2)


# ==========================================
# TESTS : generate_voronoi_grid
# ==========================================

class TestGenerateVoronoiGrid:
    
    def test_dimensions_retour(self):
        """Vérifie que la matrice retournée correspond à la résolution demandée."""
        points = np.array([[0, 0], [10, 10]])
        res = 50
        X, Y, Z = generate_voronoi_grid(points, resolution=res)
        
        assert X.shape == (res, res)
        assert Y.shape == (res, res)
        assert Z.shape == (res, res)

    def test_logique_zones(self):
        """Vérifie que l'algorithme attribue correctement les pixels aux points proches."""
        points = np.array([[0, 0], [10, 10]])
        res = 10
        X, Y, Z = generate_voronoi_grid(points, resolution=res, padding=0.0)
        
        # Z contient les indices des points.
        # Le coin inférieur gauche doit appartenir au point 0 ([0,0])
        assert Z[0, 0] == 0
        
        # Le coin supérieur droit doit appartenir au point 1 ([10,10])
        assert Z[-1, -1] == 1

    def test_alignement_points(self):
        """Vérifie la robustesse avec des points purement alignés (cas particulier géométrique)."""
        points = np.array([[1, 1], [2, 1], [3, 1]])
        X, Y, Z = generate_voronoi_grid(points, resolution=30)
        
        # Vérifie qu'aucune erreur LinAlg n'est levée et que 3 zones distinctes sont créées
        assert len(np.unique(Z)) == 3
