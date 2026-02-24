"""
Tests unitaires pour l'application Voronoï - SAÉ S6 Phase 2.
Lancer avec : pytest test_voronoi.py -v
"""

import math
import os
import tempfile

import numpy as np
import pytest

from voronoi_app import (
    lire_coordonnees,
    calculer_distance,
    trouver_site_le_plus_proche,
    normaliser_points,
    generer_grille_voronoi,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def fichier_points_valide(tmp_path):
    """Crée un fichier temporaire avec des points valides."""
    contenu = "2,4\n5.3,4.5\n18,29\n12.5,23.7\n"
    fichier = tmp_path / "points.txt"
    fichier.write_text(contenu, encoding="utf-8")
    return str(fichier)


@pytest.fixture
def fichier_un_seul_point(tmp_path):
    """Crée un fichier temporaire avec un seul point."""
    fichier = tmp_path / "un_point.txt"
    fichier.write_text("5,10\n", encoding="utf-8")
    return str(fichier)


@pytest.fixture
def fichier_mal_formate(tmp_path):
    """Crée un fichier temporaire avec un format incorrect."""
    fichier = tmp_path / "mauvais.txt"
    fichier.write_text("abc,def\n", encoding="utf-8")
    return str(fichier)


@pytest.fixture
def fichier_vide(tmp_path):
    """Crée un fichier temporaire vide."""
    fichier = tmp_path / "vide.txt"
    fichier.write_text("", encoding="utf-8")
    return str(fichier)


@pytest.fixture
def points_simples():
    """Liste de 4 points pour les tests."""
    return [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]


# ─────────────────────────────────────────────
# Tests : lire_coordonnees
# ─────────────────────────────────────────────

class TestLireCoordonnees:
    """Tests pour la fonction lire_coordonnees."""

    def test_lecture_fichier_valide(self, fichier_points_valide):
        """Vérifie qu'un fichier valide est correctement lu."""
        points = lire_coordonnees(fichier_points_valide)
        assert len(points) == 4
        assert points[0] == (2.0, 4.0)
        assert points[1] == (5.3, 4.5)
        assert points[2] == (18.0, 29.0)
        assert points[3] == (12.5, 23.7)

    def test_retourne_des_floats(self, fichier_points_valide):
        """Vérifie que les coordonnées sont bien en float."""
        points = lire_coordonnees(fichier_points_valide)
        for x, y in points:
            assert isinstance(x, float)
            assert isinstance(y, float)

    def test_fichier_inexistant(self):
        """Vérifie que FileNotFoundError est levée si le fichier n'existe pas."""
        with pytest.raises(FileNotFoundError):
            lire_coordonnees("fichier_qui_nexiste_pas.txt")

    def test_fichier_vide(self, fichier_vide):
        """Vérifie que ValueError est levée pour un fichier vide."""
        with pytest.raises(ValueError, match="au moins 2 points"):
            lire_coordonnees(fichier_vide)

    def test_un_seul_point(self, fichier_un_seul_point):
        """Vérifie que ValueError est levée si moins de 2 points."""
        with pytest.raises(ValueError, match="au moins 2 points"):
            lire_coordonnees(fichier_un_seul_point)

    def test_format_incorrect(self, fichier_mal_formate):
        """Vérifie que ValueError est levée pour un format invalide."""
        with pytest.raises(ValueError):
            lire_coordonnees(fichier_mal_formate)

    def test_lignes_vides_ignorees(self, tmp_path):
        """Vérifie que les lignes vides sont ignorées."""
        contenu = "1,2\n\n3,4\n\n"
        fichier = tmp_path / "avec_vides.txt"
        fichier.write_text(contenu, encoding="utf-8")
        points = lire_coordonnees(str(fichier))
        assert len(points) == 2


# ─────────────────────────────────────────────
# Tests : calculer_distance
# ─────────────────────────────────────────────

class TestCalculerDistance:
    """Tests pour la fonction calculer_distance."""

    def test_distance_connue(self):
        """Vérifie la distance entre (0,0) et (3,4) = 5."""
        assert calculer_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_distance_point_identique(self):
        """La distance d'un point à lui-même doit être 0."""
        assert calculer_distance((5, 7), (5, 7)) == pytest.approx(0.0)

    def test_distance_positive(self):
        """La distance doit toujours être positive."""
        assert calculer_distance((1, 2), (4, 6)) > 0

    def test_distance_symetrique(self):
        """La distance de A à B est égale à la distance de B à A."""
        a = (2.5, 3.7)
        b = (8.1, 1.2)
        assert calculer_distance(a, b) == pytest.approx(calculer_distance(b, a))

    def test_distance_horizontale(self):
        """Vérifie la distance sur un axe horizontal."""
        assert calculer_distance((0, 0), (10, 0)) == pytest.approx(10.0)

    def test_distance_verticale(self):
        """Vérifie la distance sur un axe vertical."""
        assert calculer_distance((0, 0), (0, 7)) == pytest.approx(7.0)


# ─────────────────────────────────────────────
# Tests : trouver_site_le_plus_proche
# ─────────────────────────────────────────────

class TestTrouverSiteLePlusProche:
    """Tests pour la fonction trouver_site_le_plus_proche."""

    def test_pixel_proche_du_premier_point(self):
        """Un pixel près du premier point doit retourner l'index 0."""
        points = [(0.0, 0.0), (100.0, 100.0)]
        assert trouver_site_le_plus_proche(1, 1, points) == 0

    def test_pixel_proche_du_deuxieme_point(self):
        """Un pixel près du deuxième point doit retourner l'index 1."""
        points = [(0.0, 0.0), (100.0, 100.0)]
        assert trouver_site_le_plus_proche(99, 99, points) == 1

    def test_avec_quatre_coins(self, points_simples):
        """Vérifie les 4 coins avec des sites aux 4 coins."""
        assert trouver_site_le_plus_proche(1, 1, points_simples) == 0
        assert trouver_site_le_plus_proche(9, 1, points_simples) == 1
        assert trouver_site_le_plus_proche(1, 9, points_simples) == 2
        assert trouver_site_le_plus_proche(9, 9, points_simples) == 3

    def test_retourne_un_index_valide(self, points_simples):
        """L'index retourné doit être dans la plage valide."""
        index = trouver_site_le_plus_proche(5, 5, points_simples)
        assert 0 <= index < len(points_simples)


# ─────────────────────────────────────────────
# Tests : normaliser_points
# ─────────────────────────────────────────────

class TestNormaliserPoints:
    """Tests pour la fonction normaliser_points."""

    def test_nombre_points_conserve(self, points_simples):
        """Le nombre de points ne doit pas changer après normalisation."""
        normalises = normaliser_points(points_simples)
        assert len(normalises) == len(points_simples)

    def test_points_dans_les_bornes(self, points_simples):
        """Tous les points normalisés doivent être dans [0, largeur/hauteur]."""
        largeur, hauteur = 500, 500
        normalises = normaliser_points(points_simples, largeur, hauteur)
        for nx, ny in normalises:
            assert 0 <= nx <= largeur
            assert 0 <= ny <= hauteur

    def test_marge_respectee(self, points_simples):
        """Les points normalisés doivent respecter la marge."""
        marge = 30
        normalises = normaliser_points(points_simples, 500, 500, marge)
        for nx, ny in normalises:
            assert nx >= marge
            assert ny >= marge
            assert nx <= 500 - marge
            assert ny <= 500 - marge

    def test_ordre_relatif_conserve(self):
        """L'ordre relatif des points doit être conservé après normalisation."""
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]
        normalises = normaliser_points(points)
        # x et y doivent rester croissants
        for i in range(len(normalises) - 1):
            assert normalises[i][0] < normalises[i + 1][0]
            assert normalises[i][1] < normalises[i + 1][1]


# ─────────────────────────────────────────────
# Tests : generer_grille_voronoi
# ─────────────────────────────────────────────

class TestGenererGrilleVoronoi:
    """Tests pour la fonction generer_grille_voronoi."""

    def test_dimensions_correctes(self):
        """La grille doit avoir les bonnes dimensions."""
        points = [(50.0, 50.0), (200.0, 200.0)]
        grille = generer_grille_voronoi(points, largeur=100, hauteur=80)
        assert grille.shape == (80, 100)

    def test_valeurs_dans_plage(self):
        """Chaque cellule doit contenir un index valide."""
        points = [(10.0, 10.0), (90.0, 90.0)]
        grille = generer_grille_voronoi(points, largeur=50, hauteur=50)
        assert grille.min() >= 0
        assert grille.max() < len(points)

    def test_deux_zones_distinctes(self):
        """Avec 2 points éloignés, la grille doit contenir les 2 index."""
        points = [(5.0, 25.0), (45.0, 25.0)]
        grille = generer_grille_voronoi(points, largeur=50, hauteur=50)
        assert 0 in grille
        assert 1 in grille

    def test_type_numpy(self):
        """La grille retournée doit être un tableau numpy."""
        points = [(10.0, 10.0), (40.0, 40.0)]
        grille = generer_grille_voronoi(points, largeur=50, hauteur=50)
        assert isinstance(grille, np.ndarray)
