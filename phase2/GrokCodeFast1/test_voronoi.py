import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch
from voronoi_app import load_points, generate_voronoi, plot_voronoi, export_voronoi


class TestLoadPoints:
    def test_valid_file(self):
        """Test chargement d'un fichier valide."""
        content = "2,4\n5.3,4.5\n18,29\n12.5,23.7"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            filename = f.name
        try:
            points = load_points(filename)
            expected = np.array([[2, 4], [5.3, 4.5], [18, 29], [12.5, 23.7]])
            np.testing.assert_array_equal(points, expected)
        finally:
            os.unlink(filename)

    def test_file_not_found(self):
        """Test erreur si fichier introuvable."""
        with pytest.raises(FileNotFoundError, match="introuvable"):
            load_points("nonexistent.txt")

    def test_invalid_format(self):
        """Test erreur si format incorrect."""
        content = "2,4\ninvalid,line"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            filename = f.name
        try:
            with pytest.raises(ValueError, match="non numériques"):
                load_points(filename)
        finally:
            os.unlink(filename)

    def test_insufficient_points(self):
        """Test erreur si moins de 2 points."""
        content = "2,4"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            filename = f.name
        try:
            with pytest.raises(ValueError, match="Au moins 2 points"):
                load_points(filename)
        finally:
            os.unlink(filename)

    def test_empty_lines_ignored(self):
        """Test que les lignes vides sont ignorées."""
        content = "2,4\n\n5.3,4.5"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            filename = f.name
        try:
            points = load_points(filename)
            expected = np.array([[2, 4], [5.3, 4.5]])
            np.testing.assert_array_equal(points, expected)
        finally:
            os.unlink(filename)


class TestGenerateVoronoi:
    def test_generation(self):
        """Test génération du Voronoï."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        vor = generate_voronoi(points)
        assert vor is not None
        assert len(vor.vertices) > 0  # Vérifie que des sommets sont générés


class TestPlotVoronoi:
    @patch('matplotlib.pyplot.gca')
    def test_plot(self, mock_gca):
        """Test tracé du Voronoï (mocké pour éviter affichage)."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        vor = generate_voronoi(points)
        plot_voronoi(vor, points)
        mock_gca.assert_called()


class TestExportVoronoi:
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_export(self, mock_close, mock_savefig):
        """Test export (mocké pour éviter écriture fichier)."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        vor = generate_voronoi(points)
        export_voronoi(vor, points, "test", "svg")
        mock_savefig.assert_called_with("test.svg", format="svg", bbox_inches='tight')
        mock_close.assert_called()