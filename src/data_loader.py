"""
1_data_loader.py
--------------------------------------------
POO: clase abstracta DataLoader y dos
implementaciones:

* SyntheticDataLoader -> lee CSV generado en /data
* RealDataLoader      -> *placeholder* para futura BD

Todas usan **Polars** por rendimiento.
--------------------------------------------
"""

from abc import ABC, abstractmethod
from pathlib import Path
import polars as pl
from utils import Logger

class DataLoaderError(Exception):
    """Errores relacionados con los data-loader."""

class DataLoader(ABC):
    """Interfaz base: define `load()` que debe devolver pl.DataFrame."""
    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)

    @abstractmethod
    def load(self) -> pl.DataFrame: ...

class SyntheticDataLoader(DataLoader):
    """Carga el CSV de sintéticos generado por nuestro pipeline."""
    def load(self) -> pl.DataFrame:
        if not self.source.exists():
            raise DataLoaderError(f"Archivo {self.source} no encontrado.")
        Logger.info(f"Cargando dataset sintético desde {self.source}")
        return pl.read_csv(self.source)

class RealDataLoader(DataLoader):
    """
    Ejemplo esqueleto para datos reales.
    Aquí solo mostramos la firma; se podría
    implementar conexión a SQL o API.
    """
    def load(self) -> pl.DataFrame:
        raise NotImplementedError(
            "RealDataLoader aún no implementado — "
            "integra tu conexión a la BD aquí."
        )
