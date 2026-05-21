import numpy as np


class PeripheralNerve:
    """
    Methods
    -------
    generate_fiber_population()
    segregate_fibers_into_fascicular_groups()
    plot(ax)
    """
    def __init__(self, topography, fiber_population=None):
        self._topography = topography
        self._fiber_population = fiber_population

    def generate_fiber_population(self):
        return

    def segregate_fibers_into_fascicular_groups(self):
        return

    def plot(self, ax):
        return

    @property
    def topography(self):
        return self._topography

    @property
    def fiber_population(self):
        return self._fiber_population
