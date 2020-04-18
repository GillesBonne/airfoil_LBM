import time
import pickle
import os


class SimulationResult:
    def __init__(self, fp, rho, ux, uy, m, fx, fy):
        self.fp = fp
        self.rho = rho
        self.ux = ux
        self.uy = uy
        self.m = m
        self.fx = fx
        self.fy = fy

    def save(self, folder=None, filename=None):
        if folder is None:
            folder = '../output'
        if filename is None:
            filename = f"simulation_result_{time.time():.0f}"
        filename += ".pkl"

        if not os.path.isdir(folder):
            os.mkdir(folder)

        pickle.dump(self, open(os.path.join(folder, filename), "wb"))

    @staticmethod
    def load(filename, folder=None):
        if folder is None:
            folder = '../output'
        fullfile = os.path.join(folder, filename + ".pkl")
        return pickle.load(open(fullfile, "rb"))
