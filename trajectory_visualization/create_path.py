import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import splrep, splev, interp1d
from scipy.special import comb
import os
from abc import ABC, abstractmethod
from typing import Optional
import types

Script_Root = os.path.abspath(os.path.dirname(__file__))


class PathCreator:

    def __init__(self):
        self.steps = 0
        self.step_length = 0.001
        self.initPos = (0, 0)
        self.initTheta = 0
        self.terminalTheta = 0
        self.arc = []
        self.curvature = []
        self.phi_curve = []
        self.bezier_points = []

    def create_path(self):
        path = types.SimpleNamespace()
        path.steps = self.steps
        path.arc = self.arc
        path.curvature = self.curvature
        path.phi_curve = self.phi_curve
        path.initTheta = self.initTheta
        path.initPos = self.initPos
        path.terminalTheta = self.terminalTheta
        path.bezier_points = self.bezier_points
        return path

    def create_circle(self, factor=1.25, radius=2, initPos=(0,0), initTheta=0):
        arc_length = factor * np.pi * radius
        self.initPos = initPos
        self.initTheta = initTheta
        self.steps = int(arc_length / self.step_length) + 1
        self.arc = [a * self.step_length for a in range(0, self.steps)]
        self.curvature = [1 / radius] * len(self.arc)
        self.terminalTheta = initTheta + factor * np.pi
        self.phi_curve = [s/radius for s in self.arc]
        return self.create_path()

    def create_bezier(self, points: np.ndarray):
        self.initPos = tuple(points[0,:])
        self.bezier_points = points
        bezier = MyBezier(points=points)
        self.steps = int(bezier.arc_length / self.step_length) + 1
        uniform_arc = np.linspace(0, bezier.arc_length, self.steps)
        self.arc = uniform_arc

        # build up the mapping from t to arc
        t_interpolate = np.linspace(0, 1, self.steps * 20)
        arc_interpolate = np.array(
            [integrate.quad(bezier.cal_norm, 0, t, epsabs=1e-12, epsrel=1e-12, limit=1000)[0] for t in
             t_interpolate])

        tck = splrep(arc_interpolate, t_interpolate, k=3)
        uniform_t_spline = splev(uniform_arc, tck)

        # arc_to_t_func = interp1d(arc_interpolate, t_interpolate, kind='cubic')
        # uniform_t = arc_to_t_func(uniform_arc)

        uniform_curve = np.array([bezier.interpolate(t) for t in uniform_t_spline])

        def cal_curvature(points):
            dx = np.gradient(points[:, 0])
            dy = np.gradient(points[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            return -(ddx * dy - ddy * dx) / (dx ** 2 + dy ** 2) ** 1.5

        self.curvature = cal_curvature(np.array(uniform_curve))
        initPtr = bezier.derivative(0)
        terminalPtr = bezier.derivative(1)
        self.initTheta = np.arctan2(initPtr[1], initPtr[0])
        self.terminalTheta = np.arctan2(terminalPtr[1], terminalPtr[0])
        return self.create_path()


class Bezier_normal:
    """
    directly using Bezier_normal to generate cure causes high inaccuracy
    using the formular_generator to get explicit interpolation and derivative function
    """

    def __init__(self, points: np.ndarray):
        self.control_points = points
        self.n = len(self.control_points) - 1
        self.curve = np.ndarray
        self.arc_length = 0
        self.create_curve()

    def comb(self, n, k):
        def factorial(start, end):
            if start <= end:
                return end
            return start * factorial(start - 1, end)

        return factorial(n, n - k + 1) / factorial(k, 1)

    def interpolate(self, t: float):
        result = np.zeros_like(self.control_points[0], dtype='float')
        for i, p in enumerate(self.control_points):
            result += p * comb(self.n, i) * t ** i * (1 - t) ** (self.n - i)
        return result

    def derivative(self, t: float):
        result = np.zeros_like(self.control_points[0], dtype='float')
        for i, p in enumerate(self.control_points):
            if i > 0:
                result += p * comb(self.n, i) * i * t ** (i - 1) * (1 - t) ** (self.n - 1)
            if i < self.n:
                result += -p * comb(self.n, i) * t ** i * (self.n - i) * (1 - t) ** (self.n - i - 1)
        return result

    def cal_norm(self, t):
        return np.linalg.norm(self.derivative(t))

    def create_curve(self):
        t = np.linspace(0, 1, 100)
        self.curve = np.array([self.interpolate(i) for i in t])
        self.arc_length, error = integrate.quad(self.cal_norm, 0, 1, epsabs=1.49e-12, epsrel=1.49e-12, limit=1000)
        print('created bezier curve with estimated integral error: ', error)

    @staticmethod
    def formular_generator(n):
        interpolate_f = []
        derivate_f = []
        for i in range(n + 1):
            interpolate_f.append(f"p[{i}] * {comb(n, i)} * t ** {i} * (1 - t) ** {n - i}")
            if i > 0:
                derivate_f.append(f"p[{i}] * {comb(n, i)} * {i} * t ** ({i - 1}) * (1 - t) ** {n - i}")
            if i < n:
                derivate_f.append(f"(-p[{i}] * {comb(n, i)} * t ** {i} * {n - i} * (1 - t) ** {n - i - 1})")
        return " + ".join(interpolate_f), " + ".join(derivate_f)


class AbstractBezierProduct(ABC):

    @abstractmethod
    def interpolate(self, p, t):
        return

    @abstractmethod
    def derivative(self, p, t):
        return


class BezierN3(AbstractBezierProduct):
    def interpolate(self, p, t):
        return p[0] * 1.0 * t ** 0 * (1 - t) ** 3 + p[1] * 3.0 * t ** 1 * (1 - t) ** 2 + p[2] * 3.0 * t ** 2 * (
                1 - t) ** 1 + p[3] * 1.0 * t ** 3 * (1 - t) ** 0

    def derivative(self, p, t):
        return (-p[0] * 1.0 * t ** 0 * 3 * (1 - t) ** 2) + p[1] * 3.0 * 1 * t ** (0) * (1 - t) ** 2 + (
                -p[1] * 3.0 * t ** 1 * 2 * (1 - t) ** 1) + p[2] * 3.0 * 2 * t ** (1) * (1 - t) ** 1 + (
                -p[2] * 3.0 * t ** 2 * 1 * (1 - t) ** 0) + p[3] * 1.0 * 3 * t ** (2) * (1 - t) ** 0


class BezierN4(AbstractBezierProduct):
    def interpolate(self, p, t):
        return p[0] * 1.0 * t ** 0 * (1 - t) ** 4 + \
            p[1] * 4.0 * t ** 1 * (1 - t) ** 3 + \
            p[2] * 6.0 * t ** 2 * (1 - t) ** 2 + \
            p[3] * 4.0 * t ** 3 * (1 - t) ** 1 + \
            p[4] * 1.0 * t ** 4 * (1 - t) ** 0

    def derivative(self, p, t):
        return (-p[0] * 1.0 * t ** 0 * 4 * (1 - t) ** 3) + p[1] * 4.0 * 1 * t ** (0) * (1 - t) ** 3 + (
                -p[1] * 4.0 * t ** 1 * 3 * (1 - t) ** 2) + p[2] * 6.0 * 2 * t ** (1) * (1 - t) ** 2 + (
                -p[2] * 6.0 * t ** 2 * 2 * (1 - t) ** 1) + p[3] * 4.0 * 3 * t ** (2) * (1 - t) ** 1 + (
                -p[3] * 4.0 * t ** 3 * 1 * (1 - t) ** 0) + p[4] * 1.0 * 4 * t ** (3) * (1 - t) ** 0

class BezierN5(AbstractBezierProduct):
    def interpolate(self, p, t):
        return p[0] * 1.0 * t ** 0 * (1 - t) ** 5 + \
            p[1] * 5.0 * t ** 1 * (1 - t) ** 4 + \
            p[2] * 10.0 * t ** 2 * (1 - t) ** 3 + \
            p[3] * 10.0 * t ** 3 * (1 - t) ** 2 +\
            p[4] * 5.0 * t ** 4 * (1 - t) ** 1 + \
            p[5] * 1.0 * t ** 5 * (1 - t) ** 0


    def derivative(self, p, t):
        return (-p[0] * 1.0 * t ** 0 * 5 * (1 - t) ** 4) + \
            p[1] * 5.0 * 1 * t ** (0) * (1 - t) ** 4 + \
            (-p[1] * 5.0 * t ** 1 * 4 * (1 - t) ** 3) + \
            p[2] * 10.0 * 2 * t ** (1) * (1 - t) ** 3 + \
            (-p[2] * 10.0 * t ** 2 * 3 * (1 - t) ** 2) + \
            p[3] * 10.0 * 3 * t ** (2) * (1 - t) ** 2 + \
            (-p[3] * 10.0 * t ** 3 * 2 * (1 - t) ** 1) + \
            p[4] * 5.0 * 4 * t ** (3) * (1 - t) ** 1 + \
            (-p[4] * 5.0 * t ** 4 * 1 * (1 - t) ** 0) + \
            p[5] * 1.0 * 5 * t ** (4) * (1 - t) ** 0

class AbstractBezierCreator(ABC):

    @abstractmethod
    def create_bezier(self) -> AbstractBezierProduct:
        return

    def get(self):
        product = self.create_bezier()
        return product


class ConcreteBezierN3Creator(AbstractBezierCreator):

    def create_bezier(self) -> AbstractBezierProduct:
        return BezierN3()


class ConcreteBezierN4Creator(AbstractBezierCreator):

    def create_bezier(self) -> AbstractBezierProduct:
        return BezierN4()


class ConcreteBezierN5Creator(AbstractBezierCreator):

    def create_bezier(self) -> AbstractBezierProduct:
        return BezierN5()


class MyBezier:

    def __init__(self, points: np.ndarray):
        self.control_points = points
        self.core: Optional[AbstractBezierProduct] = None
        self.check_order()
        self.curve = np.ndarray
        self.arc_length = 0
        self.create_curve()

    def check_order(self):
        myDic = {
            3: ConcreteBezierN3Creator,
            4: ConcreteBezierN4Creator,
            5: ConcreteBezierN5Creator
        }
        self.initialize_core(myDic[len(self.control_points) - 1]())

    def initialize_core(self, creator: AbstractBezierCreator):
        self.core = creator.get()

    def interpolate(self, t: float):
        return self.core.interpolate(self.control_points, t)

    def derivative(self, t: float):
        return self.core.derivative(self.control_points, t)

    def cal_norm(self, t):
        return np.linalg.norm(self.derivative(t))

    def create_curve(self):
        t = np.linspace(0, 1, 1000)
        self.curve = np.array([self.interpolate(i) for i in t])
        self.arc_length, error = integrate.quad(self.cal_norm, 0, 1, epsabs=1e-12, epsrel=1e-12, limit=1000)
        print('created bezier curve with estimated integral error: ', error)


if __name__ == "__main__":
    interpolate, der = Bezier_normal.formular_generator(5)
    print(interpolate)
    print(der)