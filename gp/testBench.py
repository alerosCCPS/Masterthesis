from data_sampler import Sampler
from gp_3D import Trainer as Tra3D
from sim_3D import Simulator as Sim3D
from gp_2D import Trainer as Tra2D
from sim_2D import Simulator as Sim2D
from visualization import SimPlotter, ResultePlotter

def test_gp_3D():
    reversed = False
    case_name = 'test_traj_3D'
    sampler = Sampler(case_name)
    sampler.sampling()

    trainer = Tra3D(case_name)
    trainer.process()

    simulator = Sim3D(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

def test_gp_2D():
    reversed = False
    case_name = 'test_traj_2D'
    sampler = Sampler(case_name)
    sampler.sampling()

    trainer = Tra2D(case_name)
    trainer.process()

    simulator = Sim2D(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

test_gp_3D()
test_gp_2D()