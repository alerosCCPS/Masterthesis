from data_sampler import Sampler
from gp.gp_3D import Trainer as Tra3D
from gp.sim_3D import Simulator as Sim3D
from gp.gp_2D import Trainer as Tra2D
from gp.sim_2D import Simulator as Sim2D
from gp.gp_3D_LF import Trainer as Tra3DLF
from gp.sim_3D_LF import Simulator as Sim3DLF
from gp.gp_2D_LF import Trainer as Tra2DLF
from gp.sim_2D_LF import Simulator as Sim2DLF
from gp.visualization import SimPlotter, ResultePlotter

def test_gp_3D():
    reversed = False
    case_name = 'test_traj_3D'
    # sampler = Sampler(case_name,sample_rate=0.1)
    # sampler.sampling()
    #
    # trainer = Tra3D(case_name)
    # trainer.process()

    simulator = Sim3D(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

def test_gp_2D():
    reversed = False
    case_name = 'test_traj_2D'
    # sampler = Sampler(case_name,sample_rate=0.1)
    # sampler.sampling()
    #
    # trainer = Tra2D(case_name)
    # trainer.process()

    simulator = Sim2D(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

def test_gp_3D_LF():
    reversed = False
    case_name = 'test_traj_3D_LF'
    # sampler = Sampler(case_name,sample_rate=0.1)
    # sampler.sampling()
    #
    # trainer = Tra3DLF(case_name)
    # trainer.process()

    simulator = Sim3DLF(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

def test_gp_2D_LF():
    reversed = False
    case_name = 'test_traj_2D_LF'
    # sampler = Sampler(case_name,sample_rate=0.1)
    # sampler.sampling()
    #
    # trainer = Tra2DLF(case_name)
    # trainer.process()

    simulator = Sim2DLF(case_name, reversed)
    simulator.start_sim()

    plo = SimPlotter(case_name, reversed)
    plo.plot_traj()
    replot = ResultePlotter(case_name)
    replot.plot()

test_gp_3D()
test_gp_2D()
test_gp_3D_LF()
test_gp_2D_LF()