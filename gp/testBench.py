from data_sampler import Sampler
from gp import Trainer
from sim import Simulator
from visualization import SimPlotter

sampler = Sampler()
sampler.split_data()

trainer = Trainer()
trainer.process()

simulator = Simulator()
simulator.start_sim()

path_name = 'test_traj'
plo = SimPlotter(path_name)
plo.plot_traj()