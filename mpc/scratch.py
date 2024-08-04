import json
import os
import numpy as np

Script_Root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(Script_Root, "ros2_ws", "src","mpc","mpc","setup.json"), "r")as f:
    data = json.load(f)
    print(data)
    print(data["direction"])

