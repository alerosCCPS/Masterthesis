import json
import os
import numpy as np

Script_Root = os.path.abspath(os.path.dirname(__file__))

# with open(os.path.join(Script_Root, "ros2_ws", "src","mpc","mpc","setup.json"), "r")as f:
#     data = json.load(f)
#     print(data)
#     print(data["direction"])
#               a            b            c             d           e            f          g
#     ['None', 'None',     'None',       'None',      'None',     'None',     'None',     'None']
# a   ['None', 'up_left',  'left',       'left',      'left',     'left',     'left',     'left']
# b   ['None', 'up',       'up_left',    'left',      'left',     'left',     'left',     'left']
# 1   ['None', 'up',       'up',         'up',        'up',       'up',       'up',       'up']
# c   ['None', 'up',       'up',         'up_left',   'left',     'left',     'left',     'left']
# d   ['None', 'up',       'up',         'up',        'up_left',  'left',     'left',     'left']
# e   ['None', 'up',       'up',         'up',        'up',       'up_left',  'left',     'left'],
# f   ['None', 'up',       'up',         'up',        'up',       'up',       'up_left',  'left'],
# g   ['None', 'up',       'up',         'up',        'up',       'up',       'up',       'up_left']
# a   ['None', 'up_left',  'up',         'up',        'up',       'up',       'up',       'up'],
# b   ['None', 'up', 'up_left', 'up', 'up', 'up', 'up', 'up'],
# c   ['None', 'up', 'up', 'up_left', 'up', 'up', 'up', 'up'],
# 1   ['None', 'up', 'up', 'up', 'up', 'up', 'up', 'up'],
# d   ['None', 'up', 'up', 'up', 'up_left', 'up', 'up', 'up'],
# e   ['None', 'up', 'up', 'up', 'up', 'up_left', 'up', 'up'],
# f   ['None', 'up', 'up', 'up', 'up', 'up', 'up_left', 'up'],
# g   ['None', 'up', 'up', 'up', 'up', 'up', 'up', '