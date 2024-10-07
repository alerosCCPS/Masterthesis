from setuptools import find_packages, setup
import os
# Script_Root = os.path.abspath(os.path.dirname(__file__))
package_name = 'mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    # install_requires=['setuptools'] + open(os.path.join(Script_Root,'requirements.txt')).read().splitlines(),
    install_requires=['setuptools', 'pandas', 'casadi', 'numpy'],
    zip_safe=True,
    maintainer='yangyin',
    maintainer_email='yangyin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "controller_handle=mpc.mpc_node:start_controller",
            "controller_noBeta=mpc.mpc_noBeta:start_controller"
        ],
    },
)
