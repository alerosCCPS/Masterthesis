from setuptools import find_packages, setup

package_name = 'gp'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=find_packages(where='src'),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pandas', 'torch', 'numpy', 'gpytorch'],
    zip_safe=True,
    maintainer='yangyin',
    maintainer_email='yangyin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        "gp_3D_handle=gp.gp_3D_node:start_gp_3D",
        "gp_3D_LF_handle=gp.gp_3D_LF_node:start_gp_3D_LF",
        "gp_2D_handle=gp.gp_2D_node:start_gp_2D",
        "gp_2D_LF_handle=gp.gp_2D_LF_node:start_gp_2D_LF"
        ],
    },
)
