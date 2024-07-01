from setuptools import find_packages, setup

package_name = 'model_iden'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pandas'],
    zip_safe=True,
    maintainer='yangyin',
    maintainer_email='yangyin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "comm_sub_handle=model_iden.myNode:start_comm_sub",
            "comm_pub_handle=model_iden.myNode:start_comm_pub",
            "data_sender_handle=model_iden.myNode:start_data_sender",
            "mess_sub_handle=model_iden.myNode:start_mess_sub"
        ],
    },
)
