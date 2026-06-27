from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'robot_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='warxen',
    maintainer_email='yaschenia@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        "stereo_camera_node = robot_camera.stereo_camera_node:main",
        "isaac_vslam_relay_node = robot_camera.isaac_vslam_relay_node:main",        
        ],
    },
)
