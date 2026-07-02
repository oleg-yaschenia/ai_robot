from setuptools import setup
from glob import glob
import os

package_name = 'robot_vision_assistant'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='warxen',
    maintainer_email='warxen@example.com',
    description='Hybrid local/cloud vision assistant for home robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_assistant_node = robot_vision_assistant.vision_assistant_node:main',
            'local_perception_node = robot_vision_assistant.local_perception_node:main',
            'yolo_perception_node = robot_vision_assistant.yolo_perception_node:main',
            'tts_node = robot_vision_assistant.tts_node:main',
            'asr_node = robot_vision_assistant.asr_node:main',
            'voice_manager_node = robot_vision_assistant.voice_manager_node:main',
            'voice_led_bridge_node = robot_vision_assistant.voice_led_bridge_node:main',
            'perception_entity_adapter_node = robot_vision_assistant.perception_entity_adapter_node:main',
            'scene_interpreter_node = robot_vision_assistant.scene_interpreter_node:main',
        ],
    },
)
