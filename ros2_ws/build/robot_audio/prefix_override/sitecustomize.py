import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/warxen/ai_robot/ros2_ws/install/robot_audio'
