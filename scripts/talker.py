#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from rl_control.msg import State
from rl_control.msg import Control

def stateCallback(data):
  print("Hey")

class SensorsBridge:
  state_sub = rospy.Subscriber("state", State, stateCallback)
  control_pub = rospy.Publisher("control", Control, queue_size=10)
  
  def __init__(self):
    rospy.spin()
    
    

if __name__ == '__main__':
  rospy.init_node('sensors_bridge', anonymous=True)
  SensorsBridge()