import os
import time
import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# import from hqp module                                                                                         
import hqp.tasks as Task
import hqp.trajectories as Traj
from hqp.wrapper import Wrapper
from hqp.viewer_utils import Viewer
from hqp.simulator import Simulator
from hqp.solvers import NProjections
import robot_config as rconf
import mocap_config as mconf
import paths as lp
from trajectory_extractor import References
import pinocchio as se3
from pinocchio.utils import zero as mat_zeros

idxTraceur = mconf.traceurs_list.index('Lucas')
trial = References(mconf.traceurs_list[idxTraceur])
trial.loadModel()
trial.display()
trial.getTrials()


robotNode = 'world/Lucas/'
filename = lp.objects+'/parkour_structure_cage_simplefied.obj'
position = se3.SE3.Identity()
position.translation += np.matrix([-.5,-0.98,1.]).T

trial.viewer.viewer.gui.addMesh(robotNode+'cage', filename)
trial.viewer.placeObject(robotNode+'cage', position, True)
trial.viewer.setVisibility("Lucas/floor", "OFF")

r=1
#190
#350
#575
#680
def playMotions(first=0, last=1):
    for i in range(first, last):
        trial.play(trial.trial[r]['pinocchio_data'][i])
        print i
        
playMotions(190,191)
#playMotions(350,351)
#playMotions(575,576)
#playMotions(680,681)

robot = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, 'Robot', True)
trial.viewer.addRobot(robot)
i = 350
q = trial.trial[r]['pinocchio_data'][i]
trial.viewer.display(q,'Robot')

robot2 = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, 'Robot2', True)
trial.viewer.addRobot(robot2)
i = 575
q = trial.trial[r]['pinocchio_data'][i]
trial.viewer.display(q,'Robot2')

robot3 = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, 'Robot3', True)
trial.viewer.addRobot(robot3)
i = 680
q = trial.trial[r]['pinocchio_data'][i]
trial.viewer.display(q,'Robot3')

trial.viewer.setVisibility("Lucas/floor", "OFF")
trial.viewer.viewer.gui.setColor("world/Lucas",(1,.5,.5,1))
trial.viewer.setVisibility("Robot/floor", "OFF")
trial.viewer.viewer.gui.setColor("world/Robot",(1,1,1,1))
trial.viewer.setVisibility("Robot2/floor", "OFF")
trial.viewer.viewer.gui.setColor("world/Robot2",(1,1,1,1))
trial.viewer.setVisibility("Robot3/floor", "OFF")
trial.viewer.viewer.gui.setColor("world/Robot3",(1,.5,.5,1))
trial.viewer.setVisibility("Robot3/globalCoM", "OFF")
trial.viewer.setVisibility("Robot2/globalCoM", "OFF")
trial.viewer.setVisibility("Robot/globalCoM", "OFF")
trial.viewer.setVisibility("Lucas/globalCoM", "OFF")
