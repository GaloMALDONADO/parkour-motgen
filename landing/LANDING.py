import os
import time
import sys
import numpy as np

# import from hqp module
import hqp.tasks as Task
import hqp.trajectories as Traj
from hqp.wrapper import Wrapper
from hqp.viewer_utils import Viewer
from hqp.simulator import Simulator
from hqp.solvers import NProjections 

from pinocchio.utils import zero as mat_zeros
import robot_config as rconf
import mocap_config as mconf
import paths as lp

#_ Configuration
viewerName = 'Landing'
robotName = "Traceur"
p = lp.trajectories_path

#__ Create the robot  
robot = Wrapper(lp.generic_model, lp.mesh_path)
dt = mconf.time_step
#q0 = conf.half_sitting
q0 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
v0 = mat_zeros(robot.nv)

#__ Create simulator: robot + viewer  
simulator = Simulator('Sim1', q0.copy(), v0.copy(), 0.1, robotName, robot)
nq = simulator.robot.nq
nv = simulator.robot.nv

#__ Create Solver  
solver =NProjections('Solv1', q0.copy(), v0.copy(), 0.1, robotName, robot)

#__ Create Trajectories

#Preparation phase
#initial and final posture
p1 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
initial_pose = Traj.ConstantNdTrajectory('Post1', p1) 
p2 = np.asmatrix(np.load(p+'/prepare_ref2.npy')).T
final_pose = Traj.ConstantNdTrajectory('Post2', p2)


#trajectory of CoM
cm2 = np.asmatrix(np.load(p+'/prepare_com2.npy')).T
final_cm = Traj.ConstantNdTrajectory('CM2',cm2 )

#feet static
idxRF = solver.robot.model.getFrameId('mtp_r')
rf_des = robot.framePosition(idxRF,p2)
robot.viewer.gui.addXYZaxis('world/target1', [1., 1., 0., .5], 0.03, 0.3)
robot.placeObject('world/target1', rf_des, True)
rf_traj = Traj.ConstantSE3Trajectory('RF1',rf_des)

idxLF = solver.robot.model.getFrameId('mtp_l')
lf_des = robot.framePosition(idxLF,p2)
robot.viewer.gui.addXYZaxis('world/target2', [1., 1., 0., .5], 0.03, 0.3)
robot.placeObject('world/target2', lf_des, True)
lf_traj = Traj.ConstantSE3Trajectory('LF1',lf_des)


#__ Create Tasks
RF = Task.SE3Task(solver.robot, idxRF, rf_traj,'Keep Right Foot Task')
LF = Task.SE3Task(solver.robot, idxLF, lf_traj, 'Keep Left Foot Task')
CM2 = Task.CoMTask(solver.robot, final_cm,'Center of Mass Task 2')

solver.addTask([RF, LF], 1)
solver.addTask(CM2, 1)

robot.display(p1)
time.sleep(1)

t=0
for i in range(100):
    a = solver.inverseKinematics2nd(0)
    simulator.increment2(robot.q, a, dt, t)
    t += dt


#def prepare():
#    pass
