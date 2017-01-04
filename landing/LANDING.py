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
import robot_config as conf


#_ Configuration
viewerName = 'Landing'
robotName = "Traceur"
p = conf.references_path

#__ Create the robot  
robot = Wrapper(conf.generic_model, conf.mesh_path)
dt = conf.dt
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
cm1 = np.asmatrix(np.load(p+'/prepare_com1.npy')).T
initial_cm = Traj.ConstantNdTrajectory('CM1',cm1 )
cm2 = np.asmatrix(np.load(p+'/prepare_com2.npy')).T
final_cm = Traj.ConstantNdTrajectory('CM2',cm2 )

#feet static
idxRF = solver.robot.model.getFrameId('mtp_r')
rf_des = robot.position(p1,idxRF)
rf_traj = Traj.ConstantSE3Trajectory('RF1',rf_des)
idxLF = solver.robot.model.getFrameId('mtp_l')
lf_des = robot.position(p2,idxLF)
lf_traj = Traj.ConstantSE3Trajectory('LF1',lf_des)


#__ Create Tasks
RF = Task.SE3Task(solver.robot, idxRF, rf_traj,'Keep Right Foot Task')
#RF.kp = 10
#RF.kv = 1
LF = Task.SE3Task(solver.robot, idxLF, lf_traj, 'Keep Left Foot Task')
#LF.kp = 10
#LF.kv = 1

CM1 = Task.CoMTask(solver.robot, initial_cm,'Center of Mass Task 1')
#CM1.kp = 1
#CM1.kv = 0.1
CM2 = Task.CoMTask(solver.robot, final_cm,'Center of Mass Task 2')
#CM2.kp = 1
#CM2.kv = 0.1

solver.addTask(CM2, 1)
solver.addTask([RF, LF], 1)

robot.display(p1)
time.sleep(3)

t=0
for i in range(1000):
    a = solver.inverseKinematics2nd(0)
    simulator.increment2(robot.q, a, dt, t)
    t += dt


#def prepare():
#    pass
