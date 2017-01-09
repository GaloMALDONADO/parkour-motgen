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
import robot_config as rconf
import mocap_config as mconf
import paths as lp
import pinocchio as se3
from pinocchio.utils import zero as mat_zeros


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


#__ Create the motions
class PrepareToJump:
    DURATION = 150
    IDXRF = solver.robot.model.getFrameId('mtp_r')
    IDXLF = solver.robot.model.getFrameId('mtp_l')
    def __init__(self, visualize=True):
        self.desPosture1 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
        self.desPosture2 = np.asmatrix(np.load(p+'/prepare_ref2.npy')).T
        self.desCoM = np.asmatrix(np.load(p+'/prepare_comtrajectory.npy')).T
        self.desRF = robot.framePosition(self.IDXRF,self.desPosture2)
        self.desLF = robot.framePosition(self.IDXLF,self.desPosture2)
        self.trajectories = self.createTrajectories()
        self.tasks = self.createTasks()
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def createTrajectories(self):
        #initial and final posture
        self.post1Traj = Traj.ConstantNdTrajectory('Post1', self.desPosture1) 
        self.post2Traj = Traj.ConstantNdTrajectory('Post2', self.desPosture2)
        #follor the trajectory of the CoM
        self.cmTraj = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 15)
        #to keep the feet static
        self.rfTraj = Traj.ConstantSE3Trajectory('RF1',self.desRF)
        self.lfTraj = Traj.ConstantSE3Trajectory('LF1',self.desLF)
        return [self.cmTraj, self.rfTraj, self.lfTraj, self.post2Traj]
    
    def createTasks(self):
        #__ Create Tasks
        idxRF = solver.robot.model.getFrameId('mtp_r')
        idxLF = solver.robot.model.getFrameId('mtp_l')
        self.RF = Task.SE3Task(solver.robot, self.IDXRF, self.rfTraj,'Keep Right Foot Task')
        self.LF = Task.SE3Task(solver.robot, self.IDXLF, self.lfTraj, 'Keep Left Foot Task')
        self.CM = Task.CoMTask(solver.robot, self.cmTraj,'Center of Mass Task')
        self.PS = Task.JointPostureTask(solver.robot, self.post2Traj, 'Posture 1')
        return [self.CM, self.RF, self.LF, self.PS]

    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        robot.viewer.gui.addXYZaxis('world/target1', [1., 1., 0., .5], 0.03, 0.3)
        robot.placeObject('world/target1', cm, True)
        robot.viewer.gui.addXYZaxis('world/target2', [1., 1., 0., .5], 0.03, 0.3)
        robot.placeObject('world/target2', self.desRF, True)
        robot.viewer.gui.addXYZaxis('world/target3', [1., 1., 0., .5], 0.03, 0.3)
        robot.placeObject('world/target3', self.desLF, True)
    
    def pushTasks(self):
        solver.addTask(self.PS, 1)
        solver.addTask(self.CM, 1)
        solver.addTask([self.RF, self.LF], 1)

    def startSimulation(self):
        robot.display(self.desPosture1)
        t = 0.0
        for i in range(0,self.DURATION):
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(robot.q, a, dt, t)
            t += dt

class Jump():
    DURATION = 235
    def __init__(self, prevTasks, visualize=True):
        #self.actAngMom = se3.ccrba(robot.model, robot.data, robot.q, qdot)[3:6]
        self.desCoM = np.asmatrix(np.load(p+'/push_comprofile.npy')).T
        self.desPosture = np.asmatrix(np.load(p+'/push_reff.npy')).T
        self.desAngMom = np.asmatrix(np.load(p+'/push_hoprofile.npy')).T
        self.createTrajectories()
        self.createTasks()
        self.prevTasks=prevTasks
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def createTrajectories(self):
        #initial and final posture
        self.postTraj = Traj.ConstantNdTrajectory('Post1', self.desPosture) 
        #follor the trajectory of the CoM
        self.cmTraj = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 15)
        #follow angular momentum
        #self.angMomTraj = Traj.ConstantNdTrajectory('AngMom', self.desAngMom)
    
    def createTasks(self):
        self.PS = Task.JointPostureTask(solver.robot, self.postTraj, 'Final Posture Task')
        self.CM = Task.CoMTask(solver.robot, self.cmTraj,'Center of Mass Task')
        #self.AM = Task.AngularMomentumTask(solver.robot,'Ang Mom Task')

    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        robot.viewer.gui.addXYZaxis('world/target1', [1., 1., 0., .5], 0.03, 0.3)
        robot.placeObject('world/target1', cm, True)
    
    def pushTasks(self):
        #solver.addTask(self.AM,1)
        solver.addTask(self.PS, 1)
        solver.addTask(self.CM, 1)
        solver.addTask(self.prevTasks,1)
        #solver.addTask(self.PS, 1)

    def startSimulation(self):
        t = 0.0
        for i in range(0,self.DURATION):
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(robot.q, a, dt, t)
            t += dt

class Fly():
    DURATION = 235
    def __init__(self, visualize=True):
        self.desPosture = np.asmatrix(np.load(p+'/fly_ref.npy')).T
        self.desPelvisRot = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[:,:7].T
        self.createTrajectories()
        self.createTasks()
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def createTrajectories(self):
        #initial and final posture
        self.postTraj = Traj.ConstantNdTrajectory('Post at IC', self.desPosture)
        self.rotPelvTraj = Traj.SmoothedNdTrajectory('Pelvis Rot traj', self.desPelvisRot, dt, 15)
    
    def createTasks(self):
        self.PS = Task.JointPostureTask(solver.robot, self.postTraj, 'Final Posture Task')
        self.PR = Task.FreeFlyerTask(solver.robot, self.rotPelvTraj, 'Rotation Pelvis Task')
        self.PR.mask(np.array([0,0,0,1,1,1]))

    def pushTasks(self):
        solver.addTask(self.PS, 1)
        solver.addTask(self.PR, 1)

    def startSimulation(self):
        t = 0.0
        #g = robot.biais(robot.q,0*robot.v)
        #b = robot.biais(robot.q,robot.v)
        #gm = -np.linalg.inv(robot.data.M)*(g)#+b
        for i in range(0,self.DURATION):
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(robot.q, a, dt, t)
            #simulator.increment2(robot.q, gm, dt, t)
            t += dt

def startSimulation():
    prepare = PrepareToJump()
    prepare.startSimulation()
    #transition
    solver.emptyStack()
    # ---
    jump = Jump([prepare.RF, prepare.LF])
    jump.startSimulation()
    solver.emptyStack()
    fly = Fly(visualize=False)
    fly.startSimulation()

startSimulation()
#TODO posture trajectories must be generated differently, not with the actual methods.
# 
