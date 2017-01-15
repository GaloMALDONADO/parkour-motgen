import os
import time
import sys
import numpy as np

import matplotlib.pyplot as plt


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


Viewer.ENABLE_VIEWER = rconf.ENABLE_VIEWER
dt = mconf.time_step
t  = 0.0

#_ Configuration
viewerName = 'Landing'
participantName = 'Lucas'
robotName = "Robot"
robotNode = 'world/'+robotName+'/'

p = lp.trajectories_path


#__ Create the robots  
#robot = Wrapper(lp.generic_model, lp.mesh_path, robotName, True)
robot = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, robotName, True)
robot.q0 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
robot.dt = dt
traceur = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, participantName, True)
traceur.q0 = rconf.half_sitting

idxTraceur = mconf.traceurs_list.index('Lucas')
trial = References(mconf.traceurs_list[idxTraceur])
trial.loadModel()
trial.display()
trial.getTrials()
r=1
for i in xrange(0,len(trial.trial[1]['pinocchio_data'])):
    trial.trial[r]['pinocchio_data'][i,1] = trial.trial[r]['pinocchio_data'][i,1]+1

#__ Create simulator: robot + viewer  
simulator = Simulator('Sim1', robot)
simulator.viewer.addRobot(traceur)
nq = simulator.robot.nq
nv = simulator.robot.nv
#simulator.viewer.setVisibility("floor", "ON" if rconf.SHOW_VIEWER_FLOOR else "OFF");

#Add objects
filename = lp.objects+'/platform1.stl'
position = se3.SE3.Identity()
position.translation += np.matrix([0.25,-0.35,0.]).T
simulator.viewer.viewer.gui.addMesh(robotNode+'platform1', filename)
simulator.viewer.placeObject(robotNode+'platform1', position, True)
position = se3.SE3.Identity()
position.translation += np.matrix([0.25,0.25,0.]).T
simulator.viewer.viewer.gui.addMesh(robotNode+'platform2', filename)
simulator.viewer.placeObject(robotNode+'platform2', position, True)
#__ Create Solver  
solver =NProjections('Solv1', 
                     simulator.robot.q0.copy(), 
                     simulator.robot.v0.copy(), 
                     dt, simulator.robot.name, 
                     simulator.robot)


#__ Create the motions
class PrepareToJump:
    DURATION = 150
    IDXRF = solver.robot.model.getFrameId('mtp_r')
    IDXLF = solver.robot.model.getFrameId('mtp_l')

    def __init__(self, visualize=True):
        self.desPosture1 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
        self.desPosture2 = np.asmatrix(np.load(p+'/prepare_ref2.npy')).T
        self.desCoM = np.asmatrix(np.load(p+'/prepare_comtrajectory.npy')).T
        self.desRF = simulator.robot.framePosition(self.IDXRF,self.desPosture2)
        self.desLF = simulator.robot.framePosition(self.IDXLF,self.desPosture2)
        self.trajectories = self.createTrajectories()
        self.tasks = self.createTasks()
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def createTrajectories(self):
        #initial and final posture
        self.post1Traj = Traj.VaryingNdTrajectory('Post1', self.desPosture1, dt) 
        self.post2Traj = Traj.VaryingNdTrajectory('Post2', self.desPosture2, dt)
        #follor the trajectory of the CoM
        self.cmTraj = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 55)
        #self.cmTraj = Traj.VaryingNdTrajectory('CMtrj', self.desCoM, dt)
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
        self.RF.mask(np.array([1,1,1,0,0,0]))
        self.LF.mask(np.array([1,1,1,0,0,0]))
        self.CM = Task.CoMTask(solver.robot, self.cmTraj,'Center of Mass Task')
        self.PS = Task.JointPostureTask(solver.robot, self.post2Traj, 'Posture 1')
        return [self.CM, self.RF, self.LF, self.PS]

    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target1', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target1', cm, True)
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target2', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target2', self.desRF, True)
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target3', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target3', self.desLF, True)
    
    def pushTasks(self):
        #solver.addTask(self.PS, 1)
        solver.addTask(self.CM, 1)
        solver.addTask([self.RF, self.LF], 1)

    def startSimulation(self):
        simulator.viewer.display(self.desPosture1, 'Robot')
        t = 0.0
        for i in range(0,self.DURATION):
            #print 'Time: ', t
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(simulator.robot.q, a, dt, t)
            trial.playTrial(rep=r,dt=dt,stp=1,start=i,end=i+1)
            t += dt

class Jump():
    DURATION = 236
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
        self.cmTraj = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 131)
        #follow angular momentum
        self.angMomTraj = Traj.ConstantNdTrajectory('AngMom', self.desAngMom)
    
    def createTasks(self):
        self.PS = Task.JointPostureTask(solver.robot, self.postTraj, 'Final Posture Task')
        self.CM = Task.CoMTask(solver.robot, self.cmTraj,'Center of Mass Task')
        #self.AM = Task.AngularMomentumTask(solver.robot,'Ang Mom Task')

    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target1', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target1', cm, True)
    
    def pushTasks(self):
        #solver.addTask(self.AM,1)
        solver.addTask(self.PS, 1)
        solver.addTask(self.CM, 1)
        solver.addTask(self.prevTasks,1)

    def startSimulation(self):
        t = 0.0        
        for i in range(0,self.DURATION):
            #print 'Time: ', t
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(simulator.robot.q, a, dt, t)
            trial.playTrial(rep=r,dt=dt,stp=1,start=i+150,end=i+151)
            t += dt

class Fly():
    DURATION = 178 #278
    def __init__(self, visualize=True):
        self.desPosture = np.asmatrix(np.load(p+'/fly_ref.npy')).T
        self.desPelvis = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[:,:7].T
        self.CoM = self.calculateCoMParabole(
            simulator.robot.data.com[0],
            simulator.robot.data.vcom[0],
            simulator.robot.data.acom[0])
        self.desCoM=np.matrix(self.CoM).T
        self.createTrajectories()
        self.createTasks()
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def calculateCoMParabole(self,pcom,vcom,acom):
        CoM = []
        acom += np.matrix([0.,0.,-9.809]).T
        for i in range(0,self.DURATION):    
            vcom += acom*dt
            pcom += vcom*dt
            CoM += [np.array(pcom).squeeze()]
        return CoM
    
    def createTrajectories(self):
        #initial and final posture
        self.postTraj = Traj.ConstantNdTrajectory('Post at IC', self.desPosture)
        self.pelvTraj = Traj.SmoothedNdTrajectory('Pelvis Rot traj', self.desPelvis, dt, 15)
        #self.pelvTraj = Traj.ConstantNdTrajectory('Pelvis Rot traj', self.desPelvis)
        self.cmTraj = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 15)
        
        
    def createTasks(self):
        self.PS = Task.JointPostureTask(solver.robot, self.postTraj, 'Final Posture Task')
        self.PR = Task.FreeFlyerTask(solver.robot, self.pelvTraj, 'Rotation Pelvis Task')
        self.PR.mask(np.array([0,0,0,1,1,1]))
        self.CM = Task.CoMTask(solver.robot, self.cmTraj,'Center of Mass Task')

    def visualizeTasks(self):
        pass
        #visualize the parabola
        for i in xrange (0,self.DURATION-1,1):
            simulator.viewer.addLine(robotNode+'comFly'+str(i),
                                     self.CoM[i], 
                                     self.CoM[i+1], 
                                     color=(1.,1.,0,1.0))

    def pushTasks(self):
        solver.addTask(self.PS, 1)
        solver.addTask(self.PR, 1)
        solver.addTask(self.CM, 1)

    def startSimulation(self):
        #a = simulator.robot.data.acom[0]
        #g = robot.biais(robot.q,0*robot.v)
        #b = robot.biais(robot.q,robot.v)
        #g = -np.linalg.inv(robot.data.M)*(g)#+b
        t = 0.0
        dt=0.0025
        for i in range(0,self.DURATION):
            #print 'Time: ', t
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(simulator.robot.q, a, dt, t)
            trial.playTrial(rep=r,dt=dt,stp=1,start=i+386,end=i+387)
            t += dt
            #print i
            

#def startSimulation():
print 'Preparation Phase'
prepare = PrepareToJump()
prepare.startSimulation()
print 'Jump Phase'
solver.emptyStack()
jump = Jump([prepare.RF, prepare.LF])
jump.startSimulation()
print 'Fly Phase'
solver.emptyStack()
fly = Fly(visualize=True)
fly.startSimulation()
#simulator.viewer.updateRobotConfig(robot.q0.copy(),participantName)

#TODO : add angular momentum task


plt.ion()
fig = plt.figure('CoM')

ax = fig.add_subplot ('111')
ax.plot(fly.desCoM[0].A1,'r', linewidth=3.0)
ax.plot(fly.desCoM[1].A1,'g', linewidth=3.0)
ax.plot(fly.desCoM[2].A1,'b', linewidth=3.0)

#plt.close()
