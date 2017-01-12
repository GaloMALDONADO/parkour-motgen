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



Viewer.ENABLE_VIEWER = rconf.ENABLE_VIEWER
dt = mconf.time_step
#dt = 0.005
t  = 0.0

#_ Configuration
viewerName = 'Landing'
participantName = 'Traceur'
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

#__ Create simulator: robot + viewer  
simulator = Simulator('Sim1', robot)
simulator.viewer.addRobot(traceur)
nq = simulator.robot.nq
nv = simulator.robot.nv
#simulator.viewer.setVisibility("floor", "ON" if rconf.SHOW_VIEWER_FLOOR else "OFF");

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
        #self.angMomTraj = Traj.ConstantNdTrajectory('AngMom', self.desAngMom)
    
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
        #solver.addTask(self.PS, 1)

    def startSimulation(self):
        t = 0.0        
        for i in range(0,self.DURATION):
            #print 'Time: ', t
            a = solver.inverseKinematics2nd(t)
            simulator.increment2(simulator.robot.q, a, dt, t)
            t += dt

class Fly():
    DURATION = 278
    def __init__(self, visualize=True):
        self.desPosture = np.asmatrix(np.load(p+'/fly_ref.npy')).T
        self.desPelvis = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[:,:7].T
        self.CoM = self.calculateCoMParabole(
            simulator.robot.data.com[0],
            simulator.robot.data.vcom[0],
            simulator.robot.data.acom[0])
        print simulator.robot.data.acom[0]
        self.desCoM=np.matrix(self.CoM).T
        self.createTrajectories()
        self.createTasks()
        self.pushTasks()
        if visualize is True:
            self.visualizeTasks()

    def calculateCoMParabole(self,pcom,vcom,acom):
        CoM = []
        for i in range(0,self.DURATION):    
            #acom += acom+np.matrix([0.,0.,-9.809]).T#/simulator.robot.data.mass[0]]).T
            acom += acom+np.matrix([0.,0.,-9.809]).T#/simulator.robot.data.mass[0]]).T#
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
        #for i in xrange (0,self.DURATION-1,1):
        #    simulator.viewer.addLine(robotNode+'comFly'+str(i),self.CoM[i], self.CoM[i+1], color=(1.,1.,0,1.0),)

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
            t += dt
            #simulator.increment2(robot.q, gm, dt, t)
            

#def startSimulation():
print 'Preparation Phase'
prepare = PrepareToJump()
prepare.startSimulation()
#transition
# ---
print 'Jump Phase'
solver.emptyStack()
jump = Jump([prepare.RF, prepare.LF])
jump.startSimulation()
print 'Fly Phase'
solver.emptyStack()
fly = Fly(visualize=True)
fly.startSimulation()
simulator.viewer.updateRobotConfig(robot.q0.copy(),'Traceur')
#TODO
#simulator.viewer.robots['Traceur'].display(q0)

#startSimulation()
# 

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
plt.ion()
fig = plt.figure()

ax = fig.add_subplot ('111')
ax.plot(fly.COM,'r', linewidth=3.0)
#ax.plot(com[:,2],'g', linewidth=3.0)
#ax.plot(com[:,3],'b', linewidth=3.0)
'''

