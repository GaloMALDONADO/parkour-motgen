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

p = lp.trajectories_path
Viewer.ENABLE_VIEWER = rconf.ENABLE_VIEWER
dt = mconf.time_step
t  = 0.0
g = 9.809

#_ Configuration
viewerName = 'Landing'
participantName = 'Lucas'
robotName = "Robot"
robotNode = 'world/'+robotName+'/'

#__ Create the robots  
#robot = Wrapper(lp.generic_model, lp.mesh_path, robotName, True)
robot = Wrapper(lp.models_path+'/Lucas.osim', lp.mesh_path, robotName, True)
robot.q0 = np.asmatrix(np.load(p+'/prepare_ref1.npy')).T
robot.dt = dt

idxTraceur = mconf.traceurs_list.index('Lucas')
trial = References(mconf.traceurs_list[idxTraceur])
trial.loadModel()
trial.display()
trial.getTrials()
r=1
v=[]; q=[]; hg=[];
v += [robot.v]
q += [robot.q0]
hg = []
for i in xrange(0,len(trial.trial[1]['pinocchio_data'])):
    trial.trial[r]['pinocchio_data'][i,1] = trial.trial[r]['pinocchio_data'][i,1]+1
    q += [trial.trial[r]['pinocchio_data'][i]]
    if i != 0:
        v+=[se3.differentiate(trial.human.model,trial.trial[r]['pinocchio_data'][i-1], trial.trial[r]['pinocchio_data'][i])/dt]
    #se3.computeAllTerms(trial.human.model, trial.human.data, q[i],  v[i])
    se3.ccrba(trial.human.model, trial.human.data, q[i],  v[i])
    hg += [trial.human.data.hg.np.squeeze().A1]

traceur = trial.human
traceur.q0 = rconf.half_sitting

#__ Create simulator: robot + viewer  
simulator = Simulator('Sim1', robot)
nq = simulator.robot.nq
nv = simulator.robot.nv
#simulator.viewer.setVisibility("floor", "ON" if rconf.SHOW_VIEWER_FLOOR else "OFF")
simulator.viewer.setVisibility("Lucas/floor", "OFF")
simulator.viewer.setVisibility("Robot/floor", "OFF")



#Add objects
filename = lp.objects+'/parkour_structure_cage.stl'
position = se3.SE3.Identity()
position.translation += np.matrix([-.5,-0.98,1.]).T
#simulator.viewer.viewer.gui.addMesh(robotNode+'cage', filename)
#simulator.viewer.placeObject(robotNode+'cage', position, True)


#__ Create Solver  
solver =NProjections('Solv1', 
                     simulator.robot.q0.copy(), 
                     simulator.robot.v0.copy(), 
                     dt, simulator.robot.name, 
                     simulator.robot)

#__ Operational Points
IDXRF = solver.robot.model.getFrameId('mtp_r')
IDXLF = solver.robot.model.getFrameId('mtp_l')

#__ Create the motions
class PrepareToJump:
    DURATION = 150    
    def __init__(self, visualize=True):
        # Posture Task
        self.desPosture = np.asmatrix(np.load(p+'/prepare_ref2.npy')).T
        self.trajPosture = Traj.ConstantNdTrajectory('Posture',self.desPosture)
        self.taskPosture = Task.JointPostureTask(solver.robot, self.trajPosture, 'Posture')
        # Center of mass
        self.desCoM = np.asmatrix(np.load(p+'/prepare_comtrajectory.npy')).T
        self.trajCoM = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 55)
        self.taskCoM = Task.CoMTask(solver.robot, self.trajCoM,'Center of Mass Task')
        # Foot Position
        self.desRF = simulator.robot.framePosition(IDXRF,self.desPosture)
        self.desLF = simulator.robot.framePosition(IDXLF,self.desPosture)
        self.trajRF = Traj.ConstantSE3Trajectory('RF1',self.desRF)
        self.trajLF = Traj.ConstantSE3Trajectory('LF1',self.desLF)
        self.taskRF = Task.SE3Task(solver.robot, IDXRF, self.trajRF,'Keep Right Foot Task')
        self.taskLF = Task.SE3Task(solver.robot, IDXLF, self.trajLF, 'Keep Left Foot Task')
        self.taskRF.mask(np.array([1,1,1,0,0,0]))
        self.taskLF.mask(np.array([1,1,1,0,0,0]))
        if visualize is True:
            self.visualizeTasks()        
        
    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target1', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target1', cm, True)
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target2', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target2', self.desRF, True)
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'target3', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'target3', self.desLF, True)
        

class Jump():
    DURATION = 200
    def __init__(self, visualize=True):
        # Posture
        self.desPosture = np.asmatrix(np.load(p+'/push_reff.npy')).T
        self.trajPosture = Traj.ConstantNdTrajectory('Posture',self.desPosture)
        self.taskPosture = Task.JointPostureTask(solver.robot, self.trajPosture, 'Posture')
        # Linear momentum
        #self.desMom = np.matrix([-g*22, 0., g*23, 0., 2., 0.]).T 
        self.desMom = np.matrix([-g*5, 0., g*5, 0., 2., 0.]).T #np.sqrt(g*5)
        self.trajMom = Traj.ConstantNdTrajectory('Momentum', self.desMom)
        self.taskLinMom = Task.MomentumTask(solver.robot, self.trajMom, 'Linear Momentum AP and V')
        self.taskLinMom.mask(np.array([1,0,1,0,0,0]))
        self.taskLinMom.kp = 20
        self.taskLinMom.kv = 1
        gainVector = np.ones(simulator.robot.nv)
        # shoulder flexion
        gainVector[26]=0; gainVector[34]=0;
        # elbow flexion
        gainVector[29]=0; gainVector[37]=0;
        self.taskLinMom.setGain(gainVector)
        # Angular Momentum
        self.taskAngMom = Task.MomentumTask(solver.robot, self.trajMom, 'Angular Momentum around ML')
        self.taskAngMom.mask(np.array([0,0,0,0,1,0]))
        self.taskAngMom.kp = 1
        self.taskAngMom.kv = 1
        gainVector = np.zeros(simulator.robot.nv)
        # shoulder flexion
        gainVector[26]=1; gainVector[34]=1;
        # elbow flexion
        gainVector[29]=1; gainVector[37]=1;
        self.taskAngMom.setGain(gainVector)
        if visualize is True:
            self.visualizeTasks()
        
    def visualizeTasks(self):
        self.desCoM = np.asmatrix(np.load(p+'/push_comprofile.npy')).T
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'targetJump', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'targetJump', cm, True)
    
class Fly():
    DURATION = 187 #278 178
    def __init__(self, visualize=True):
        # Posture
        self.desPosture = np.asmatrix(np.load(p+'/fly_ref.npy')).T
        self.trajPosture = Traj.ConstantNdTrajectory('Post at IC', self.desPosture)
        self.taskPosture = Task.JointPostureTask(solver.robot, self.trajPosture, 'Final Posture Task')
        # Pelvis orientation
        self.desPelvis = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[:,:7].T
        self.trajPelvis = Traj.SmoothedNdTrajectory('Pelvis Rot traj', self.desPelvis, dt, 15)
        self.taskPelvis = Task.FreeFlyerTask(solver.robot, self.trajPelvis, 'Rotation Pelvis Task')
        self.taskPelvis.mask(np.array([0,0,0,1,1,1]))
        # Center of Mass
        self.CoM = self.calculateCoMParabole(
            simulator.robot.data.com[0],
            simulator.robot.data.vcom[0],
            simulator.robot.data.acom[0])
        self.desCoM=np.matrix(self.CoM).T
        self.trajCoM = Traj.SmoothedNdTrajectory('CMtrj', self.desCoM, dt, 15)
        self.taskCoM = Task.CoMTask(solver.robot, self.trajCoM,'Center of Mass Task')
        if visualize is True:
            self.visualizeTasks()

    def calculateCoMParabole(self,pcom,vcom,acom):
        CoM = []
        acom += np.matrix([0.,0.,-9.809]).T
        for i in range(0,self.DURATION):    
            vcom += acom*dt
            pcom += vcom*dt
            CoM += [np.array(pcom).squeeze()]
        #print vcom
        return CoM

    def visualizeTasks(self):
        #visualize the parabola
        cm = se3.SE3.Identity()
        for i in xrange (0,self.DURATION-1,2):
            simulator.viewer.addLine(robotNode+'comFly'+str(i),
                                     self.CoM[i], 
                                     self.CoM[i+1], 
                                     color=(1.,1.,0,1.0))

            
class Land():
    DURATION = 122
    def __init__(self, visualize=True):
        # Feet
        self.desRF = simulator.robot.framePosition(IDXRF,simulator.robot.q)
        self.trajRF = Traj.ConstantSE3Trajectory('RF1',self.desRF)
        self.taskRF = Task.SE3Task(solver.robot, IDXRF, self.trajRF,'Keep Right Foot Task')
        self.taskRF.mask(np.array([1,1,1,1,1,1]))
        self.desLF = simulator.robot.framePosition(IDXLF,simulator.robot.q)
        self.trajLF = Traj.ConstantSE3Trajectory('LF1',self.desLF)
        self.taskLF = Task.SE3Task(solver.robot, IDXLF, self.trajLF, 'Keep Left Foot Task')
        self.taskLF.mask(np.array([1,1,1,1,1,1]))

        # Linear momentum
        self.desMom = np.matrix([0., 0., 0., 0., 0., 0.]).T 
        self.trajMom = Traj.ConstantNdTrajectory('Momentum', self.desMom)
        self.taskLinMom = Task.MomentumTask(solver.robot, self.trajMom, 'Linear Momentum AP and V')
        self.taskLinMom.mask(np.array([0,0,1,0,0,0]))
        self.taskLinMom.kp = 10
        self.taskLinMom.kv = 1
        gainVector = np.ones(simulator.robot.nv)
        # shoulder flexion
        gainVector[26]=0; gainVector[34]=0;
        # elbow flexion
        gainVector[29]=0; gainVector[37]=0;
        self.taskLinMom.setGain(gainVector)

        # Angular Momentum
        self.taskAngMom = Task.MomentumTask(solver.robot, self.trajMom, 'Angular Momentum around ML')
        self.taskAngMom.mask(np.array([0,0,0,0,1,0]))
        self.taskAngMom.kp = 1
        self.taskAngMom.kv = 1
        gainVector = np.zeros(simulator.robot.nv)
        # shoulder flexion
        gainVector[26]=1; gainVector[34]=1;
        # elbow flexion
        gainVector[29]=1; gainVector[37]=1;
        self.taskAngMom.setGain(gainVector)

        # Center of Mass
        self.desCoM = (np.matrix([0.,0.,0.4]).T + 
                       (self.desRF.translation
                        + self.desLF.translation)/2.)
        self.trajCoM = Traj.ConstantNdTrajectory('CMtrj', self.desCoM)
        self.taskCoM = Task.CoMTask(solver.robot, self.trajCoM,'Center of Mass Task')
        if visualize is True:
            self.visualizeTasks()

    def visualizeTasks(self):
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'targetLand1', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'targetLand1', cm, True)        
            
class Plot():
    def __init__(self):
        self.q = []
        self.hg = []
        self.p_error_hl = []
        self.p_error_ho = []
        plt.ion()

    def Momentum(self):
        #plt.close()
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(hg)
        axarr[0].set_title('Momentum from trial')
        axarr[0].legend(['hlx','hly','hlz','hox','hoy','hoz'])
        axarr[1].plot(self.hg)
        axarr[1].set_title('Momentum from Simulation')
        axarr[1].legend(['hlx','hly','hlz','hox','hoy','hoz'])
        #plt.figure('Momentum Task Error')

    def iMomentum(self,i):
        plt.subplot(231)
        plt.scatter(i, self.hg[i][0],color='red')
        plt.subplot(232)
        plt.scatter(i, self.hg[i][1],color='green')
        plt.subplot(233)
        plt.scatter(i, self.hg[i][2],color='blue')
        plt.subplot(234)
        plt.scatter(i, self.hg[i][3],color='red')
        plt.subplot(235)
        plt.scatter(i, self.hg[i][4],color='green')
        plt.subplot(236)
        plt.scatter(i, self.hg[i][5],color='blue')
        plt.pause(0.0025)
        
    def iErrorMomentum(self,i):
        plt.subplot(211)
        plt.scatter(i, self.p_error_hl[i],color='red')
        plt.subplot(212)
        plt.scatter(i, self.p_error_ho[i],color='green')
        plt.pause(0.0025)


''' **********************  MAIN SCRIPT ********************************** '''
q=[]
plot = Plot()
prepare = PrepareToJump()
print 'Preparation Phase'
solver.addTask(prepare.taskPosture, 1)
solver.addTask(prepare.taskCoM, 1)
solver.addTask([prepare.taskRF, prepare.taskLF], 1)
t = 0.0
for i in range(0,150):
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i,end=i+1)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]

print 'Jump Phase'
solver.emptyStack()
jump = Jump()
solver.addTask(jump.taskAngMom, 1)
solver.addTask(jump.taskLinMom, 1)
solver.addTask([prepare.taskRF, prepare.taskLF], 1)
t = 0.0
for i in range(0,80):#100
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i+prepare.DURATION,end=i+prepare.DURATION+1)
    t += dt
    q+=[simulator.robot.q]
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
    plot.p_error_hl+=[solver.tasks[0].p_error]
    plot.p_error_ho+=[solver.tasks[1].p_error]

solver.emptyStack()    
for i in range(0,170):#150
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]

#Last Push in Antero Posterior direction
print 'Fly Phase'
solver.emptyStack()
fly = Fly()
solver.addTask(fly.taskPelvis, 1)
solver.addTask(fly.taskPosture, 1)
solver.addTask(fly.taskCoM, 1)
t = 0.0
while True:
#for i in range(0, fly.DURATION):
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    # check contact with the ground
    if simulator.robot.framePosition(IDXRF,simulator.robot.q).translation[2] <= 0.2 :
        break
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i+386,end=i+387)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]

print 'Land Phase'
solver.emptyStack()
land = Land()
solver.addTask(land.taskAngMom, 1)
solver.addTask(land.taskLinMom, 1)
solver.addTask([land.taskRF, land.taskLF], 1)
t = 0.0
for i in range(0,land.DURATION):
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i+573,end=i+574)
    t += dt
    q+=[simulator.robot.q]
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
    #print simulator.robot.data.acom[0]
    #print simulator.robot.data.kinetic_energy

def playMotions(first=0, last=1, plotFlag=False):
    if plotFlag is True:
        plt.close()
        #plt.ion()
        #plt.figure('Momentum')
    
    for i in range(first, last):
        trial.playTrial(rep=r,dt=dt,stp=1,start=i,end=i+1)
        simulator.viewer.display(q[i], simulator.robot.name)
        #plot.Momentum(i)
        if plotFlag is True:
            plt.subplot(231)
            plt.scatter(i,hg[i][0],color='red')
            plt.subplot(232)
            plt.scatter(i,hg[i][1],color='green')
            plt.subplot(233)
            plt.scatter(i,hg[i][2],color='blue')
            plt.subplot(234)
            plt.scatter(i,hg[i][3],color='red')
            plt.subplot(235)
            plt.scatter(i,hg[i][4],color='green')
            plt.subplot(236)
            plt.scatter(i,hg[i][5],color='blue')
            plt.pause(dt)

#print simulator.robot.data.acom[0]
#print simulator.robot.data.kinetic_energy
    
#simulator.viewer.updateRobotConfig(robot.q0.copy(),participantName)

#import utils
#from scipy.integrate import odeint

#state0 = [np.array(simulator.robot.data.com[0][2])[0][0] , 
#          np.array(simulator.robot.data.vcom[0][2])[0][0] ]
#t = np.arange(0.0, .3, dt)

#state = odeint(utils.MassSpring, state0)#, t, simulator.robot.data.mass[0], 10.)
#utils.plot(t,state)





#TODO : add gain for angular momentum task


'''
# plot velocity of center of mass 
com = []
for i in xrange(100):
    com += [np.array(se3.centerOfMass(robot.model, 
                                      robot.data, 
                                      trial.land[1]['pinocchio_data'][i].T, True).T).squeeze()]
com = np.matrix(com)

plt.ion()
fig = plt.figure('CoM')
ax = fig.add_subplot ('111')
ax.plot(trial.land[1]['time'].T, com[:,0],'r', linewidth=3.0)
ax.plot(trial.land[1]['time'].T, com[:,1],'g', linewidth=3.0)
ax.plot(trial.land[1]['time'].T, com[:,2],'b', linewidth=3.0)


#acceleration before impact
#acom = 
#Fcontact = simulator.robot.data.mass[0]*simulator.robot.data.acom
#comTrj = - Fcontact/ks #+ Fcontact/kd
b = 100
k = 10
comTrj = []
for i in xrange():
    comTrj += [simulator.robot.data.mass[0]*simulator.robot.data.acom[0] +  
              b*simulator.robot.data.vcom[0] + 
              k*simulator.robot.data.com[0]]
'''
'''
for i in range(0,self.DURATION):    
            vcom += acom*dt
            pcom += vcom*dt
            CoM += [np.array(pcom).squeeze()]
        print vcom
        return CoM

hcomMax = np.asmatrix(np.load(p+'/push_comprofile.npy')).T[2,-1]
Ein = simulator.robot.data.kinetic_energy + simulator.robot.data.mass[0]*hcomMax*9.81

#ax.plot(fly.desCoM[0].A1,'r', linewidth=3.0)
#ax.plot(fly.desCoM[1].A1,'g', linewidth=3.0)
#ax.plot(fly.desCoM[2].A1,'b', linewidth=3.0)

#plt.close()
'''
