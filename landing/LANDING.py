import os
import subprocess
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
import gains
import indexes as idx
import pinocchio as se3
from pinocchio.utils import zero as mat_zeros

# call the gepetto viewer server
gvs = subprocess.Popen(["./gepetto-viewer.sh","&"])
#gvs.kill()
print 'Loading the viewer ...'
time.sleep(2)


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
filename = lp.objects+'/parkour_structure_cage.obj'
position = se3.SE3.Identity()
position.translation += np.matrix([-.5,-0.98,1.]).T
simulator.viewer.viewer.gui.addMesh(robotNode+'cage', filename)
simulator.viewer.placeObject(robotNode+'cage', position, True)


#__ Create Solver  
solver =NProjections('Solv1', 
                     simulator.robot.q0.copy(), 
                     simulator.robot.v0.copy(), 
                     dt, simulator.robot.name, 
                     simulator.robot)


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
        self.desRF = simulator.robot.framePosition(idx.FRF,self.desPosture)
        self.desLF = simulator.robot.framePosition(idx.FLF,self.desPosture)
        self.trajRF = Traj.ConstantSE3Trajectory('RF1',self.desRF)
        self.trajLF = Traj.ConstantSE3Trajectory('LF1',self.desLF)
        self.taskRF = Task.SE3Task(solver.robot, idx.FRF, self.trajRF,'Keep Right Foot Task')
        self.taskLF = Task.SE3Task(solver.robot, idx.FLF, self.trajLF, 'Keep Left Foot Task')
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
        

class Jump:
    DURATION = 200 #200
    def __init__(self, visualize=True):
        w = gains.JumpGains()
        # Vision
        self.desGaze = np.matrix([0.,0.]).T
        self.trajGaze = Traj.ConstantNdTrajectory('Gaze',self.desGaze)
        self.target = se3.SE3.Identity()#np.matrix([0.,0.,0.]).T #3d point
        self.target.translation = np.matrix([0.,0.,0.]).T
        self.op = np.matrix([0.0,0.1,0.12]).T #operational point wrt to neck
        self.taskGaze = Task.GazeSE3Task(solver.robot, idx.FHD, self.op, self.target, self.trajGaze)
        self.taskGaze.kp = w.kp_gaze #10
        #simulator.viewer.viewer.gui.addXYZaxis(robotNode+'targetGaze', [1., 1., 0., .5], 0.03, 0.3)
        #simulator.viewer.viewer.gui.addXYZaxis(robotNode+'Vision', 0.05,0.2, [1., 1., 0., .5])
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'oMi', [1., 0.5, 0.5, .5], 0.03, 0.3)
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'iMo', [1., 0., 1., .5], 0.03, 0.3)
        # Posture
        self.desPosture = np.asmatrix(np.load(p+'/push_reff.npy')).T
        self.trajPosture = Traj.ConstantNdTrajectory('Posture',self.desPosture)
        self.taskPosture = Task.JointPostureTask(solver.robot, self.trajPosture, 'Posture')
        # Linear momentum
        #self.desMom = np.matrix([-g*22, 0., g*23, 0., 2., 0.]).T 
        self.L_ap = simulator.robot.data.hg.np.A1[0]
        self.L_v  = simulator.robot.data.hg.np.A1[2]
        self.K_ap = simulator.robot.data.hg.np.A1[4]
        #self.desMom = np.matrix([-g*12, 0., g*5, 0., 2., 0.]).T #np.sqrt(g*5)
        self.desMom = np.matrix([self.L_ap-(g*g), 0., self.L_v+(g*g), 0., 2., 0.]).T 
        self.trajMom = Traj.ConstantNdTrajectory('Momentum', self.desMom)
        self.taskLinMom = Task.MomentumTask(solver.robot, self.trajMom, 'Linear Momentum AP and V')
        self.taskLinMom.mask(np.array([1,1,1,0,0,0]))
        self.taskLinMom.kp = w.kp_lin_mom #5
        self.taskLinMom.kv = w.kv_ang_mom
        self.taskLinMom.setGain(w.gain_vector_lin_mom)
        # Angular Momentum
        self.taskAngMom = Task.MomentumTask(solver.robot, self.trajMom, 'Angular Momentum around ML')
        self.taskAngMom.mask(np.array([0,0,0,0,1,0]))
        self.taskAngMom.kp = w.kp_ang_mom #0.03
        self.taskAngMom.kv = w.kv_ang_mom
        self.taskAngMom.setGain(w.gain_vector_ang_mom)
        if visualize is True:
            self.visualizeTasks()
        
    def visualizeTasks(self):
        self.desCoM = np.asmatrix(np.load(p+'/push_comprofile.npy')).T
        cm = se3.SE3.Identity()
        cm.translation = self.desCoM[0:3,-1]
        simulator.viewer.viewer.gui.addXYZaxis(robotNode+'targetJump', [1., 1., 0., .5], 0.03, 0.3)
        simulator.viewer.placeObject(robotNode+'targetJump', cm, True)
    
class Fly():
    DURATION = 0 #278 178 187
    def __init__(self, visualize=True):
        w = gains.FlyGains()
        # Constant variation of momentum
        self.desMom = np.matrix([0., 0., 0., 0., 0., 0.]).T 
        self.trajMom = Traj.ConstantNdTrajectory('Momentum', self.desMom)
        self.taskMom = Task.FlyMomentumTask(solver.robot, self.trajMom, 'Linear Momentum AP and V')
        self.taskMom.mask(np.array([1,1,1,1,1,1]))
        self.taskMom.kv = 20

        # Posture
        self.desPosture = np.asmatrix(np.load(p+'/fly_ref.npy')).T
        self.trajPosture = Traj.ConstantNdTrajectory('Post at IC', self.desPosture)
        self.taskPosture = Task.JointPostureTask(solver.robot, self.trajPosture, 'Final Posture Task')
        self.taskPosture.kp = w.kp_posture#100
        # Pelvis orientation
        #self.desPelvis = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[:,:7].T
        #self.trajPelvis = Traj.SmoothedNdTrajectory('Pelvis Rot traj', self.desPelvis, dt, 15) 
        self.desPelvis = np.asmatrix(np.load(p+'/fly_refprofile.npy'))[-1,:7].T
        self.trajPelvis = Traj.ConstantNdTrajectory('Pelvis Rot traj', self.desPelvis)
        self.taskPelvis = Task.FreeFlyerTask(solver.robot, self.trajPelvis, 'Rotation Pelvis Task')
        self.taskPelvis.mask(np.array([0,0,0,1,1,1]))
        self.taskPelvis.kp = w.kp_pelvis
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
        for i in range(0,300):    
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

            
class Land:
    DURATION = 122
    def __init__(self, visualize=True):
        w = gains.LandGains()
        # Head
        self.desHD = simulator.robot.framePosition(IDXHD,simulator.robot.q)
        self.trajHD = Traj.ConstantSE3Trajectory('HD',self.desHD)
        self.taskHD = Task.SE3Task(solver.robot, idx.FHD, self.trajHD,'Keep Right Foot Task')
        self.taskHD.mask(np.array([0,0,0,1,1,1]))
        self.taskHD.kp = w.kp_head
        self.taskHD.kv = w.kv_head
        # Feet
        self.desRF = simulator.robot.framePosition(IDXRF,simulator.robot.q)
        self.trajRF = Traj.ConstantSE3Trajectory('RF1',self.desRF)
        self.taskRF = Task.SE3Task(solver.robot, idx.FRF, self.trajRF,'Keep Right Foot Task')
        self.taskRF.mask(np.array([1,1,1,1,1,1]))
        self.taskRF.kp = w.kp_right_foot
        self.taskRF.kv = w.kv_right_foot
        
        self.desLF = simulator.robot.framePosition(IDXLF,simulator.robot.q)
        self.trajLF = Traj.ConstantSE3Trajectory('LF1',self.desLF)
        self.taskLF = Task.SE3Task(solver.robot, idx.FLF, self.trajLF, 'Keep Left Foot Task')
        self.taskLF.mask(np.array([1,1,1,1,1,1]))
        self.taskLF.kp = w.kp_left_foot
        self.taskLF.kv = w.kv_left_foot
        
        # Ankle
        self.desRA = simulator.robot.framePosition(IDXRA,simulator.robot.q)
        self.trajRA = Traj.ConstantSE3Trajectory('RF1',self.desRA)
        self.taskRA = Task.SE3Task(solver.robot, idx.FRA, self.trajRA,'Keep Right Foot Task')
        self.taskRA.mask(np.array([1,1,1,1,1,1]))
        self.taskRA.kp = w.kp_right_ankle
        self.taskRA.kv = w.kv_right_ankle
        
        self.desLA = simulator.robot.framePosition(IDXLA,simulator.robot.q)
        self.trajLA = Traj.ConstantSE3Trajectory('LF1',self.desLA)
        self.taskLA = Task.SE3Task(solver.robot, idx.FLA, self.trajLA, 'Keep Left Foot Task')
        self.taskLA.mask(np.array([1,1,1,1,1,1]))
        self.taskLA.kp = w.kp_left_ankle
        self.taskLA.kv = w.kv_left_ankle
        
        # Linear momentum
        self.desMom = np.matrix([0., 0., 0., 0., 0., 0.]).T 
        self.trajMom = Traj.ConstantNdTrajectory('Momentum', self.desMom)
        self.taskLinMom = Task.MomentumTask(solver.robot, self.trajMom, 'Linear Momentum AP and V')
        self.taskLinMom.mask(np.array([1,1,1,0,0,0]))
        self.taskLinMom.kp = w.kp_lin_mom #9 
        self.taskLinMom.kv = w.kv_lin_mom
        self.taskLinMom.setGain(w.gain_vector_lin_mom)

        # Angular Momentum
        self.taskAngMom = Task.MomentumTask(solver.robot, self.trajMom, 'Angular Momentum around ML')
        self.taskAngMom.mask(np.array([0,0,0,1,1,1]))
        self.taskAngMom.kp = w.kp_ang_mom #20#18
        self.taskAngMom.kv = w.kv_ang_mom
        self.taskAngMom.setGain(w.gain_vector_ang_mom)
        
        
            
class Plot:
    def __init__(self):
        self.q = []
        self.hg = []
        self.p_error_hl = []
        self.p_error_ho = []
        plt.ion()

    def Momentum(self):
        plt.close()
        self.f, self.axarr = plt.subplots(2, sharex=True)
        self.axarr[0].plot(hg)
        self.axarr[0].set_title('Momentum from trial')
        self.axarr[0].legend(['*hlx','hly','*hlz','hox','*hoy','hoz'])
        self.axarr[1].plot(self.hg)
        self.axarr[1].set_title('Momentum from Simulation')
        self.axarr[1].legend(['*hlx','hly','*hlz','hox','*hoy','hoz'])
        self.y1 = self.axarr[0].viewLim._points[0,1]
        self.y2 = self.axarr[1].viewLim._points[1,1]
        self.plotPhaseLines()
        #plt.figure('Momentum Task Error')

    def plotPhaseLines(self):
        i=prepare.DURATION
        self.axarr[0].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[1].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        i+=jump.DURATION
        self.axarr[0].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[1].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        iaux=i+278
        i+=fly.DURATION
        self.axarr[0].plot((iaux, iaux), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[1].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        iaux+=land.DURATION
        i+=land.DURATION
        self.axarr[0].plot((iaux, iaux), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[1].plot((i, i), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[0].plot((0, 0), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        self.axarr[1].plot((0, 0), (self.y1, self.y2), linewidth=1, color = 'k',linestyle='--')
        plt.pause(0.0001)

    def update_line(self,i):
        self.axarr[0].lines.pop()
        self.axarr[1].lines.pop()
        self.axarr[0].plot((i, i), (self.y1, self.y2), linewidth=2, color = 'k',linestyle='-')
        self.axarr[1].plot((i, i), (self.y1, self.y2), linewidth=2, color = 'k',linestyle='-')
        plt.pause(0.0001)       

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
vi = []
plot = Plot()
prepare = PrepareToJump()
print 'Preparation Phase'
solver.addTask(prepare.taskPosture, 1)
solver.addTask(prepare.taskCoM, 1)
solver.addTask([prepare.taskRF, prepare.taskLF], 1)
t = 0.0
for i in range(0,prepare.DURATION):#150
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i,end=i+1)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
    vi+=[se3.SE3.Identity()]
print 'Jump Phase'
solver.emptyStack()
jump = Jump()
#solver.addTask(jump.taskGaze, 1)
solver.addTask(jump.taskAngMom, 1)
solver.addTask(jump.taskLinMom, 1)
solver.addTask([prepare.taskRF, prepare.taskLF], 1)
t = 0.0

for i in range(0,jump.DURATION):#100 80 180
    a = solver.inverseKinematics2nd(t)
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i+prepare.DURATION,end=i+prepare.DURATION+1)
    t += dt
    q+=[simulator.robot.q]
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
    #plot.p_error_hl+=[solver.tasks[0].p_error]
    #plot.p_error_ho+=[solver.tasks[1].p_error]
    #if simulator.robot.framePosition(IDXRF,simulator.robot.q).translation[2] <= 0.2 :
    #    break

#orienatation of left foot
'''
for i in range(0,170):#150
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
'''


#Last Push in Antero Posterior direction
print 'Fly Phase'
solver.emptyStack()
fly = Fly()
solver.addTask(fly.taskPelvis, 1)
solver.addTask(fly.taskPosture, 1)
solver.addTask(fly.taskCoM, 1)
#solver.addTask(fly.taskMom, 1)
t = 0.0
i = 0
while True:
#for i in range(0, 100):
    i +=1 
    a = solver.inverseKinematics2nd(t)
    #a[0:6]+=simulator.robot.model.gravity.vector.copy()
    simulator.increment2(simulator.robot.q, a, dt, t, False)
    # check contact with the ground
    if simulator.robot.framePosition(idx.FRF,simulator.robot.q).translation[2] <= 0.2 : #0.2
        break
    #trial.playTrial(rep=r,dt=dt,stp=1,start=i+386,end=i+387)
    t += dt
    q+=[simulator.robot.q]
    se3.ccrba(simulator.robot.model, simulator.robot.data, simulator.robot.q, simulator.robot.v)
    plot.hg+=[simulator.robot.data.hg.np.squeeze().A1]
fly.DURATION=i

print 'Land Phase'
solver.emptyStack()
land = Land()
#solver.addTask(land.taskHD, 1)
#solver.addTask([land.taskRA, land.taskLA], 1)
#solver.addTask(land.taskAngMom, 1)
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






plot.Momentum()
simulator.viewer.viewer.gui.addXYZaxis(robotNode+'neckJoint', [0., 1., 0., .5], 0.03, 0.3)
oMop = se3.SE3.Identity()
def playMotions(first=0, last=1):
    for i in range(first, last):
        #trial.playTrial(rep=r,dt=dt,stp=1,start=i,end=i+1)
        trial.play(trial.trial[r]['pinocchio_data'][i])
        simulator.viewer.display(q[i], simulator.robot.name)
        oMi = simulator.robot.framePosition(26,q[i])
        oMop.translation = oMi.translation+oMi.rotation*jump.op
        oMop.rotation = oMi.rotation.copy()
        vis = oMop.copy()
        #vis.rotation = se3.utils.rpyToMatrix(np.matrix(jump.taskGaze.vision).T)
        #vis.rotation = se3.utils.rpyToMatrix(jump.taskGaze.vision)
        #simulator.viewer.placeObject(robotNode+'iMo', vis, True)
        #simulator.viewer.placeObject(robotNode+'neckJoint', oMi, True)robotNode+'oMi'
        #simulator.viewer.placeObject(robotNode+'oMi', vi[i], True)
        #plot.update_line(i)

