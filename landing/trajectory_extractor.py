import os
#os.sys.path.append('/local/gmaldona/devel/Parkour/HQP')
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io

from hqp.wrapper import Wrapper
from models.osim_parser import readOsim

import pinocchio as se3
import mocap.config as protocol


# For use in interactive python mode (ipthyon -i)  
#interactivePlot = True

class References:
    def __init__(self, participant):
        self.name = participant
        self.model_path = protocol.models_path+self.name+'.osim'
        self.mesh_path = protocol.mesh_path
        self.trial_path = protocol.trials_path
        self.motion = protocol.name
        self.phases = protocol.phases

        mn = protocol.MotionNames()
        if participant is 'Cyril':
            self.motions = mn.Cyril()
        elif participant is 'Lucas':
            self.motions = mn.Lucas()
        elif participant is 'Melvin':
            self.motions = mn.Melvin()
        elif participant is 'Michael':
            self.motions = mn.Michael()
        elif participant is 'Yoan':
            self.motions = mn.Yoan()
        else:
            print 'Participant not in the list of traceurs defined in the configuration file of the Mocap protocol'
            return
        self.trial_names = self.motions['trial_names'][0]
        self.jump_names = self.motions['jump_names'][0]
        self.fly_names = self.motions['fly_names'][0]
        self.land_names = self.motions['land_names'][0]

    def loadModel(self):
        self.human = Wrapper(self.model_path, self.mesh_path)
        self.human.initDisplay("world/"+self.name, loadModel=False)
        self.human.loadDisplayModel("world/"+self.name, self.name)

    def display(self, pose='zero'):
        self.human.initDisplay()
        if pose is 'hs':
            self.human.display(self.human.half_sitting())
        else:
            self.human.display(self.human.zero_poseDisplay())

    def store(self,store_path):
        #store_path = '/local/gmaldona/devel/biomechatronics/motions/landing/'+self.name+'/' 
        pass
    
    def getTrials(self):
        self.trial = []; self.jump = []; self.fly = []; self.land = []
        for trls in xrange(len(self.trial_names)):
            self.trial.append(readOsim(self.trial_path+'/'+self.name+'/'+self.trial_names[trls]))
            self.jump.append(readOsim(self.trial_path+'/'+self.name+'/'+self.jump_names[trls]))
            self.fly.append(readOsim(self.trial_path+'/'+self.name+'/'+self.fly_names[trls]))
            self.land.append(readOsim(self.trial_path+'/'+self.name+'/'+self.land_names[trls]))

    def playAllTrials(self, dt=0.0025):
        for trls in xrange(len(self.trial_names)):
            self.human.playForwardKinematics(self.trial[trls]['pinocchio_data'])
            time.sleep(dt)

    def playAllJumps(self, dt=0.0025):
        for trls in xrange(len(self.trial_names)):
            self.human.playForwardKinematics(self.jump[trls]['pinocchio_data'])
            time.sleep(dt)


    def playTrial(self, rep=0, dt=0.0025, stp=1, start=0, end=None):
        self.human.playForwardKinematics(self.trial[rep]['pinocchio_data'][start:end], sleep=dt, step=stp)

    def playJump(self, rep=0,  dt=0.0025, stp=1, start=0, end=None):
        self.human.playForwardKinematics(self.jump[rep]['pinocchio_data'][start:end], sleep=dt, step=stp)

    def getCoMfromTrial(self, rep=0, start=0, end=None):
        # return variable and jacobian of variable
        CM = self.human.record(self.trial[rep]['pinocchio_data'][start:end], 'com')[0][0]
        return np.squeeze(CM)
    #def getRefConfigfromTrial(self, rep=0, frame=1):
    #    se3.forwardKinematics(self.human.model, self.human.data, self.trial[rep]['pinocchio_data'][frame])
    #    return self.human.q.copy()

    def writeMat(self):
        pass
        #filename = 'Momentum_'+os.path.splitext(trial_names[trls])[0]
        #scipy.io.savemat(store_path+filename, dictionary) 

    def refs(self):
        pass

