import numpy as np
import indexes as idx
Ndof = 42

class JumpGains:
    def __init__(self):
        # kp
        self.kp_gaze = 10
        self.kp_lin_mom = 5
        self.kp_ang_mom = 0.03
        # kp
        self.kv_gaze = 1
        self.kv_lin_mom = 1
        self.kv_ang_mom = 1
        # gain vectors
        # linear momentum
        self.gain_vector_lin_mom = np.ones(Ndof)
        self.gain_vector_lin_mom[idx.VRSHFL]=0
        self.gain_vector_lin_mom[idx.VLSHFL]=0
        self.gain_vector_lin_mom[idx.VRELFL]=0
        self.gain_vector_lin_mom[idx.VLELFL]=0
        self.gain_vector_lin_mom[idx.VNECFL]=0
        self.gain_vector_lin_mom[idx.VBACFL]=0.5
        self.gain_vector_lin_mom[idx.VRKNFL]=0.1
        self.gain_vector_lin_mom[idx.VLKNFL]=0.1
        self.gain_vector_lin_mom[idx.VRHIFL]=0.5
        self.gain_vector_lin_mom[idx.VLHIFL]=0.5
        # angular momentum
        self.gain_vector_ang_mom = np.zeros(Ndof)
        self.gain_vector_ang_mom[idx.VRSHFL]=1
        self.gain_vector_ang_mom[idx.VLSHFL]=1
        self.gain_vector_ang_mom[idx.VRELFL]=1
        self.gain_vector_ang_mom[idx.VLELFL]=1
        
class FlyGains:
    def __init__(self):
        self.kp_posture = 100
        self.kp_pelvis = 100
        self.kv_posture = 1
        self.kv_pelvis = 1

class LandGains:
    def __init__(self):
        # kp
        self.kp_head = 1
        self.kp_right_foot = 15
        self.kp_left_foot = 15
        self.kp_right_ankle = 15
        self.kp_left_ankle = 15
        self.kp_lin_mom = 9
        self.kp_ang_mom = 20
        # kv
        self.kv_head = 1
        self.kv_right_foot = 1
        self.kv_left_foot = 1
        self.kv_right_ankle = 1
        self.kv_left_ankle = 1
        self.kv_lin_mom = 1
        self.kv_ang_mom = 1

        #gain vectors

        self.gain_vector_lin_mom = np.ones(Ndof)
        # shoulder and elbow joints are not used
        self.gain_vector_lin_mom[idx.VRSHFL]=0
        self.gain_vector_lin_mom[idx.VLSHFL]=0
        self.gain_vector_lin_mom[idx.VRELFL]=0
        self.gain_vector_lin_mom[idx.VLELFL]=0
        # hip not desired dof motion
        self.gain_vector_lin_mom[idx.VRHIFL+1]=0
        self.gain_vector_lin_mom[idx.VLHIFL+1]=0
        self.gain_vector_lin_mom[idx.VRHIFL+2]=0
        self.gain_vector_lin_mom[idx.VLHIFL+2]=0
        # back 
        self.gain_vector_lin_mom[idx.VBACFL]=0.3
        # neck is not used
        self.gain_vector_lin_mom[idx.VNECFL]=0
        # lower limbs flexion
        # hip
        self.gain_vector_lin_mom[idx.VRHIFL]=3
        self.gain_vector_lin_mom[idx.VLHIFL]=3
        # knee
        self.gain_vector_lin_mom[idx.VRKNFL]=0.1
        self.gain_vector_lin_mom[idx.VLKNFL]=0.1
        
        # Angular Momentum
        self.gain_vector_ang_mom = np.zeros(Ndof)
        self.gain_vector_ang_mom[idx.VRSHFL]=1
        self.gain_vector_ang_mom[idx.VLSHFL]=1
        self.gain_vector_ang_mom[idx.VRELFL]=1
        self.gain_vector_ang_mom[idx.VLELFL]=1
        
