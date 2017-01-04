import wrapper
import numpy as np
import matplotlib.pyplot as plt
from pinocchio.explog import *
import pinocchio as se3
import scipy
import time

plt.show ()

model_path = '/local/gmaldona/devel/biomechatronics/models/GX.osim'
robot = wrapper.Wrapper(model_path)
robot.initDisplay()
robot.display(robot.zero_poseDisplay())

# --------------------------------
def increment(q, dq):
    M = se3.SE3(se3.Quaternion(q[6, 0], q[3, 0], q[4, 0], q[5, 0]).matrix(), q[:3])
    dM = exp(dq[:6])
    M = M * dM
    q[:3] = M.translation
    q[3:7] = se3.Quaternion(M.rotation).coeffs()
    # right hip 7
    M = se3.Quaternion(q[10, 0], q[7, 0], q[8, 0], q[9, 0]).matrix()
    dM = exp(dq[6:9])
    M = M * dM
    q[7:11] = se3.Quaternion(M).coeffs()
    q[11:15] += dq[9:13]
    # left hip 15
    M = se3.Quaternion(q[18, 0], q[15, 0], q[16, 0], q[17, 0]).matrix()
    dM = exp(dq[13:16])
    M = M * dM
    q[15:19] = se3.Quaternion(M).coeffs()
    q[19:23] += dq[16:20]
    # back 23
    M = se3.Quaternion(q[26, 0], q[23, 0], q[24, 0], q[25, 0]).matrix()
    dM = exp(dq[20:23])
    M = M * dM
    q[23:27] = se3.Quaternion(M).coeffs()
    # neck 27
    M = se3.Quaternion(q[30, 0], q[27, 0], q[28, 0], q[29, 0]).matrix()
    dM = exp(dq[23:26])
    M = M * dM
    q[27:31] = se3.Quaternion(M).coeffs()
    # right acromial 31
    M = se3.Quaternion(q[34, 0], q[31, 0], q[32, 0], q[33, 0]).matrix()
    dM = exp(dq[26:29])
    M = M * dM
    q[31:35] = se3.Quaternion(M).coeffs()
    q[35:40] += dq[29:34]
    # left acromial 40
    M = se3.Quaternion(q[43, 0], q[40, 0], q[41, 0], q[42, 0]).matrix()
    dM = exp(dq[34:37])
    M = M * dM
    q[40:44] = se3.Quaternion(M).coeffs()
    q[44:] += dq[37:]
    return q

def errorInSE3( M,Mdes):
  '''
    Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
    between M and Mdes, both element of SE3.
  '''
  error = se3.log(M.inverse()*Mdes)
  return error.vector


def null(A, eps=1e-12):
    '''Compute a base of the null space of A.'''
    u, s, vh = np.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)



#_GENERAL CONFIGURATION_______________________________
# -------------------------------
N = 50
dt = 0.0025
k1 = 1
k2 = 0.08
rf = robot.model.getJointId('mtp_r')
lf = robot.model.getJointId('mtp_l')

def ePD(x, xdes, kp, kd):
    return -kp*(x-xdes)-kd(xdot)
def eP(x,xdes,kp):
    return -kp(x-xdes)

#_PREPARE_________________________________________________
p = '/local/gmaldona/devel/biomechatronics/src/tests/refs/'
pr_com1_ref = np.asmatrix(np.load(p+'prepare_com1.npy')).T
pr_com2_ref = np.asmatrix(np.load(p+'prepare_com2.npy')).T
pr_q1_ref = np.asmatrix(np.load(p+'prepare_ref1.npy')).T
pr_q2_ref = np.asmatrix(np.load(p+'prepare_ref2.npy')).T
rf_des = robot.position(pr_q2_ref,rf)
lf_des = robot.position(pr_q2_ref,lf)

robot.display(pr_q1_ref)
time.sleep(1)
for i in range(50):
    #t = i*dt
    #_LF
    kp = 205 #15
    kv = 2*np.sqrt(kp)
    Jlf = robot.jacobian(robot.q,lf).copy()
    Jlf[:3] = robot.position(robot.q,lf).rotation*Jlf[:3]  #w.r.t. world
    error = errorInSE3(robot.position(robot.q,lf), lf_des)
    #error_dot = np.gradient(error[i]-error[i-1])
    errLF = kp*error  
    #_RF
    Jrf = robot.jacobian(robot.q,rf).copy()
    Jrf[:3] = robot.position(robot.q,rf).rotation*Jrf[:3]  #w.r.t. world
    errRF = 205*errorInSE3(robot.position(robot.q,rf), rf_des)
    #_COM
    Jcom = robot.Jcom(robot.q)
    errCOM = 0.1*(pr_com2_ref-robot.com(robot.q))#0.1
    #_POSTURE
    errPost = 0.3*se3.differentiate(robot.model, robot.q,pr_q2_ref)#3
    Jpost = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])
    #_TASK1 STACK
    J1 = np.vstack([Jlf, Jrf, Jcom])
    err1 = np.vstack([errLF, errRF, errCOM])
    #_TASK2 STACK
    #J2 = np.vstack([Jcom])
    #err2 = np.vstack([errCOM])
    #_TASK3 STACK
    J2 = np.vstack([Jpost])
    err2 = np.vstack([errPost[6:]])
    #_Hierarchy solver
    qdot = np.linalg.pinv(J1)*err1
    Z = null(J1)
    qdot += Z*np.linalg.pinv(J2*Z)*(err2 - J2*qdot)
    #Z2 = null(J2)
    #qdot += Z2*np.linalg.pinv(J3*Z2)*(err3 - J3*qdot)
    #_INTEGRATE
    robot.increment(robot.q,qdot)
    #_DISPLAY
    robot.display(robot.q)
    
    #time.sleep(0.025)
    #e = np.linalg.norm(err1)+np.linalg.norm(err2)
    #if e < 0.05:
    #    print 'true'
    #    break
    

#_PUSH_____________________________________________________
pu_com_refp = np.asmatrix(np.load(p+'push_comprofile.npy'))
pu_qf_ref = np.asmatrix(np.load(p+'push_reff.npy')).T
rf_des = robot.position(pu_qf_ref,rf).copy()
lf_des = robot.position(pu_qf_ref,lf).copy()
p_h_profile = np.asmatrix(np.load(p+'push_hprofile.npy'))
p_h_profile[1] = p_h_profile[2].copy()
p_h_profile[0] = p_h_profile[1].copy()
l = len(p_h_profile)
p_h_profile[l-2] = p_h_profile[l-3].copy()
p_h_profile[l-1] = p_h_profile[l-2].copy()
p_ho_profile = np.asmatrix(np.load(p+'push_hoprofile.npy'))
p_ho_profile[1] = p_ho_profile[2].copy()
p_ho_profile[0] = p_ho_profile[1].copy()
l = len(p_h_profile)
p_ho_profile[l-2] = p_ho_profile[l-3].copy()
p_ho_profile[l-1] = p_ho_profile[l-2].copy()
#time.sleep(1)    
#robot.display(pr_q2_ref)
se3.forwardKinematics(robot.model, robot.data, robot.q)
JMom = se3.ccrba(robot.model, robot.data, robot.q, qdot)
JAng = JMom[3:6].copy()
errAng = np.asmatrix(np.zeros(3)).T
errMom = np.asmatrix(np.zeros(6)).T
ho = []
ho.append(robot.data.hg.copy())
robot.com(robot.q)

for i in range(len(pu_com_refp)):
    #_LF
    Jlf = robot.jacobian(robot.q,lf).copy()
    Jlf[:3] = robot.position(robot.q,lf).rotation*Jlf[:3]  #w.r.t. world
    errLF = 0.013*errorInSE3(robot.position(robot.q,lf), lf_des)#0.0001
    #_RF
    Jrf = robot.jacobian(robot.q,rf).copy()
    Jrf[:3] = robot.position(robot.q,rf).rotation*Jrf[:3]  #w.r.t. world
    errRF = 0.013*errorInSE3(robot.position(robot.q,rf), rf_des)#0.001
    #_CoM profile
    Jcom = robot.Jcom(robot.q)#[[0,2],:]   
    errCOM = 1.8*(pu_com_refp[i].T - robot.com(robot.q) )#[[0,2],0]
    qdot = np.linalg.pinv(Jcom)*errCOM
    #_POSTURE
    errPost = 0.002*se3.differentiate(robot.model, robot.q, pu_qf_ref)[6:]#0.001
    Jpost = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])
    if i > 1:
        kp_l = 0.01 * 70 
        kp_a = 0.005
        kv_l = 2*np.sqrt(kp_l)
        errAng = -kp_a*(ho[i].angular-p_ho_profile[i].T)
        errLin = -kp_l*(robot.com(robot.q)-pu_com_refp[i].T) - kv_l*(ho[i].linear-p_h_profile[i].T)
        errMom = np.vstack([errLin, errAng])
        JAng = JMom[3:6].copy() 

    #_TASK1 STACK                                                                                            
    J1 = np.vstack([Jrf, Jlf, Jcom])
    err1 = np.vstack([errRF, errLF, errCOM])
    #_TASK2
    J2 = np.vstack([JAng, Jpost]) 
    err2 = np.vstack([errAng, errPost])
    #_Hierarchy solver
    qdot = np.linalg.pinv(J1)*err1    
    Z = null(J1)
    qdot += Z*np.linalg.pinv(J2*Z)*(err2 - J2*qdot)
    
    #3rd task
    #Z2 = null(np.linalg.pinv(J2*Z))
    #qdot += Z2*np.linalg.pinv(J3*Z2)*(err3 - J3*qdot)
    #_INTEGRATE
    robot.increment(robot.q,qdot)
    
    #_DISPLAY
    robot.display(robot.q)
    #save previous
    se3.ccrba(robot.model, robot.data, robot.q, qdot)
    ho.append(robot.data.hg.copy())

#_FLY__________________
f_h_profile = np.asmatrix(np.load(p+'fly_hprofile.npy'))
f_h_profile[1] = f_h_profile[2].copy()
f_h_profile[0] = f_h_profile[1].copy()
l = len(f_h_profile)
f_h_profile[l-2] = f_h_profile[l-3].copy()
f_h_profile[l-1] = f_h_profile[l-2].copy()
f_ho_profile = np.asmatrix(np.load(p+'fly_hoprofile.npy'))
f_ho_profile[1] = f_ho_profile[2].copy()
f_ho_profile[0] = f_ho_profile[1].copy()
l = len(f_h_profile)
f_ho_profile[l-2] = f_ho_profile[l-3].copy()
f_ho_profile[l-1] = f_ho_profile[l-2].copy()
f_com_refp = np.asmatrix(np.load(p+'fly_comprofile.npy'))
f_qf_ref = np.asmatrix(np.load(p+'fly_ref.npy')).T
rf_des = robot.position(f_qf_ref,rf).copy()
lf_des = robot.position(f_qf_ref,lf).copy()
f_profref = np.asmatrix(np.load(p+'fly_refprofile.npy'))
visual_target= se3.SE3.Identity()
#visual_target.translation = np.matrix([-0.5,0.,0.]).T
robot.viewer.gui.addXYZaxis('world/targetVis', [1., 1., 0., .5], 0.03, 0.3)
robot.placeObject('world/targetVis', visual_target, True)

#robot.display(pr_q2_ref)
from biomechanics.maths import rotation_matrix, rotation_from_matrix
H = se3.SE3.Identity()
nk = robot.model.getJointId('neck')
Hvis = se3.SE3.Identity()
com_dot = np.gradient(f_com_refp)
edot = np.zeros((3,1))
dt = 1/400
ho = []
ang_mom = []
for i in range(len(f_com_refp)):
    if i is 0:
        se3.forwardKinematics(robot.model, robot.data, robot.q)
        Qprev = robot.q.copy()
        CMprev = robot.com(Qprev.copy())
        JMom = se3.ccrba(robot.model, robot.data, robot.q, qdot)
        Jang = JMom[3:6].copy()
        errAng = np.asmatrix(np.zeros(3)).T
        errMom = np.asmatrix(np.zeros(6)).T
        ho.append(robot.data.hg.copy())
        
    #_LF
    e = errorInSE3(robot.position(robot.q,lf), lf_des)
    kmax = 0.01
    kmin = 1
    beta = 1 #velocity of the transition
    kp = (kmin-kmax)*np.exp(-beta*np.linalg.norm(e)) + kmax
    Jlf = robot.jacobian(robot.q,lf).copy()
    Jlf[:3] = robot.position(robot.q,lf).rotation*Jlf[:3]  #w.r.t. world
    errLF = kp*e #0.3
    #_RF
    e = errorInSE3(robot.position(robot.q,rf), rf_des)
    kmax = 0.01
    kmin = 1
    beta = 1 #velocity of the transition
    kp = (kmin-kmax)*np.exp(-beta*np.linalg.norm(e)) + kmax
    Jrf = robot.jacobian(robot.q,rf).copy()
    Jrf[:3] = robot.position(robot.q,rf).rotation*Jrf[:3]  #w.r.t. world
    errRF = kp*e
    #_CoM profile
    Jcom = robot.Jcom(robot.q)
    e = f_com_refp[i].T-robot.com(robot.q)
    kmax = 0.5
    kmin = 2
    beta = 1.1 #velocity of the transition
    kp = (kmin-kmax)*np.exp(-beta*np.linalg.norm(e)) + kmax
    errCOM = 1.5*e     
    #qdot = np.linalg.pinv(Jcom)*errCOM
    #_POSTURE
    errPost = 0.01*se3.differentiate(robot.model, robot.q, f_qf_ref)[6:]
    Jpost = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])
    #Jpost[20,26]=10
    #Jpost[28,34]=10
    #_ORIENTATION PELVIS
    errPelv = 0.0001*se3.differentiate(robot.model, robot.q, f_profref[i])
    Jpelv = np.hstack([np.zeros([robot.nv, 3]), np.eye(robot.nv,3), np.zeros([robot.nv,robot.nv-6])])
    #_VISION
    #op_point = robot.data.oMi[nk].copy()
    #op_point.translation = op_point.translation + np.matrix([0.,0.,0.15]).T
    iMp = robot.data.oMi[nk].inverse().act(visual_target)
    d_norm = np.linalg.norm(iMp.translation)
    axis_dir = np.array( (0, 
                          iMp.translation[2,0]/d_norm,
                          -iMp.translation[1,0]/d_norm))
    theta = -np.arccos(iMp.translation[0]/d_norm)
    H.rotation = np.asmatrix(rotation_matrix(axis_dir,theta))
    #m = se3.utils.matrixToRpy(Hrot)
    #m2 = np.matrix( [-m[0,0], m[1,0], m[2,0]] ).T
    #H.rotation = se3.utils.rpyToMatrix(m2)
    #H.rotation = se3.utils.rotate('y', np.pi/2) * H.rotation.copy()
    #H.rotation = se3.utils.rotate('x',np.pi) * H.rotation
    #robot.viewer.gui.addXYZaxis('world/op_gaze', [1., 0., 0., .5], .03, .5)
    #robot.placeObject('world/op_gaze', op_point , True)
    Jvis = robot.jacobian(robot.q,nk).copy()[4:5]
    errVis = 0.5*errorInSE3(robot.position(robot.q,nk), H)[4:5]#4:5
    #print errVis
    # ANGULAR MOMENTUM constant
    if i > 4:
        errAng = -0.5*(ho[i-1].angular-ho[i-2].angular)
        errLin = -0.0001*(ho[i-1].linear-ho[i-2].linear)
        errMom = np.vstack([errLin, errAng])
        JAng = JMom[3:6].copy() 
    #_TASK1 STACK                                                                                          
    J1 = np.vstack([Jcom, Jang]) 
    err1 =np.vstack([errCOM, errAng])
    #_TASK2
    J2 = np.vstack([Jrf, Jlf, Jpost, Jpelv]) 
    err2 = np.vstack([errRF, errLF,errPost, errPelv])
    #_TASK3
    J3 = np.vstack([Jpost]) 
    err3 = np.vstack([errPost])
    #_Hierarchy solver
    qdot = np.linalg.pinv(J1)*err1    
    Z = null(J1)
    qdot += Z*np.linalg.pinv(J2*Z)*(err2 - J2*qdot)
    #Z2 = null(J2)
    #qdot += Z2*np.linalg.pinv(J3*Z2)*(err3 - J3*qdot)
    #_INTEGRATE
    robot.increment(robot.q,qdot)
    #_DISPLAY
    robot.display(robot.q)
    se3.ccrba(robot.model, robot.data, robot.q, qdot)
    ho.append(robot.data.hg.copy())
    ang_mom.append(robot.data.hg.angular.copy())
    #print time.time()
    #time.sleep(self.1)

#_LAND
l_com_refp = np.asmatrix(np.load(p+'land_comprofile.npy'))
l_qf_ref = np.asmatrix(np.load(p+'land_ref.npy')).T
rf_des = robot.position(l_qf_ref,rf).copy()
lf_des = robot.position(l_qf_ref,lf).copy()


W = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])

for i in range(len(l_com_refp)):
    #_LF
    Jlf = robot.jacobian(robot.q,lf)[:3]
    Jlf = robot.position(robot.q,lf).rotation*Jlf  #w.r.t. world
    errLF = 0.1*(lf_des.translation-robot.position(robot.q,lf).translation)
    #_RF
    Jrf = robot.jacobian(robot.q,rf)[:3]
    Jrf = robot.position(robot.q,rf).rotation*Jrf  #w.r.t. world
    errRF = 0.1*(rf_des.translation-robot.position(robot.q,rf).translation)
    #_CoM profile
    Jcom = robot.Jcom(robot.q)    
    errCOM = -(robot.com(robot.q)-l_com_refp[i].T )
    qdot = np.linalg.pinv(Jcom)*errCOM
    #_POSTURE
    errPost =0.1*se3.differentiate(robot.model, robot.q, l_qf_ref)[6:]
    Jpost = np.hstack([np.zeros([robot.nv-6,6]),np.eye(robot.nv-6)])
    #_TASK1 STACK                                                                                              
    J1 = np.vstack([Jlf,Jrf])
    err1 = np.vstack([errLF,errRF])
    #_TASK2
    J2 = np.vstack([Jcom, Jpost]) 
    err2 = np.vstack([errCOM, errPost])
    #_Hierarchy solver
    qdot = np.linalg.pinv(J1)*err1    
    Z = null(J1)
    qdot += Z*np.linalg.pinv(J2*Z)*(err2 - J2*qdot)
    #_INTEGRATE
    robot.increment(robot.q,qdot)
    #q = increment(robot.q,qdot)
    #_DISPLAY
    robot.display(robot.q)
    #robot.display(q)
    #time.sleep(0.01)


