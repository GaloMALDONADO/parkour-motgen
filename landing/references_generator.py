from trajectory_extractor import References
import Mocap.config as conf

idx = conf.traceurs_list.index('Lucas')
trial = References(conf.traceurs_list[idx])
trial.loadModel()
trial.display()
trial.getTrials()


r=1
#_Prepare_______________________________    
trial.playTrial(r,0.0025,1,0,1)
ref1 = trial.human.q.copy()
com1 = trial.getCoMfromTrial(r,start=0,end=1)
trial.playTrial(r,0.0025,1,149,150)
com2 = trial.getCoMfromTrial(r,start=149,end=150)
ref2 = trial.human.q.copy()

#p = '/local/gmaldona/devel/biomechatronics/src/tests/refs'
p = '/galo/deve/gepetto/parkour/references'
f = p+'/prepare_com1'
np.save(f,com1)
f = p+'/prepare_com2'
np.save(f,com2)
f = p+'/prepare_ref1'
np.save(f,ref1)
f = p+'/prepare_ref2'
np.save(f,ref2)
np.load(f+'.npy')

# traj com  
tr =trial.trial[1]['pinocchio_data'][0:150]
CM = trial.human.record(tr,'com')
CM = np.array(CM[0]).squeeze()
f = p+'/prepare_comprofile'
np.save(f,CM)

#_Push_________________________________________   
# com of mass velocity at the end of the pushing    
trial.playTrial(r,0.0025,1,385,386)
com3 = trial.getCoMfromTrial(r,start=385,end=386)
ref = trial.human.q.copy()
f = p+'/push_comf'
np.save(f,com3)
# posture at the end   
f = p+'/push_reff'
np.save(f,ref)
# final com velocity
comMinusOne = trial.getCoMfromTrial(r,start=384,end=385)
comPlusOne = trial.getCoMfromTrial(r,start=386,end=387)
CM = np.vstack( [comMinusOne, com3, comPlusOne])
cmv = np.gradient(CM,0.0025)[0]
f = p+'/push_comfv'
np.save(f,cmv[1])
# profile of com 
CMp = trial.human.record(trial.trial[1]['pinocchio_data'][150:385],'com')
CMp = np.array(CMp[0]).squeeze()
f = p+'/push_comprofile'
np.save(f,CMp)
# Momentum   
qpu, vpu = trial.human.kine(trial.trial[1]['pinocchio_data'][150:385])
hg = trial.human.cam(qpu,vpu)
l = len(trial.trial[1]['pinocchio_data'][150:385])
hcx=np.zeros((l))
hcy=np.zeros((l))
hcz=np.zeros((l))
hx=np.zeros((l))
hy=np.zeros((l))
hz=np.zeros((l))
h = np.zeros((l,3))
ho = np.zeros((l,3))

for i in xrange (l-1):
    hcx[i] =  np.array(hg[i].angular[0])[0][0]
    hcy[i] =  np.array(hg[i].angular[1])[0][0]
    hcz[i] =  np.array(hg[i].angular[2])[0][0]
    hx[i] =  np.array(hg[i].linear[0])[0][0]
    hy[i] =  np.array(hg[i].linear[1])[0][0]
    hz[i] =  np.array(hg[i].linear[2])[0][0]

h = np.vstack([hx,hy,hz]).T
hc = np.vstack([hcx,hcy,hcz]).T
f = p+'/push_hprofile'
np.save(f,h)
f = p+'/push_hoprofile'
np.save(f,hc)



#fly 385   
trial.playTrial(r,0.0025,1,384,578)
tr =trial.trial[1]['pinocchio_data'][384:578]
# ref at the end   
ref = trial.human.q.copy()
f = p+'/fly_ref'
np.save(f,ref)
# profile com  
CMp2 = trial.human.record(tr,'com')
CMp2 = np.array(CMp2[0]).squeeze()
f = p+'/fly_comprofile'
np.save(f,CMp2)
#posture profile
refp = []
for i in range(300,578):
    trial.playTrial(r,0.0025,1,i,i+1)
    refp.append(trial.human.q.copy())
f = p+'/fly_refprofile'
np.save(f,refp)

# Momentum   
q, v = trial.human.kine(tr)
hg = trial.human.cam(q,v)
l = len(tr)
hcx=np.zeros((l))
hcy=np.zeros((l))
hcz=np.zeros((l))
hx=np.zeros((l))
hy=np.zeros((l))
hz=np.zeros((l))
h = np.zeros((l,3))
ho = np.zeros((l,3))

for i in xrange (l-1):
    hcx[i] =  np.array(hg[i].angular[0])[0][0]
    hcy[i] =  np.array(hg[i].angular[1])[0][0]
    hcz[i] =  np.array(hg[i].angular[2])[0][0]
    hx[i] =  np.array(hg[i].linear[0])[0][0]
    hy[i] =  np.array(hg[i].linear[1])[0][0]
    hz[i] =  np.array(hg[i].linear[2])[0][0]

h = np.vstack([hx,hy,hz]).T
hc = np.vstack([hcx,hcy,hcz]).T
f = p+'/fly_hprofile'
np.save(f,h)
f = p+'/fly_hoprofile'
np.save(f,hc)

# Land
trial.playTrial(r,0.0025,1,578,700)
# ref at the end                                                                                                
ref = trial.human.q.copy()
f = p+'/land_ref'
np.save(f,ref)

CMp3 = trial.human.record(trial.trial[1]['pinocchio_data'][578:700],'com')
CMp3 = np.array(CMp3[0]).squeeze()
f = p+'/land_comprofile'
np.save(f,CMp3)

q, v = trial.human.kine(trial.trial[1]['pinocchio_data'][578:700])

#trial.playTrial(1,0.0025,1,0,1)
#ref1 = trial.human.q.copy()
#trial.playTrial(1,0.0025,1,99,100)  
#ref2 = trial.human.q.copy() 


