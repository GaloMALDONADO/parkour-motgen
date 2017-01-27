def MassSpring(state,t, m, k):
    # k : Newtons per metre
    # m : Kilograms
    # unpack the state vector
    x = state[0]
    xd = state[1]

    # these are our constants
    g = 9.809 # metres per second

    # compute acceleration xdd
    xdd = ((-k*x)/m) + g

    # return the two state derivatives
    return [xd, xdd]

def plotSpring(x,y):
    plt.ion()
    fig = plt.figure('CoMz')
    ax = fig.add_subplot ('111')
    ax.plot(x,y,'b', linewidth=3.0)
    ax.xlabel('TIME (sec)')
    ax.ylabel('STATES')
    ax.title('Mass-Spring System')
    ax.legend(('$x$ (m)', '$\dot{x}$ (m/sec)'))
