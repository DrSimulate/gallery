import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# helper functions
COLOR0 = np.array([255, 255, 255])/255 # white
COLOR1 = np.array([255, 0, 0])/255 # red
COLOR2 = np.array([0, 255, 0])/255 # green
COLOR3 = np.array([0, 0, 255])/255 # blue
COLOR4 = np.array([255, 255, 0])/255 # yellow
COLOR5 = np.array([255, 0, 255])/255 # purple
COLOR6 = np.array([0, 255, 255])/255 # turquise

def plot_vector_field(ax,nodes,vector,linewidths=0.3,color=COLOR0):
    ax.quiver(nodes[:,0], nodes[:,1], nodes[:,2],
              vector[:,0], vector[:,1], vector[:,2],linewidths=linewidths,length=1.0,normalize=False,color=color)
    return ax

def plot_coo(ax,scale=.25,shift=-np.array([.25,.25,.25]),linewidths=0.5,fontsize=5,shifttxt=0.5):
    # plot coordinate system arrows
    if ax.name != "3d":
        e0 = np.array([0,0]) + shift[0:1]
        e1 = np.array([scale,0]); e1txt = e0 + (1+shifttxt)*e1
        e2 = np.array([0,scale]); e2txt = e0 + (1+shifttxt)*e2
        ax.quiver(e0[0],e0[1],e1[0],e1[1],scale=1,linewidths=linewidths,headwidth=2,headlength=1,headaxislength=1,color=COLOR4)
        ax.quiver(e0[0],e0[1],e2[0],e2[1],scale=1,linewidths=linewidths,headwidth=2,headlength=1,headaxislength=1,color=COLOR6)
        ax.text(e1txt[0],e1txt[1],r'$x_1$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e2txt[0],e2txt[1],r'$x_2$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
    if ax.name == "3d":
        e0 = np.array([0,0,0]) + shift
        e1 = np.array([scale,0,0]); e1txt = e0 + (1+shifttxt)*e1
        e2 = np.array([0,scale,0]); e2txt = e0 + (1+shifttxt)*e2
        e3 = np.array([0,0,scale]); e3txt = e0 + (1+shifttxt)*e3
        ax.quiver(e0[0],e0[1],e0[2],e1[0],e1[1],e1[2],linewidths=1,colors=COLOR4)
        ax.quiver(e0[0],e0[1],e0[2],e2[0],e2[1],e2[2],linewidths=1,colors=COLOR5)
        ax.quiver(e0[0],e0[1],e0[2],e3[0],e3[1],e3[2],linewidths=1,colors=COLOR6)
        ax.text(e1txt[0],e1txt[1],e1txt[2],r'$x_1$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e2txt[0],e2txt[1],e2txt[2],r'$x_2$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e3txt[0],e3txt[1],e3txt[2],r'$x_3$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
    return ax

# constants
radius = 0.5
unit = 0.1
sigma_given = 5 * unit
factor = 1

# sphere
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
y = radius * np.sin(u) * np.sin(v)
x = radius * np.cos(u) * np.sin(v)
z = radius * np.cos(v)
normal_position_nomalized = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [-1,0,0],
    [0,-1,0],
    [0,0,-1],
    [1,1,0],
    [0,1,1],
    [1,0,1],
    [-1,-1,0],
    [0,-1,-1],
    [-1,0,-1],
    [1,-1,0],
    [0,1,-1],
    [1,0,-1],
    [-1,1,0],
    [0,-1,1],
    [-1,0,1],
    [1,1,1],
    [-1,1,1],
    [1,-1,1],
    [1,1,-1],
    [1,-1,-1],
    [-1,1,-1],
    [-1,-1,1],
    [-1,-1,-1],
    ])
normal_position_nomalized = normal_position_nomalized / np.repeat(np.array([np.sum(np.abs(normal_position_nomalized)**2,axis=-1)**(1./2)]).T,3,axis=1)
normal_position = radius * np.copy(normal_position_nomalized)
normal = unit * np.copy(normal_position_nomalized)

# loop over stress components
for i in range(1):
    COMPONENT = 1 + i
    # COMPONENT = 1 --> sigma_11
    # COMPONENT = 2 --> sigma_22
    # COMPONENT = 3 --> sigma_33
    # COMPONENT = 4 --> sigma_12 & sigma_21
    # COMPONENT = 5 --> sigma_13 & sigma_31
    # COMPONENT = 6 --> sigma_23 & sigma_32
    sigma = np.zeros([3,3])
    if COMPONENT == 1:
        sigma[0,0] = sigma_given
    if COMPONENT == 2:
        sigma[1,1] = sigma_given
    if COMPONENT == 3:
        sigma[2,2] = sigma_given        
    if COMPONENT == 4:
        sigma[0,1] = sigma_given
        sigma[1,0] = sigma_given
    if COMPONENT == 5:
        sigma[0,2] = sigma_given
        sigma[2,0] = sigma_given
    if COMPONENT == 6:
        sigma[1,2] = sigma_given
        sigma[2,1] = sigma_given
        
    traction = np.matmul(sigma,normal_position_nomalized.T).T
    
    # animate
    _dpi = 600
    _figsize = 2
    _path = 'media/videos/matplotlib_animate/stress_element_sphere_component_' + str(int(COMPONENT)) + "_animation"
    FACTOR_FRAMES = 12
    FRAMES = 60
    fig = plt.figure(figsize=(_figsize,_figsize),dpi=_dpi)
    ax = plt.axes(projection='3d')
    def animate(frame):
        ax.cla() # ax.clf() # ax.collections.clear()
        plot_coo(ax,shift=np.zeros(3))
        ax.plot_surface(x,y,z,alpha=.2,color=COLOR0)
        plot_vector_field(ax,normal_position,normal,color=COLOR1)
        if factor > 1e-9:
            plot_vector_field(ax,normal_position,factor*traction,color=COLOR0)
        
        # set black background
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.view_init(elev=20, azim=-60 + frame/2)
        ax.set_xlim(-.75, .75)
        ax.set_ylim(-.75, .75)
        ax.set_zlim(-.75, .75)
        plt.axis('off')
        if frame == 0:
            ax.set_box_aspect((1,1,1))
        
        if frame == 0: fig.savefig(_path + '_start.png', transparent=True)
        if frame == FRAMES/2: fig.savefig(_path + '_middle.png', transparent=True)
        if frame == FRAMES - 1: fig.savefig(_path + '_end.png', transparent=True)
        
        return
    
    # frames = maximum number of frames
    # interval = milliseconds (0.001 seconds) between two frames
    # interval = 1/0.03 leads to 30 frames per second
    ani = animation.FuncAnimation(fig,animate,frames=FACTOR_FRAMES*FRAMES,interval=1/0.03)
    
    # for transparent background
    ani.save(_path + '.mov',
            codec="png",
            dpi=_dpi,
            bitrate=-1,
            savefig_kwargs={"transparent": True, "facecolor": "none"})
    

