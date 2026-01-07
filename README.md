<p style="color:red; font-weight:bold; text-align:center;">
  ⚠️ ⚠️ ⚠️ Please allow your browser some time to load the animated GIFs. ⚠️ ⚠️ ⚠️
</p>

![Banner](/media/images/banner.png)

# Gallery: How to visualize computational mechanics 

Welcome to the gallery of visualizations in computational mechanics. These visualizations were created as part of the development of a series of educational video resources on computational mechanics on [YouTube](https://www.youtube.com/@DrSimulate) (Ref. 2). Below, we highlight selected example animations along with their corresponding source code, generated using the Python libraries Matplotlib and Manim, as well as the 3D animation software Blender. More detailed explanations can be found in the accompanying publication (Ref. 1). 

## Matplotlib

A straightforward way to create animations is by using the animation functionality of the Python library [Matplotlib](https://matplotlib.org/). Install Matplotlib and import it in Python.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

#### Example: Stress Tensor Components

Description: We visualize the individual components of the Cauchy stress tensor in a fixed Cartesian basis by illustrating the outward unit normal vectors (in red) and the traction vectors (in white) acting on an infinitesimal volume element. The subroutine `animate(frame)` is called once per frame and generates a plot of an infinitesimal volume element viewed from a different angle. The function `animation.FuncAnimation()` combines the static plots from each frame into a continuous animation.

Video link: [https://youtu.be/NtTVEzZS3Bg](https://youtu.be/NtTVEzZS3Bg)

```python
# !!! full code in matplotlib_animate.py !!!
# sphere
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
y = radius * np.sin(u) * np.sin(v)
x = radius * np.cos(u) * np.sin(v)
z = radius * np.cos(v)
# loop over stress components
for i in range(6):
    traction = np.matmul(sigma,normal_position_nomalized.T).T
    # animate
    fig = plt.figure(figsize=(_figsize,_figsize),dpi=_dpi)
    ax = plt.axes(projection='3d')
    def animate(frame):
        ax.cla()
        plot_coo(ax,shift=np.zeros(3))
        ax.plot_surface(x,y,z,alpha=.2,color=COLOR0)
        plot_vector_field(ax,normal_position,normal,color=COLOR1)
        plot_vector_field(ax,normal_position,factor*traction,color=COLOR0)
        ax.view_init(elev=20, azim=-60 + frame/2)
        return
    ani = animation.FuncAnimation(fig,animate,frames=FACTOR_FRAMES*FRAMES,interval=1/0.03)
    ani.save(_path + '.mov',codec="png",dpi=_dpi,bitrate=-1,savefig_kwargs={"transparent": True, "facecolor": "none"})
```

![Stress Tensor Components](/media/videos/gifs/gif_stress_components.gif)

## Manim

In the following, we highlight example animations implemented in the [Manim Community Edition](https://github.com/ManimCommunity/manim). To run these examples, install ManimCE by following the [installation instructions](https://github.com/ManimCommunity/manim?tab=readme-ov-file#installation) and import it in Python.

```python
from manim import *
```

#### Example: Kinematics 2D

Description: We visualize large deformation kinematics in two dimensions. We define a triangle in the reference configuration. This triangle is then deformed according to the deformation mapping `phi(X,t)` which depends on time through the `ValueTracker`. 

Video link: [https://youtu.be/dT30kLnQNUE](https://youtu.be/dT30kLnQNUE)

```python
# !!! full code in manim.py !!!
class Kinematics2D(MovingCameraScene):
    def construct(self):
        time = ValueTracker(0.00) # set a scalar parameter that evolves with time
        # reference configuration
        reference = Triangle().stretch(1.3, dim=1).move_to(2*UP + 1.25*RIGHT)
        # deformed configuration
        def phi(X,t):
            x0 = (1 - 0.25*t)*X[0] + t*(X[1]-2)**2 + 2.5*t
            x1 = (1 + 0.25*t)*X[1] + 0.25*t
            x2 = X[2]
            return (x0,x1,x2)
        current = always_redraw(
            lambda: reference.copy().apply_function(lambda X: phi(X,time.get_value()))
        )
        # animate
        self.wait(.25)
        self.play(time.animate.set_value(1.00), rate_func=linear)
        self.wait(.5)
        self.play(time.animate.set_value(0.00), rate_func=linear)
        self.wait(.25)
```

![Kinematics 2D](/media/videos/manim/1080p60/Kinematics2D_ManimCE_v0.18.1.gif)

#### Example: Tensor Components

Description: We visualize how the deformation gradient tensor transforms a set of unit vectors by illustrating their deformed images. In addition, we demonstrate how the tensor components change under a rotation of the basis. This example highlights that the physical transformation described by the tensor remains unchanged, even though its component representation depends on the chosen basis.

Video link: [https://youtu.be/1GwAEnegaRs](https://youtu.be/1GwAEnegaRs)

```python
# !!! full code in manim.py !!!
class TensorComponents(MovingCameraScene):
    def construct(self):
        F_np = np.array([[2.0, 1.0],[0.5, 1.5]])
        F_3d = np.array([[2.0, 1.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 1.0]])
        alpha = ValueTracker(0.00) # set a scalar parameter that varies with the angle of rotation
        # rotating grid
        grid_rot = always_redraw(lambda:
            GRID.copy().apply_matrix(Q(alpha.get_value()))
        )
        # tensor components
        F_rot_text = always_redraw(lambda:
            MathTex(r"\boldsymbol{F} = ", matrix2text(rotate_matrix(F_np,-alpha.get_value()))
                            ).move_to(LEFT * 11)
            )
        # tensor visualization
        scale = 2
        Line_coords = scale * get_Line_coords()
        lines = get_lines(ORIGIN,Line_coords,F_3d)
        Volume = Circle(color=WHITE,stroke_width=6).scale(scale)
        volume = Volume.copy().apply_function(lambda X: F_3d @ X)
        self.add(grid_rot,F_rot_text,lines,volume)
        # animate
        self.wait(0.125)
        self.play(alpha.animate.set_value(30 * DEGREES), run_time=3)
        self.wait(0.25)
        self.play(alpha.animate.set_value(0 * DEGREES), run_time=4)
        self.wait(0.125)
```

![Tensor Components](/media/videos/manim/480p15/TensorComponents_ManimCE_v0.18.1.gif)

#### Example: Finite Element Method

Description:  An unknown function has infinitely many degrees of freedom prior to discretization. Finite element discretization reduces this to a finite number, which can be visualized by adjusting the associated parameters.

Video link: [https://youtu.be/xZpESocdvn4](https://youtu.be/xZpESocdvn4)

```python
# !!! full code in manim.py !!!
class FiniteElements(MovingCameraScene):
    def construct(self):
        variable1 = Variable(0, MathTex(r"u_0"), num_decimal_places=2).scale(0.5)
        variable2 = Variable(0, MathTex(r"u_1"), num_decimal_places=2).scale(0.5)
        variable3 = Variable(0, MathTex(r"u_2"), num_decimal_places=2).scale(0.5)
        variable4 = Variable(0, MathTex(r"u_3"), num_decimal_places=2).scale(0.5)
        variable5 = Variable(0, MathTex(r"u_4"), num_decimal_places=2).scale(0.5)
        variable6 = Variable(0, MathTex(r"u_5"), num_decimal_places=2).scale(0.5)
        ax_u = Axes(
            x_range=[0, 1.05, 0.2],
            y_range=[-0.5, 0.5, 1],
            x_length=5.5,
            y_length=1.75,
            axis_config={"include_tip": False},
        )
        labels_u = VGroup(
            MathTex(r"x").next_to(ax_u.coords_to_point(1.05, 0),RIGHT),
            MathTex(r"u(x)").next_to(ax_u.coords_to_point(0, 0.5),UP,buff=0).shift(UP * 0.1),
            )
        graph_uN = ax_u.plot(lambda x : func_uN(x,np.array([variable1.tracker.get_value(),
                                                            variable2.tracker.get_value(),
                                                            variable3.tracker.get_value(),
                                                            variable4.tracker.get_value(),
                                                            variable5.tracker.get_value(),
                                                            variable6.tracker.get_value()])), x_range=[0,1], use_smoothing=False, color=YELLOW)
        
        graph_uN.add_updater(lambda m : m.become(
            ax_u.plot(lambda x : func_uN(x,np.array([variable1.tracker.get_value(),
                                                     variable2.tracker.get_value(),
                                                     variable3.tracker.get_value(),
                                                     variable4.tracker.get_value(),
                                                     variable5.tracker.get_value(),
                                                     variable6.tracker.get_value()])), x_range=[0,1], use_smoothing=False, color=YELLOW)
            ))
        # animate
        self.wait(0.125)
        x_val = np.linspace(0,1,6)
        u_val = func_u_sin(x_val)
        self.wait()
        self.play(variable1.tracker.animate.set_value(u_val[0]),
                  variable2.tracker.animate.set_value(u_val[1]),
                  variable3.tracker.animate.set_value(u_val[2]),
                  variable4.tracker.animate.set_value(u_val[3]),
                  variable5.tracker.animate.set_value(u_val[4]),
                  variable6.tracker.animate.set_value(u_val[5]))
```

![Finite Elements](/media/videos/manim/1080p60/FiniteElements_ManimCE_v0.18.1.gif)

## Blender

Objects animated using Matplotlib or Manim generally do not support realistic surfaces and textures. Such features can be created using [Blender](https://www.blender.org/). Deformations of the object can be imported from simulation results or hard-coded directly into the mesh.

#### Example: Finite Element Simulation Results in Blender

Description: The results of finite element simulations can be visualized in Blender. The example below shows a rubber specimen whose deformation was computed using the finite element method and subsequently exported from [ParaView](https://www.paraview.org/) to [Blender](https://www.blender.org/) with the [Stop-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ) add-on.

Video link: [https://youtu.be/svIs3-0t2LY](https://youtu.be/svIs3-0t2LY)

![Dogbone Simulation Rubber](/media/videos/gifs/gif_dogbone_simulation_rubber.gif)

#### Example: Imposed Object Deformation in Blender

Description: Blender’s Python scripting interface allows direct modification and deformation of meshes. For example, a prescribed deformation mapping `phi(X,t)` can be applied to a predefined object, such as the water drop shown below.

Video link: [https://youtu.be/1GwAEnegaRs](https://youtu.be/1GwAEnegaRs)

```python
# !!! full code in blender.py !!!
# material
mat = bpy.data.materials.new(name="Water")
# define deformation
def phi(XYZ,t=0):
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]
    x = (1 + 0.75*t)*X
    y = Y
    z = (1 - 0.25*t)*Z + 0.35*t*X**2 - 1.5*t
    return np.array([x, y, z])
# animate
for frame in range(1,int(TOTAL_FRAMES)+1):
    t = frame / TOTAL_FRAMES
    bpy.ops.mesh.primitive_uv_sphere_add(radius=4)
    obj = bpy.context.active_object
    mesh = obj.data
    reference = np.array([v.co[:] for v in mesh.vertices])
    for i, v in enumerate(mesh.vertices):
        v.co = Vector(phi(reference[i],t))
    for face in mesh.polygons:
        face.use_smooth = True
    visible_on_frame(obj,frame)
```

![Water Drop](/media/videos/gifs/gif_water_drop.gif)

## Software

All presented animations were created using [Matplotlib](https://matplotlib.org/) (version 3.10.1), [ManimCE](https://github.com/manimCommunity/manim) (version 0.18.1), and [Blender](https://www.blender.org/) (version 4.5).

## References

1. Flaschel, Moritz. *How to visualize computational mechanics: Animating finite elements, continuum mechanics, and tensor calculus*. arXiv preprint, 2026. DOI: []()

2. Flaschel, Moritz. *Educational video resources on computational mechanics*. YouTube, 2024. URL: [https://www.youtube.com/@DrSimulate](https://www.youtube.com/@DrSimulate)

3. Flaschel, Moritz. *Software for visualizing computational mechanics*. GitHub, 2026. URL: [https://github.com/DrSimulate](https://github.com/DrSimulate)