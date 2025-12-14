# Gallery: How to visualize computational mechanics 

<p style="color:red; font-weight:bold;">
⚠️ ⚠️ ⚠️ Please allow your browser some time to load the animated GIFs. ⚠️ ⚠️ ⚠️
</p>


Welcome to the gallery of visualizations in computational mechanics. These visualizations were created as part of the development of a series of educational video resources on computational mechanics on [YouTube](https://www.youtube.com/@DrSimulate) (Ref. 2). Below, we highlight selected example animations along with their corresponding source code, generated using the Python libraries Matplotlib and Manim, as well as the 3D animation software Blender. More detailed explanations can be found in the accompanying publication (Ref. 1). 

## Matplotlib

## Manim

In the following, we highlight example animations implemented in the [Manim Community Edition](https://github.com/ManimCommunity/manim). Install ManimCE following the [installation instructions](https://github.com/ManimCommunity/manim?tab=readme-ov-file#installation) and import it in Python.

```python
from manim import *
```

#### Example: Kinematics 2D

Description: We define a triangle in the reference configuration. This triangle is then deformed according to the deformation mapping `phi(X,t)` which depends on the time through the `ValueTracker`. 

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

## Blender

#### Example: Finite Element Simulation Results in Blender

Description: The results of finite element simulations can be visualized in Blender. The example below shows a rubber specimen whose deformation was computed using the finite element method and subsequently exported from ParaView to Blender with the [Stop-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ) add-on.

Video link: [https://youtu.be/svIs3-0t2LY](https://youtu.be/svIs3-0t2LY)

![Dogbone Simulation Rubber](/media/videos/gifs/gif_dogbone_simulation_rubber.gif)

#### Example: Imposed Object Deformation in Blender

Description: Blender’s Python scripting interface allows direct modification and deformation of meshes. For example, a prescribed deformation mapping `phi(X,t)` can be applied to a predefined object, such as the water drop shown below.

Video link: [https://youtu.be/1GwAEnegaRs](https://youtu.be/1GwAEnegaRs)

![Water Drop](/media/videos/gifs/gif_water_drop.gif)

## Software

All presented animations were created using Matplotlib (version 3.10.1), ManimCE (version 0.18.1), and Blender (version 4.5).

## References

1. Flaschel, Moritz. *How to visualize computational mechanics: Animating finite elements, continuum mechanics, and tensor calculus*. arXiv preprint, 2026. DOI: []()

2. Flaschel, Moritz. *Educational video resources on computational mechanics*. YouTube, 2024. URL: [https://www.youtube.com/@DrSimulate](https://www.youtube.com/@DrSimulate)

3. Flaschel, Moritz. *Software for visualizing computational mechanics*. GitHub, 2026. URL: [https://github.com/DrSimulate](https://github.com/DrSimulate)