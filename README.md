# Gallery: How to visualize computational mechanics 

<p style="color:red; font-weight:bold;">
⚠️ Warning: Please allow your browser some time to load the animated GIFs. ⚠️
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

## Blender


## Software

All presented animations were created using Matplotlib (version 3.10.1), ManimCE (version 0.18.1), and Blender (version 4.5).

## References

1. Flaschel, Moritz. *How to visualize computational mechanics: Animating finite elements, continuum mechanics, and tensor calculus*. arXiv preprint, 2026. DOI: []()

2. Flaschel, Moritz. *Educational video resources on computational mechanics*. YouTube, 2024. URL: [https://www.youtube.com/@DrSimulate](https://www.youtube.com/@DrSimulate)

3. Flaschel, Moritz. *Software for visualizing computational mechanics*. GitHub, 2026. URL: [https://github.com/DrSimulate](https://github.com/DrSimulate)