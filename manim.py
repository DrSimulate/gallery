import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
np.random.seed(0)
from manim import *

class Kinematics2D(MovingCameraScene):
    def construct(self):
        # constants
        Tex.set_default(font_size=32)
        MathTex.set_default(font_size=32)
        DecimalNumber.set_default(font_size=32)
        TIMESCALE = 3.0

        # camera settings and background grid
        self.camera.frame.move_to(1.25*UP + 2.25*RIGHT)
        self.camera.frame.scale(0.75)
        self.camera.background_color = "#0A3D62"
        GRID = NumberPlane(x_range=[-30,30,0.5], y_range=[-30,30,0.5]).set_opacity(0.7)
        GRID.z_index = -100
        self.add(GRID)

        # coordinate axes
        arrow1 = Arrow(start=ORIGIN, end=RIGHT, buff=0)
        arrow2 = Arrow(start=ORIGIN, end=UP, buff=0)
        label1 = MathTex(r"\boldsymbol{e}_1").next_to(arrow1, RIGHT, buff=0.1)
        label2 = MathTex(r"\boldsymbol{e}_2").next_to(arrow2, UP, buff=0.1)
        coordinate_e = VGroup(arrow1,arrow2,label1,label2)
        self.add(coordinate_e)

        # time coordinate
        time = ValueTracker(0.00) # set a scalar parameter that varies with the time
        position_t = DOWN*1.25 + LEFT*0.5
        arrow_t = Arrow(start=ORIGIN, end=5*RIGHT, buff=0.0).shift(position_t)
        label_t = MathTex(r"t").next_to(arrow_t, RIGHT, buff=0.1)
        number_t = always_redraw(lambda: VGroup(
            MathTex(r"t = "),
            DecimalNumber(
            time.get_value(),
            num_decimal_places = 2,
            )).arrange(RIGHT,buff=0.1).next_to(time.get_value() * TIMESCALE * RIGHT, UP, buff=0.25
                        ).shift(position_t)
            )
        dot_t_ref = Dot(ORIGIN, color=WHITE, radius=0.1).shift(position_t)
        dot_t = always_redraw(
            lambda: Dot(time.get_value() * TIMESCALE * RIGHT, color=YELLOW, radius=0.1).shift(position_t)
        )
        coordinate_t = VGroup(arrow_t,label_t,number_t,dot_t_ref,dot_t)
        self.add(coordinate_t)

        # reference configuration
        reference = Triangle().stretch(1.3, dim=1).move_to(2*UP + 1.25*RIGHT)
        reference_hole1 = Circle().scale(0.1).move_to(2*UP + 1.25*RIGHT + 0.1*UP + 0.05*LEFT)
        reference_hole2 = Circle().scale(0.15).move_to(1.75*UP + 1.25*RIGHT + 0.4*DOWN + 0.3*RIGHT)
        reference_hole3 = Circle().scale(0.12).move_to(1.75*UP + 1.25*RIGHT + 0.2*DOWN + 0.2*LEFT)
        reference = Difference(reference, reference_hole1)
        reference = Difference(reference, reference_hole2)
        reference = Difference(reference, reference_hole3, color=YELLOW)
        reference.set_fill(opacity=0.3)
        reference.z_index = 20
        reference_shadow = reference.copy()
        reference_shadow.set_color(WHITE)
        reference_shadow.set_fill(opacity=0.2)
        reference_shadow.z_index = 10

        # deformed configuration
        def phi(X,t):
            x0 = (1 - 0.25*t)*X[0] + t*(X[1]-2)**2 + 2.5*t
            x1 = (1 + 0.25*t)*X[1] + 0.25*t
            x2 = X[2]
            return (x0,x1,x2)
        current = always_redraw(
            lambda: reference.copy().apply_function(lambda X: phi(X,time.get_value()))
        )

        self.add(reference_shadow,current)

        # material point
        point_coordinates = ([0.75, 1.25, 0.])
        X = Dot(point=point_coordinates)
        arrow_X = Arrow(start=ORIGIN, end=point_coordinates, buff=0)
        label_X = MathTex(r"\boldsymbol{X}"
        ).next_to(arrow_X.get_center(), 2*RIGHT, buff=0.1
        )
        x = always_redraw(
            lambda: Dot(point=phi(point_coordinates,time.get_value()), color=YELLOW)
        )
        arrow_x = always_redraw(
            lambda: Arrow(start=ORIGIN, end=phi(point_coordinates,time.get_value()), buff=0, color=YELLOW)
        )
        label_x = always_redraw(lambda: MathTex(r"\boldsymbol{x}", color=YELLOW
        ).next_to(arrow_x.get_center(), 2*DOWN+RIGHT, buff=0.1
        ))
        point = VGroup(X,arrow_X,label_X,x,arrow_x,label_x)
        self.add(point)

        # animate
        self.wait(.25)
        self.play(time.animate.set_value(1.00), rate_func=linear)
        self.wait(.5)
        self.play(time.animate.set_value(0.00), rate_func=linear)
        self.wait(.25)

class TensorComponents(MovingCameraScene):
    def construct(self):
        # constants
        Tex.set_default(font_size=96)
        MathTex.set_default(font_size=96)
        DecimalNumber.set_default(font_size=96)
        F_np = np.array([[2.0, 1.0],[0.5, 1.5]])
        F_3d = np.array([[2.0, 1.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 1.0]])

        # camera settings and background grid
        self.camera.frame.shift(LEFT * 5)
        self.camera.frame.scale(0.75)
        self.camera.background_color = "#0A3D62"
        GRID = NumberPlane(x_range=[-20,20,1], y_range=[-20,20,1]).set_opacity(0.7)
        GRID.z_index = -100

        self.camera.frame.scale(2)

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
        self.play(alpha.animate.set_value(30 * DEGREES), run_time=2)
        self.wait(0.25)
        self.play(alpha.animate.set_value(0 * DEGREES), run_time=2)
        self.wait(0.125)


# helper functions
def lighten_color(color, factor=2.):
    if factor <= 1:
        raise ValueError("Factor should be greater than 1 to lighten the color.")
    lightened_color = tuple(min(c * factor, 1.0) for c in color)  # Ensure RGB values do not exceed 1
    return lightened_color

def get_hsv_color(value,MAX_VALUE=1):
    if not 0 <= value <= MAX_VALUE:
        raise ValueError("Input value must be between 0 and the maximum value.")
    normalized_value = value / MAX_VALUE
    normalized_value_temp = normalized_value + 0.5
    if normalized_value_temp >= 1.0: normalized_value_temp -= 1.0
    color = plt.cm.hsv(normalized_value_temp)
    return mcolors.rgb2hex(lighten_color(color[:3]))

def Q(alpha):
    return np.array([[np.cos(alpha), np.sin(alpha)],
                    [-np.sin(alpha),  np.cos(alpha)]])

def rotate_matrix(F,alpha):
    return Q(alpha) @ F @ Q(alpha).T

def get_Line_coords(NUM_LINES=8*4):
    angles = np.pi/2 + np.linspace(0, 2 * np.pi, NUM_LINES, endpoint=False) # Angles from 0 to 2π
    Line_coords = np.array([np.cos(angles), np.sin(angles)]).T
    Line_coords = np.hstack([Line_coords, np.zeros((Line_coords.shape[0], 1))])
    return Line_coords

def get_Lines(X_coord,Line_coords):
    NUM_LINES = Line_coords.shape[0]
    Lines = VGroup()
    for i in range(NUM_LINES):
        Lines.add(Arrow(
            start=X_coord, end=X_coord+Line_coords[i],
            buff=0, color=get_hsv_color(i,NUM_LINES), z_index=-10))
    return Lines

def get_lines(X_coord,Line_coords,F):
    # note: time may be the ValueTracker()
    NUM_LINES = Line_coords.shape[0]
    def MyVGroup(X_coord,Line_coords,F):
        lines = VGroup()
        for i in range(NUM_LINES):
            lines.add(Arrow(
                start=X_coord,
                end=X_coord+F @ Line_coords[i],
                buff=0, color=get_hsv_color(i,NUM_LINES),z_index=30))
        return lines
    lines = MyVGroup(X_coord,Line_coords,F)
    return lines

def matrix2text(A,nd=2):
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if np.abs(A[i][j]) < 1e-6: A[i][j] = 0
    if A.shape[0] == 2:
        s = (
            r"\begin{bmatrix} " +
            f"{A[0][0]:.{nd}f}" +
            r"&" +
            f"{A[0][1]:.{nd}f}" +
            r"\\" +
            f"{A[1][0]:.{nd}f}" +
            r"&" +
            f"{A[1][1]:.{nd}f}" +
            r"\\\end{bmatrix}"
        )
    elif A.shape[0] == 3:
        s = (
            r"\begin{bmatrix} " +
            f"{A[0][0]:.{nd}f}" +
            r"&" +
            f"{A[0][1]:.{nd}f}" +
            r"&" +
            f"{A[0][2]:.{nd}f}" +
            r"\\" +
            f"{A[1][0]:.{nd}f}" +
            r"&" +
            f"{A[1][1]:.{nd}f}" +
            r"&" +
            f"{A[1][2]:.{nd}f}" +
            r"\\" +
            f"{A[2][0]:.{nd}f}" +
            r"&" +
            f"{A[2][1]:.{nd}f}" +
            r"&" +
            f"{A[2][2]:.{nd}f}" +
            r"\\\end{bmatrix}"
        )
    return s