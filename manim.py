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

class FiniteElements(MovingCameraScene):
    def construct(self):
        # constants
        Tex.set_default(font_size=32)
        MathTex.set_default(font_size=32)
        DecimalNumber.set_default(font_size=32)

        # camera settings and background grid
        self.camera.frame.scale(0.5)
        self.camera.background_color = "#0A3D62"

        variable1 = Variable(0, MathTex(r"u_0"), num_decimal_places=2).scale(0.5)
        variable2 = Variable(0, MathTex(r"u_1"), num_decimal_places=2).scale(0.5)
        variable3 = Variable(0, MathTex(r"u_2"), num_decimal_places=2).scale(0.5)
        variable4 = Variable(0, MathTex(r"u_3"), num_decimal_places=2).scale(0.5)
        variable5 = Variable(0, MathTex(r"u_4"), num_decimal_places=2).scale(0.5)
        variable6 = Variable(0, MathTex(r"u_5"), num_decimal_places=2).scale(0.5)
        variable1.label.set_color(YELLOW)
        variable2.label.set_color(YELLOW)
        variable3.label.set_color(YELLOW)
        variable4.label.set_color(YELLOW)
        variable5.label.set_color(YELLOW)
        variable6.label.set_color(YELLOW)
        variable1.value.set_color(YELLOW)
        variable2.value.set_color(YELLOW)
        variable3.value.set_color(YELLOW)
        variable4.value.set_color(YELLOW)
        variable5.value.set_color(YELLOW)
        variable6.value.set_color(YELLOW)

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

        v_lines_u = get_v_lines(ax_u,[0.2,0.4,0.6,0.8,1])
                
        coord_u = VGroup(ax_u, labels_u, v_lines_u)
        
        func_N1, func_N2, func_N3, func_N4, func_N5, func_N6, func_uN, func_uNd, func_uNdd = get_func_uN()

        variable1.next_to(ax_u.coords_to_point(0, variable1.tracker.get_value()),UP)
        variable1.add_updater(lambda m : m.next_to(ax_u.coords_to_point(0, variable1.tracker.get_value()),UP))
        variable2.next_to(ax_u.coords_to_point(0.2, variable2.tracker.get_value()),UP)
        variable2.add_updater(lambda m : m.next_to(ax_u.coords_to_point(0.2, variable2.tracker.get_value()),UP))
        variable3.next_to(ax_u.coords_to_point(0.4, variable3.tracker.get_value()),UP)
        variable3.add_updater(lambda m : m.next_to(ax_u.coords_to_point(0.4, variable3.tracker.get_value()),UP))
        variable4.next_to(ax_u.coords_to_point(0.6, variable4.tracker.get_value()),UP)
        variable4.add_updater(lambda m : m.next_to(ax_u.coords_to_point(0.6, variable4.tracker.get_value()),UP))
        variable5.next_to(ax_u.coords_to_point(0.8, variable5.tracker.get_value()),UP)
        variable5.add_updater(lambda m : m.next_to(ax_u.coords_to_point(0.8, variable5.tracker.get_value()),UP))
        variable6.next_to(ax_u.coords_to_point(1.0, variable6.tracker.get_value()),UP)
        variable6.add_updater(lambda m : m.next_to(ax_u.coords_to_point(1.0, variable6.tracker.get_value()),UP))

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
        
        self.add(
            coord_u,
            graph_uN,
            variable1,
            variable2,
            variable3,
            variable4,
            variable5,
            variable6,
            )
        
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
        u_val = func_u_cos(x_val)
        self.wait(0.25)
        self.play(variable1.tracker.animate.set_value(u_val[0]),
                  variable2.tracker.animate.set_value(u_val[1]),
                  variable3.tracker.animate.set_value(u_val[2]),
                  variable4.tracker.animate.set_value(u_val[3]),
                  variable5.tracker.animate.set_value(u_val[4]),
                  variable6.tracker.animate.set_value(u_val[5]))
        u_val = func_u(x_val)
        self.wait(0.25)
        self.play(variable1.tracker.animate.set_value(u_val[0]),
                  variable2.tracker.animate.set_value(u_val[1]),
                  variable3.tracker.animate.set_value(u_val[2]),
                  variable4.tracker.animate.set_value(u_val[3]),
                  variable5.tracker.animate.set_value(u_val[4]),
                  variable6.tracker.animate.set_value(u_val[5]))
        self.wait(0.25)
        self.play(variable1.tracker.animate.set_value(0),
                  variable2.tracker.animate.set_value(0),
                  variable3.tracker.animate.set_value(0),
                  variable4.tracker.animate.set_value(0),
                  variable5.tracker.animate.set_value(0),
                  variable6.tracker.animate.set_value(0))
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

# finite elements
def func_u(x):
    return - 0.5*x**2 + x

def func_ud(x):
    return - x + 1

def func_udd(x):
    return - 1

def func_u_sin(x):
    return np.sin(PI*x) * 0.5

def func_u_cos(x):
    return -np.cos(1/2*PI*x) + 0.5

def get_func_uN():
    def func_N1(x, factor=1):
        if 0 <= x <= 0.2:
            return (1-5*x) * factor
        else:
            return 0
    
    def func_N2(x, factor=1):
        if 0 <= x <= 0.2:
            return (5*x) * factor
        elif 0.2 <= x <= 0.4:
            return (1-5*(x-0.2)) * factor
        else:
            return 0
    
    def func_N3(x, factor=1):
        if 0.2 <= x <= 0.4:
            return (5*(x-0.2)) * factor
        elif 0.4 <= x <= 0.6:
            return (1-5*(x-0.4)) * factor
        else:
            return 0
    
    def func_N4(x, factor=1):
        if 0.4 <= x <= 0.6:
            return (5*(x-0.4)) * factor
        elif 0.6 <= x <= 0.8:
            return (1-5*(x-0.6)) * factor
        else:
            return 0
    
    def func_N5(x, factor=1):
        if 0.6 <= x <= 0.8:
            return (5*(x-0.6)) * factor
        elif 0.8 <= x <= 1.0:
            return (1-5*(x-0.8)) * factor
        else:
            return 0
    
    def func_N6(x, factor=1):
        if 0.8 <= x <= 1.0:
            return (5*(x-0.8)) * factor
        else:
            return 0
    
    def func_uN(x, u_values=np.zeros(6)):
        u = func_N1(x, u_values[0])
        u += func_N2(x, u_values[1])
        u += func_N3(x, u_values[2])
        u += func_N4(x, u_values[3])
        u += func_N5(x, u_values[4])
        u += func_N6(x, u_values[5])
        return u
    
    def func_uNd(x, u_values=np.zeros(6)):
        u_diff = np.diff(u_values) / 0.2
        if 0 <= x <= 0.2:
            u = u_diff[0]
        elif 0.2 <= x <= 0.4:
            u = u_diff[1]
        elif 0.4 <= x <= 0.6:
            u = u_diff[2]
        elif 0.6 <= x <= 0.8:
            u = u_diff[3]
        elif 0.8 <= x <= 1:
            u = u_diff[4]
        return u
    
    def func_uNdd(x):
        return 0
    
    return func_N1, func_N2, func_N3, func_N4, func_N5, func_N6, func_uN, func_uNd, func_uNdd

def get_v_lines(ax,x):
    y_min, y_max, _ = ax.y_range
    v_lines = VGroup()
    for i in x:
        if y_min < 0:
            v_lines += ax.get_vertical_line(ax.c2p(i, y_min, 0))
        if y_max > 0:
            v_lines += ax.get_vertical_line(ax.c2p(i, y_max, 0))
    return v_lines
