import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import Transform
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

class KinematicsDeformationGradient2D(MovingCameraScene):
    def construct(self):
        # constants
        Tex.set_default(font_size=96)
        MathTex.set_default(font_size=96)
        DecimalNumber.set_default(font_size=96)
        TIMESCALE = 3.0

        # camera settings and background grid
        self.camera.frame.scale(2)
        self.camera.background_color = "#0A3D62"
        GRID = NumberPlane(x_range=[-30,30,1], y_range=[-30,30,1]).set_opacity(0.7)
        GRID.z_index = -100
        self.add(GRID)

        def my_phi(X,t):
            x0 = (1 + 0.75*t)*X[0]
            x1 = (1 - 0.25*t)*X[1] + 0.35*t*X[0]**2 - 1.5*t
            x2 = X[2]
            return (x0,x1,x2)
        def my_F(X,t):
            return np.array([
                [1 + 0.25*t, 0.0, 0.0],
                [0.35*t*2*X[0], 1 - 0.25*t, 0.0],
                [0.0, 0.0, 1.0]
            ])
        def get_reference():
            reference = Circle(
                        color=WHITE,
                        fill_color=GREY_D,
                        fill_opacity=1
                    ).scale(4)
            return reference
        def get_points():
            n_r = 8
            n_theta = 32
            r = np.linspace(0, 1, n_r)[1:-1]
            theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            R, T = np.meshgrid(r, theta)
            X_coord = np.column_stack((R.flatten() * np.cos(T.flatten()),
                                    R.flatten() * np.sin(T.flatten())))
            zeros = np.zeros((X_coord.shape[0], 1))
            X_coord = 4 * np.hstack([X_coord, zeros])
            return X_coord
        def CircleGroup(X_coord,time,phi,F):
            x = VGroup()
            for i in range(X_coord.shape[0]):
                x.add(
                    Circle(
                        radius=0.1,
                        color=WHITE,
                        z_index=20
                        ).apply_function(lambda X: F(X_coord[i],time) @ X).move_to(phi(X_coord[i],time))
                    )
            return x

        time = ValueTracker(0.00)
        reference = get_reference()
        current = always_redraw(
            lambda: reference.copy().apply_function(lambda X: my_phi(X,time.get_value()))
        )
        point_coord = get_points()
        circles = always_redraw(lambda: CircleGroup(point_coord,time.get_value(),my_phi,my_F))

        X1 = point_coord[131]
        F_text1 = always_redraw(lambda:
            MathTex(matrix2text(my_F(X1,time.get_value())[:2,:2])
                            ).move_to(LEFT * 9 + DOWN * 4)
            )
        Line1 = always_redraw(lambda: Line(
                start=(F_text1.get_corner(UR)),
                end=my_phi(X1,time.get_value()),
                buff=0.1,
                z_index=50)
        )

        X2 = point_coord[5]
        F_text2 = always_redraw(lambda:
            MathTex(matrix2text(my_F(X2,time.get_value())[:2,:2])
                            ).move_to(RIGHT * 10 + DOWN * 3)
            )
        Line2 = always_redraw(lambda: Line(
                start=(F_text2.get_corner(UL)),
                end=my_phi(X2,time.get_value()),
                buff=0.1,
                z_index=50)
        )

        X3 = point_coord[47]
        F_text3 = always_redraw(lambda:
            MathTex(matrix2text(my_F(X3,time.get_value())[:2,:2])
                            ).move_to(UP * 5)
            )
        Line3 = always_redraw(lambda: Line(
                start=(F_text3.get_bottom()),
                end=my_phi(X3,time.get_value()),
                buff=0.1,
                z_index=50)
        )

        self.add(current)
        self.add(circles)
        self.add(F_text1,F_text2,F_text3)
        self.add(Line1,Line2,Line3)

        # animate
        self.wait(.25)
        self.play(time.animate.set_value(1.00), rate_func=linear)
        self.wait(.5)
        self.play(time.animate.set_value(0.00), rate_func=linear)
        self.wait(.25)

SHINY_RED = '#ff3232'
SHINY_GREEN = '#00FF00'
SHINY_YELLOW = '#FFFF00'
NEON1 = '#00aaff' # blue
NEON2 = '#aa00ff' # purple
NEON3 = '#ff00aa' # pink
NEON4 = '#ffaa00' # orange
NEON5 = '#aaff00' # green
NEON6 = '#00ffaa'  # cyan / teal
COLOR_NORMAL = SHINY_RED
COLOR_TRACTION = WHITE
COLOR_FORCE = SHINY_YELLOW
COLOR_AREA = NEON6
COLOR_AREAVEC = NEON3
def lighten_hex(hex_color, amount=0.5):
    """
    Lightens the given color by moving it `amount` toward white.
    amount = 0.5  →  50% lighter
    """
    hex_color = hex_color.lstrip('#')

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # move each channel toward white (255)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)

    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

R = 1.5 # radius of sphere
MY_lam = 1.5
MY_F = np.array([
            [np.sqrt(1/MY_lam), 0.0, 0.3],
            [0.0, np.sqrt(1/MY_lam), 0.0],
            [0.0, 0.2, MY_lam],
        ])
MY_FinvT = np.linalg.inv(MY_F).T
def neo_hooke(F, p=1, mu=1):
    C = F.T @ F
    b = F @ F.T
    I = np.eye(3)
    sigma = mu*b - p*I
    P = mu*F - p*np.linalg.inv(F).T
    S = mu*I - p*np.linalg.inv(C)
    return sigma, P, S
MY_sigma, MY_P, MY_S = neo_hooke(MY_F)

def normalize(v):
    return v / np.linalg.norm(v)
def n_from_N(N):
    return normalize(MY_FinvT @ N)
def N_from_n(n):
    return normalize(MY_F.T @ n)
def rotate_vectors_to_normal(vectors, n_target):
    n0 = np.array([0.0, 0.0, 1.0])
    n = n_target / np.linalg.norm(n_target)
    k = np.cross(n0, n)
    k_norm = np.linalg.norm(k)
    if k_norm < 1e-8:
        return vectors.copy()
    k /= k_norm
    theta = np.arccos(np.clip(np.dot(n0, n), -1.0, 1.0))
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    return R @ vectors

n_vectors = 7
MY_position_multiple2 = np.zeros((3,n_vectors+1))
for i in range(n_vectors):
    theta = 2 * np.pi * i / n_vectors
    x = np.cos(theta)
    y = np.sin(theta)
    z = 0.0
    MY_position_multiple2[:, i] = [0.75*R*x, 0.75*R*y, z]
MY_position_multiple2[:, i+1] = [z, z, z]

def get_coordinates_3D(scene):
    arrow1 = Arrow3D(start=ORIGIN, end=RIGHT)
    arrow2 = Arrow3D(start=ORIGIN, end=UP)
    arrow3 = Arrow3D(start=ORIGIN, end=OUT)
    label1 = MathTex(r"\boldsymbol{e}_1").next_to(arrow1, RIGHT, buff=0.1).rotate(PI/2, axis=RIGHT)
    label2 = MathTex(r"\boldsymbol{e}_2").next_to(arrow2, UP, buff=0.2).rotate(PI/2, axis=RIGHT)
    label3 = MathTex(r"\boldsymbol{e}_3").next_to(arrow3, OUT, buff=0.1).rotate(PI/2, axis=RIGHT)
    return VGroup(arrow1,arrow2,arrow3,label1,label2,label3)

def get_area_element(
        R=1.5,
        checkerboard=True,
        color=NEON6,
        resolution=(4, 32)
        ):
        stroke_width = 1.5
        
        if checkerboard:
            colors = [color, lighten_hex(color)]
        else:
            colors = [color, color]

        area_element = Surface(
            lambda u, v: np.array([
                R * u * np.cos(v),
                R * u * np.sin(v),
                0.0,
            ]),
            u_range=[0, 1],
            v_range=[0, TAU],
            resolution=resolution,
            checkerboard_colors=colors,
        )

        if not checkerboard:
            area_element.set_stroke(width=stroke_width)

        return area_element

def get_area_element_normal(
        R=1.5,
        checkerboard=True,
        color=NEON6,
        resolution=(4, 32),
        normal=np.array([0, 0, 1])
    ):
    stroke_width = 1.5

    # Normalize the normal vector
    n = normal / np.linalg.norm(normal)

    # --- Build an orthonormal basis perpendicular to n ---
    # Pick any vector not parallel to n
    if abs(n[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(n, e1)

    # Checkerboard colors
    if checkerboard:
        colors = [color, lighten_hex(color)]
    else:
        colors = [color, color]

    # --- Surface perpendicular to n ---
    # Local parameterization: disk of radius R
    def param(u, v):
        # u in [0,1], v in [0, 2π]
        r = R * u
        return r * np.cos(v) * e1 + r * np.sin(v) * e2

    area_element = Surface(
        param,
        u_range=[0, 1],
        v_range=[0, TAU],
        resolution=resolution,
        checkerboard_colors=colors,
    )

    if not checkerboard:
        area_element.set_stroke(width=stroke_width)

    return area_element

class AreaElementRotationCauchy(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # Initial vector (normalized)
        N0 = normalize(np.array([0,0,1]))

        # ---- Tracker for time along the path ----
        t_tracker = ValueTracker(0)

        # ---- Arbitrary smooth path on the unit sphere that STARTS at N0 ----
        def N_path(t):
            angle = t  # one rotation if t ∈ [0, 2π]
            R_y = np.array([
                [ np.cos(angle), 0.0, np.sin(angle)],
                [ 0.0,           1.0, 0.0          ],
                [-np.sin(angle), 0.0, np.cos(angle)]
            ])
            N = R_y @ N0
            return normalize(N)

        # ---- Updaters ----
        def get_N():
            return N_path(t_tracker.get_value())

        def get_t_inf():
            return MY_sigma @ n_from_N(get_N())
        
        def get_pos(pos):
            return rotate_vectors_to_normal(pos, get_N())
        
        def get_arrow_t_i(i):
            return Arrow3D(
                start=MY_F @ get_pos(MY_position_multiple2[:,i]),
                end=MY_F @ get_pos(MY_position_multiple2[:,i]) + get_t_inf(),
                color=COLOR_TRACTION,
            )
        
        def get_arrow_n_i(i):
            return Arrow3D(
                start=MY_F @ get_pos(MY_position_multiple2[:,i]),
                end=MY_F @ get_pos(MY_position_multiple2[:,i]) + n_from_N(get_N()),
                color=COLOR_NORMAL,
            )

        # ---- Area element perpendicular to N(t) deformed by F ----
        area_element = always_redraw(
            lambda: get_area_element_normal(
                normal=get_N()
            ).copy().apply_matrix(MY_F)
        )

        # ---- Traction arrows ----
        arrow_t_i = VGroup()
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(0)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(1)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(2)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(3)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(4)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(5)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(6)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(7)))

        # ---- Normal arrow ----
        arrow_n_i = VGroup()
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(0)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(1)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(2)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(3)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(4)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(5)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(6)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(7)))

        # Add to scene
        self.add(area_element)
        self.add(arrow_t_i)
        self.add(arrow_n_i)
        coordinates = get_coordinates_3D(self).shift(3*DOWN + 2*LEFT)
        self.add(coordinates)

        # ---- Animate N along its path ----
        self.wait(0.25)
        self.play(
            t_tracker.animate.set_value(2 * PI),
            run_time=3,
        )
        self.wait(0.25)

class AreaElementRotation1stPiola(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # Initial vector (normalized)
        N0 = normalize(np.array([0,0,1]))

        # ---- Tracker for time along the path ----
        t_tracker = ValueTracker(0)

        # ---- Arbitrary smooth path on the unit sphere that STARTS at N0 ----
        def N_path(t):
            angle = t  # one rotation if t ∈ [0, 2π]
            R_y = np.array([
                [ np.cos(angle), 0.0, np.sin(angle)],
                [ 0.0,           1.0, 0.0          ],
                [-np.sin(angle), 0.0, np.cos(angle)]
            ])
            N = R_y @ N0
            return normalize(N)

        # ---- Updaters ----
        def get_N():
            return N_path(t_tracker.get_value())

        def get_t_inf():
            return MY_P @ get_N()
        
        def get_pos(pos):
            return rotate_vectors_to_normal(pos, get_N())
        
        def get_arrow_t_i(i):
            return Arrow3D(
                start=get_pos(MY_position_multiple2[:,i]),
                end=get_pos(MY_position_multiple2[:,i]) + get_t_inf(),
                color=COLOR_TRACTION,
            )
        
        def get_arrow_n_i(i):
            return Arrow3D(
                start=get_pos(MY_position_multiple2[:,i]),
                end=get_pos(MY_position_multiple2[:,i]) + get_N(),
                color=COLOR_NORMAL,
            )

        # ---- Area element perpendicular to N(t) ----
        area_element = always_redraw(
            lambda: get_area_element_normal(
                normal=get_N()
            )
        )

        # ---- Traction arrows ----
        arrow_t_i = VGroup()
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(0)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(1)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(2)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(3)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(4)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(5)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(6)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(7)))

        # ---- Normal arrow ----
        arrow_n_i = VGroup()
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(0)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(1)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(2)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(3)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(4)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(5)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(6)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(7)))

        # Add to scene
        self.add(area_element)
        self.add(arrow_t_i)
        self.add(arrow_n_i)

        # ---- Animate N along its path ----
        self.wait(0.25)
        self.play(
            t_tracker.animate.set_value(2 * PI),
            run_time=3,
        )
        self.wait(0.25)

class AreaElementRotation2ndPiola(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)

        # Initial vector (normalized)
        N0 = normalize(np.array([0,0,1]))

        # ---- Tracker for time along the path ----
        t_tracker = ValueTracker(0)

        # ---- Arbitrary smooth path on the unit sphere that STARTS at N0 ----
        def N_path(t):
            angle = t  # one rotation if t ∈ [0, 2π]
            R_y = np.array([
                [ np.cos(angle), 0.0, np.sin(angle)],
                [ 0.0,           1.0, 0.0          ],
                [-np.sin(angle), 0.0, np.cos(angle)]
            ])
            N = R_y @ N0
            return normalize(N)

        # ---- Updaters ----
        def get_N():
            return N_path(t_tracker.get_value())

        def get_t_inf():
            return MY_S @ get_N()
        
        def get_pos(pos):
            return rotate_vectors_to_normal(pos, get_N())
        
        def get_arrow_t_i(i):
            return Arrow3D(
                start=get_pos(MY_position_multiple2[:,i]),
                end=get_pos(MY_position_multiple2[:,i]) + get_t_inf(),
                color=COLOR_TRACTION,
            )
        
        def get_arrow_n_i(i):
            return Arrow3D(
                start=get_pos(MY_position_multiple2[:,i]),
                end=get_pos(MY_position_multiple2[:,i]) + get_N(),
                color=COLOR_NORMAL,
            )

        # ---- Area element perpendicular to N(t) ----
        area_element = always_redraw(
            lambda: get_area_element_normal(
                normal=get_N()
            )
        )

        # ---- Traction arrows ----
        arrow_t_i = VGroup()
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(0)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(1)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(2)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(3)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(4)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(5)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(6)))
        arrow_t_i.add(always_redraw(lambda: get_arrow_t_i(7)))

        # ---- Normal arrow ----
        arrow_n_i = VGroup()
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(0)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(1)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(2)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(3)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(4)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(5)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(6)))
        arrow_n_i.add(always_redraw(lambda: get_arrow_n_i(7)))

        # Add to scene
        self.add(area_element)
        self.add(arrow_t_i)
        self.add(arrow_n_i)

        # ---- Animate N along its path ----
        self.wait(0.25)
        self.play(
            t_tracker.animate.set_value(2 * PI),
            run_time=3,
        )
        self.wait(0.25)

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

class FiniteElementsMeshRefinement(MovingCameraScene):
    def construct(self):
        # constants
        Tex.set_default(font_size=32)
        MathTex.set_default(font_size=32)
        DecimalNumber.set_default(font_size=32)

        # camera settings and background grid
        self.camera.frame.scale(0.5)
        self.camera.background_color = "#0A3D62"

        # differential equation
        # u''(x) = f, u(0) = 0, u(L) = 0
        f = -4
        L = 1

        ax_u = Axes(
            x_range=[0, 1.05, 0.1],
            y_range=[-0.5, 0.5, 1],
            x_length=5.5,
            y_length=1.75,
            axis_config={"include_tip": False},
        )

        labels_u = VGroup(
            MathTex(r"x").next_to(ax_u.coords_to_point(1.05, 0),RIGHT),
            MathTex(r"u(x)").next_to(ax_u.coords_to_point(0, 0.5),UP,buff=0).shift(UP * 0.1),
            )
      
        coord_u = VGroup(ax_u, labels_u)

        graph_u_analytical = ax_u.plot(lambda x : 0.5*f*x**2 - 0.5*f*L*x, x_range=[0,1], use_smoothing=False, color=WHITE)
        
        v_lines_2 = get_v_lines(ax_u,[1/2,1])
        v_lines_3 = get_v_lines(ax_u,[1/3,2/3,1])
        v_lines_4 = get_v_lines(ax_u,[0.25,0.5,0.75,1])
        v_lines_5 = get_v_lines(ax_u,[0.2,0.4,0.6,0.8,1])

        func_uN2 = get_func_uN2()
        func_uN3 = get_func_uN3()
        func_uN4 = get_func_uN4()
        _, _, _, _, _, _, func_uN5, _, _ = get_func_uN()

        _, u2 = fem_1d_poisson(2, L=L, f=f)
        graph_uN2 = ax_u.plot(lambda x : func_uN2(x,u2), x_range=[0,1], use_smoothing=False, color=YELLOW)
        _, u3 = fem_1d_poisson(3, L=L, f=f)
        graph_uN3 = ax_u.plot(lambda x : func_uN3(x,u3), x_range=[0,1], use_smoothing=False, color=YELLOW)
        _, u4 = fem_1d_poisson(4, L=L, f=f)
        graph_uN4 = ax_u.plot(lambda x : func_uN4(x,u4), x_range=[0,1], use_smoothing=False, color=YELLOW)
        _, u5 = fem_1d_poisson(5, L=L, f=f)
        graph_uN5 = ax_u.plot(lambda x : func_uN5(x,u5), x_range=[0,1], use_smoothing=False, color=YELLOW)
        
        self.add(
            coord_u,
            graph_u_analytical,
            )
        
        # animate
        self.wait(0.25)
        self.play(
            FadeIn(v_lines_2, graph_uN2),
            )
        self.wait(0.5)
        self.play(
            FadeOut(v_lines_2, graph_uN2),
            FadeIn(v_lines_3, graph_uN3),
            )
        self.wait(0.5)
        self.play(
            FadeOut(v_lines_3, graph_uN3),
            FadeIn(v_lines_4, graph_uN4),
            )
        self.wait(0.5)
        self.play(
            FadeOut(v_lines_4, graph_uN4),
            FadeIn(v_lines_5, graph_uN5),
            )
        self.wait(0.5)
        self.play(
            FadeOut(v_lines_5, graph_uN5),
            )
        self.wait(0.25)
        

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

def get_func_uN2():
    def func_N1(x, factor=1):
        if 0 <= x <= 1/2:
            return (1-2*x) * factor
        else:
            return 0
    
    def func_N2(x, factor=1):
        if 0 <= x <= 1/2:
            return (2*x) * factor
        elif 1/2 <= x <= 1:
            return (1-2*(x-1/2)) * factor
        else:
            return 0
    
    def func_N3(x, factor=1):
        if 1/2 <= x <= 1:
            return (2*(x-1/2)) * factor
        else:
            return 0
    
    def func_uN(x, u_values=np.zeros(6)):
        u = func_N1(x, u_values[0])
        u += func_N2(x, u_values[1])
        u += func_N3(x, u_values[2])
        return u
    
    return func_uN

def get_func_uN3():
    def func_N1(x, factor=1):
        if 0 <= x <= 1/3:
            return (1-3*x) * factor
        else:
            return 0
    
    def func_N2(x, factor=1):
        if 0 <= x <= 1/3:
            return (3*x) * factor
        elif 1/3 <= x <= 2/3:
            return (1-3*(x-1/3)) * factor
        else:
            return 0
    
    def func_N3(x, factor=1):
        if 1/3 <= x <= 2/3:
            return (3*(x-1/3)) * factor
        elif 2/3 <= x <= 1:
            return (1-3*(x-2/3)) * factor
        else:
            return 0
    
    def func_N4(x, factor=1):
        if 2/3 <= x <= 1:
            return (3*(x-2/3)) * factor
        return 0
    
    def func_uN(x, u_values=np.zeros(6)):
        u = func_N1(x, u_values[0])
        u += func_N2(x, u_values[1])
        u += func_N3(x, u_values[2])
        u += func_N4(x, u_values[3])
        return u
    
    return func_uN

def get_func_uN4():
    def func_N1(x, factor=1):
        if 0 <= x <= 0.25:
            return (1-4*x) * factor
        else:
            return 0
    
    def func_N2(x, factor=1):
        if 0 <= x <= 0.25:
            return (4*x) * factor
        elif 0.25 <= x <= 0.5:
            return (1-4*(x-0.25)) * factor
        else:
            return 0
    
    def func_N3(x, factor=1):
        if 0.25 <= x <= 0.5:
            return (4*(x-0.25)) * factor
        elif 0.5 <= x <= 0.75:
            return (1-4*(x-0.5)) * factor
        else:
            return 0
    
    def func_N4(x, factor=1):
        if 0.5 <= x <= 0.75:
            return (4*(x-0.5)) * factor
        elif 0.75 <= x <= 1.0:
            return (1-4*(x-0.75)) * factor
        else:
            return 0
    
    def func_N5(x, factor=1):
        if 0.75 <= x <= 1.0:
            return (4*(x-0.75)) * factor
        else:
            return 0
    
    def func_uN(x, u_values=np.zeros(6)):
        u = func_N1(x, u_values[0])
        u += func_N2(x, u_values[1])
        u += func_N3(x, u_values[2])
        u += func_N4(x, u_values[3])
        u += func_N5(x, u_values[4])
        return u
        
    return func_uN

def get_v_lines(ax,x):
    y_min, y_max, _ = ax.y_range
    v_lines = VGroup()
    for i in x:
        if y_min < 0:
            v_lines += ax.get_vertical_line(ax.c2p(i, y_min, 0))
        if y_max > 0:
            v_lines += ax.get_vertical_line(ax.c2p(i, y_max, 0))
    return v_lines

def fem_1d_poisson(n_elements, L=1.0, f=1.0):
    """
    Solve u'' = f on [0, L] with u(0)=u(L)=0 using linear FEM.

    Parameters
    ----------
    n_elements : int
        Number of finite elements
    L : float
        Length of the domain
    f : float
        Constant right-hand side

    Returns
    -------
    x : ndarray
        Node coordinates
    u : ndarray
        FEM solution at nodes
    """
    n_nodes = n_elements + 1
    x = np.linspace(0, L, n_nodes)
    h = L / n_elements

    # Initialize global stiffness matrix and load vector
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Element stiffness matrix and load vector (linear elements)
    Ke = (1.0 / h) * np.array([[1, -1],
                               [-1, 1]])
    Fe = (-f * h / 2.0) * np.array([1, 1])

    # Assembly
    for e in range(n_elements):
        nodes = [e, e + 1]
        for i in range(2):
            F[nodes[i]] += Fe[i]
            for j in range(2):
                K[nodes[i], nodes[j]] += Ke[i, j]

    # Apply Dirichlet BCs: u(0)=0, u(L)=0
    K = K[1:-1, 1:-1]
    F = F[1:-1]

    # Solve system
    u_inner = np.linalg.solve(K, F)

    # Reconstruct full solution including boundaries
    u = np.zeros(n_nodes)
    u[1:-1] = u_inner

    return x, u