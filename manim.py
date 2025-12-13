import numpy as np
np.random.seed(0)
from manim import *

class Kinematics(MovingCameraScene):
    def construct(self):
        # constants
        TIMESCALE = 3.0
        Tex.set_default(font_size=32)
        MathTex.set_default(font_size=32)
        DecimalNumber.set_default(font_size=32)

        # camera setttings and background grid
        self.camera.frame.move_to(1.25*UP + 2*RIGHT)
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
        time = ValueTracker(0.00)
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


        # coordinates = get_coordinates(self)
        # reference = get_reference()
        # reference_shadow = get_reference_shadow(reference)
        # time = ValueTracker(0.00)
        # time_coordinate = get_time_coordinate(time)
        # current = always_redraw(
        #     lambda: reference.copy().apply_function(lambda X: phi_example1(X,time.get_value()))
        # )

        # X = MyDot(point=([1.0, 2.0, 0.]), color=GRAY_A)
        # x = always_redraw(
        #     lambda: MyDot(point=phi_example1(([1.0, 2.0, 0.]),time.get_value()), color=COLOR_BALL)
        # )
        
        # arrow_X = Arrow(start=ORIGIN, end=([1.0, 2.0, 0.]), buff=0, color=GRAY_A)
        # label_X = MathTex(r"\boldsymbol{X}",color=GRAY_A
        # ).next_to(arrow_X.get_center(), UP+2*RIGHT, buff=0.1
        # )
        
        # arrow_x = always_redraw(
        #     lambda: Arrow(start=ORIGIN, end=phi_example1(([1.0, 2.0, 0.]),time.get_value()), buff=0, color=COLOR_BALL)
        # )
        # arrow_x_temp = Arrow(start=ORIGIN, end=phi_example1(([1.0, 2.0, 0.]),1.0), buff=0, color=COLOR_BALL)
        # label_x = always_redraw(lambda: MathTex(r"\boldsymbol{x}",color=COLOR_BALL
        # ).next_to(arrow_x.get_center(), DOWN+RIGHT, buff=0.1
        # ))
        # label_x_temp = MathTex(r"\boldsymbol{x}",color=COLOR_BALL
        # ).next_to(arrow_x_temp.get_center(), DOWN+RIGHT, buff=0.1
        # )
        
        # arrow_u = always_redraw(
        #     lambda: Arrow(start=([1.0, 2.0, 0.]), end=phi_example1(([1.0, 2.0, 0.]),time.get_value()), buff=0, color=COLOR_DISP)
        # )
        # arrow_u_temp = Arrow(start=([1.0, 2.0, 0.]), end=phi_example1(([1.0, 2.0, 0.]),1.0), buff=0, color=COLOR_DISP)
        # label_u = always_redraw(lambda: MathTex(r"\boldsymbol{u}",color=COLOR_DISP
        # ).next_to(arrow_u.get_center(), UP+LEFT, buff=0.1
        # ))
        # label_u_temp = MathTex(r"\boldsymbol{u}",color=COLOR_DISP
        # ).next_to(arrow_u_temp.get_center(), UP+LEFT, buff=0.1
        # )

        # formula_X = MathTex(r"\boldsymbol{X}=\begin{bmatrix}X_1\\X_2\\X_3\\\end{bmatrix}").move_to(LEFT*3 + 4*UP)
        # formula_x = MathTex(r"\boldsymbol{x}=\begin{bmatrix}x_1\\x_2\\x_3\\\end{bmatrix}").move_to(RIGHT*7 + 4*UP) #.next_to(formula_X, RIGHT, buff=0.5)

        # self.add(reference_shadow)
        # self.add(current)
        # self.add(coordinates)
        # self.add(time_coordinate)
        # self.wait(WAIT)
        # self.play(FadeIn(X,x))
        # self.wait(WAIT)
        # self.play(time.animate.set_value(1.00), rate_func=linear, run_time=1.5)
        # self.wait(WAIT)
        # self.play(Create(arrow_X))
        # self.play(Write(label_X))
        # self.play(Write(formula_X))
        # self.wait(WAIT)
        # self.play(Create(arrow_x_temp))
        # self.play(Write(label_x_temp))
        # self.play(Write(formula_x))
        # self.wait(WAIT)
        # self.add(arrow_x,label_x)
        # self.remove(arrow_x_temp,label_x_temp)
        # self.play(time.animate.set_value(0.00), rate_func=linear, run_time=.75)
        # self.wait(.25)
        # self.play(time.animate.set_value(1.00), rate_func=linear, run_time=.75)
        # self.wait(WAIT)
        # self.play(FadeOut(label_X,formula_X,label_x,formula_x))
        # self.wait(WAIT)

        # NO DISPLACEMENT IN THIS VIDEO
        # self.play(Create(arrow_u_temp))
        # self.play(Write(label_u_temp))
        # self.wait(WAIT)
        # self.add(arrow_u,label_u)
        # self.remove(arrow_u_temp,label_u_temp)
        # self.play(time.animate.set_value(0.00), rate_func=linear, run_time=1.5)
        # self.wait(WAIT)
        # self.play(time.animate.set_value(1.00), rate_func=linear, run_time=1.5)
        # self.wait(WAIT)
        # self.play(time.animate.set_value(0.00), rate_func=linear, run_time=1.5)
        # self.wait(WAIT)
        # self.play(FadeOut(label_X,label_x,label_u))
        # self.wait(WAIT)

