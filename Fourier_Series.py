from manim import *
import numpy as np

# Clase base para Fourier
class FourierSceneAbstract(ZoomedScene):
    def __init__(self):
        super().__init__()
        self.fourier_symbol_config = {
            "stroke_width": 1.3,
            "fill_opacity": 1,
            "height": 4,
        }
        self.vector_config = {
            "buff": 0,
            "max_tip_length_to_length_ratio": 0.25,
            "tip_length": 0.15,
            "max_stroke_width_to_length_ratio": 10,
            "stroke_width": 1.0,
        }
        self.circle_config = {
            "stroke_width": 0.8,
            "stroke_opacity": 0.2,
            "color": WHITE,
        }
        self.n_vectors = 150  # Más vectores para mayor precisión
        self.cycle_seconds = 15
        self.parametric_func_step = 0.001
        self.drawn_path_stroke_width = 5
        self.drawn_path_interpolation_config = [0, 1]
        self.path_n_samples = 1000
        self.freqs = list(range(-self.n_vectors // 2, self.n_vectors // 2 + 1, 1))
        self.freqs.sort(key=abs)

    def setup(self):
        super().setup()
        self.vector_clock = ValueTracker()
        self.slow_factor_tracker = ValueTracker(0)
        self.add(self.vector_clock)

    def start_vector_clock(self):
        self.vector_clock.add_updater(
            lambda t, dt: t.increment_value(
                dt * self.slow_factor_tracker.get_value() / self.cycle_seconds
            )
        )

    def stop_vector_clock(self):
        self.vector_clock.remove_updater(self.start_vector_clock)

    def get_fourier_coefs(self, path):
        dt = 1 / self.path_n_samples
        t_range = np.arange(0, 1, dt)

        points = np.array([path.point_from_proportion(t) for t in t_range])
        complex_points = points[:, 0] + 1j * points[:, 1]

        coefficients = [
            np.sum(
                np.array(
                    [
                        c_point * np.exp(-TAU * 1j * freq * t) * dt
                        for t, c_point in zip(t_range, complex_points)
                    ]
                )
            )
            for freq in self.freqs
        ]
        return coefficients

    def get_fourier_vectors(self, path):
        coefficients = self.get_fourier_coefs(path)

        vectors = VGroup()
        v_is_first_vector = True
        for coef, freq in zip(coefficients, self.freqs):
            v = Vector([np.real(coef), np.imag(coef)], **self.vector_config)
            if v_is_first_vector:
                center_func = VectorizedPoint(ORIGIN).get_location
                v_is_first_vector = False
            else:
                center_func = last_v.get_end
            v.center_func = center_func
            last_v = v
            v.freq = freq
            v.coef = coef
            v.phase = np.angle(coef)
            v.shift(v.center_func() - v.get_start())
            v.set_angle(v.phase)
            vectors.add(v)
        return vectors

    def update_vectors(self, vectors):
        for v in vectors:
            time = self.vector_clock.get_value()
            v.shift(v.center_func() - v.get_start())
            v.set_angle(v.phase + time * v.freq * TAU)

    def get_circles(self, vectors):
        circles = VGroup()
        for v in vectors:
            c = Circle(radius=v.get_length(), **self.circle_config)
            c.center_func = v.get_start
            c.move_to(c.center_func())
            circles.add(c)
        return circles

    def update_circles(self, circles):
        for c in circles:
            c.move_to(c.center_func())

    def get_drawn_path(self, vectors):
        def fourier_series_func(t):
            fss = np.sum(
                np.array([v.coef * np.exp(TAU * 1j * v.freq * t) for v in vectors])
            )
            real_fss = np.array([np.real(fss), np.imag(fss), 0])
            return real_fss

        t_range = np.array([0, 1, self.parametric_func_step])
        vector_sum_path = ParametricFunction(
            fourier_series_func, t_range=t_range
        )
        broken_path = CurvesAsSubmobjects(vector_sum_path)
        broken_path.stroke_width = 0
        broken_path.start_width = self.drawn_path_interpolation_config[0]
        broken_path.end_width = self.drawn_path_interpolation_config[1]
        return broken_path

    def update_path(self, broken_path):
        alpha = self.vector_clock.get_value()
        n_curves = len(broken_path)
        alpha_range = np.linspace(0, 1, n_curves)
        for a, subpath in zip(alpha_range, broken_path):
            b = alpha - a
            if b < 0:
                width = 0
            else:
                width = self.drawn_path_stroke_width * interpolate(
                    broken_path.start_width,
                    broken_path.end_width,
                    1 - (b % 1),
                )
            subpath.set_stroke(width=width)


# Clase para dibujar LATEX
class FourierLATEX(FourierSceneAbstract):
    def construct(self):
        symbol = MathTex(r"\pi", font_size=144).set_color(RED)

        vectors = self.get_fourier_vectors(symbol.family_members_with_points()[0])
        circles = self.get_circles(vectors)
        drawn_path = self.get_drawn_path(vectors).set_color(RED)

        self.add(vectors, circles, drawn_path)
        self.start_vector_clock()
        vectors.add_updater(self.update_vectors)
        circles.add_updater(self.update_circles)
        drawn_path.add_updater(self.update_path)

        self.play(self.slow_factor_tracker.animate.set_value(1), run_time=15)
        self.wait(5)


# Clase para dibujar "TEXTO"
class FourierText(FourierSceneAbstract):
    def construct(self):
        text = Tex("Mery", font_size=144).set_color(BLUE)

        paths = text.family_members_with_points()
        all_vectors = VGroup()
        all_circles = VGroup()
        all_drawn_paths = VGroup()

        for path in paths:
            vectors = self.get_fourier_vectors(path)
            circles = self.get_circles(vectors)
            drawn_path = self.get_drawn_path(vectors).set_color(BLUE)

            all_vectors.add(vectors)
            all_circles.add(circles)
            all_drawn_paths.add(drawn_path)

            vectors.add_updater(self.update_vectors)
            circles.add_updater(self.update_circles)
            drawn_path.add_updater(self.update_path)

        self.add(all_vectors, all_circles, all_drawn_paths)
        self.start_vector_clock()
        self.play(self.slow_factor_tracker.animate.set_value(1), run_time=25)
        self.wait(5)


# Clase para dibujar un SVG
class FourierSVG(FourierSceneAbstract):
    def construct(self):
        svg_path = "assets/CIS.svg"  # Cambia la ruta al archivo SVG
        svg = SVGMobject(svg_path).set_color(BLUE).set_height(4)

        paths = svg.family_members_with_points()
        all_vectors = VGroup()
        all_circles = VGroup()
        all_drawn_paths = VGroup()

        for path in paths:
            vectors = self.get_fourier_vectors(path)
            circles = self.get_circles(vectors)
            drawn_path = self.get_drawn_path(vectors).set_color(BLUE)

            all_vectors.add(vectors)
            all_circles.add(circles)
            all_drawn_paths.add(drawn_path)

            vectors.add_updater(self.update_vectors)
            circles.add_updater(self.update_circles)
            drawn_path.add_updater(self.update_path)

        self.add(all_vectors, all_circles, all_drawn_paths)
        self.start_vector_clock()
        self.play(self.slow_factor_tracker.animate.set_value(1), run_time=25)
        self.wait(5)