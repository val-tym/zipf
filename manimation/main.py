from manim import *
import os
from pathlib import Path
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService


class ZipfPresentation(VoiceoverScene):
    def construct(self):
        # -------------------------------
        # Налаштування голосу
        # -------------------------------
        VOICEOVER_DIR = Path(__file__).resolve().parent / "media" / "voiceovers"
        self.set_speech_service(GTTSService(lang="uk", cache_dir=VOICEOVER_DIR))

        # -------------------------------
        # Візуальна стилізація сцени
        # -------------------------------
        BASE_DIR = os.path.dirname(__file__)
        RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results_2026-05-13_17-57-26"))

        def asset(path: str) -> str:
            return os.path.join(BASE_DIR, path)

        def result(path: str) -> str:
            return os.path.join(RESULTS_DIR, path)

        BG_COLOR = "#0B1020"
        TEXT_COLOR = WHITE
        MATH_COLOR = "#deff9a"
        ACCENT_COLOR = "#11caa0"
        ACCENT_COLOR_2 = "#22C55E"
        YELLOW_ACCENT = "#deff9a"

        self.camera.background_color = BG_COLOR

        # Делікатні "glow" плями та рамка для глибини кадру
        glow_left = Circle(radius=6, stroke_width=0).set_fill(ACCENT_COLOR, opacity=0.12)
        glow_left.move_to(LEFT * 5 + UP * 2)
        glow_right = Circle(radius=6, stroke_width=0).set_fill(ACCENT_COLOR_2, opacity=0.10)
        glow_right.move_to(RIGHT * 5 + DOWN * 1.5)
        border = Rectangle(width=config.frame_width, height=config.frame_height).set_stroke(color=WHITE, opacity=0.08, width=2).set_fill(opacity=0)
        self.add(glow_left, glow_right, border)

        def styled_text(text: str, font_size: int, color: str = TEXT_COLOR):
            m = Text(text, font_size=font_size, color=color)
            m.set_stroke(color=BLACK, width=1.5, opacity=0.35)
            return m

        def with_card(img_mobject, pad: float = 0.25, color: str = ACCENT_COLOR):
            # "Картка" навколо зображення додає контраст і цілісність стилю
            rect = RoundedRectangle(
                corner_radius=0.25,
                width=img_mobject.width + pad * 2,
                height=img_mobject.height + pad * 2,
            )
            rect.set_stroke(color=color, width=2, opacity=0.55)
            rect.set_fill(color=color, opacity=0.06)
            rect.move_to(img_mobject)
            # ImageMobject is not a VMobject, so we must use Group (not VGroup).
            return Group(rect, img_mobject)

        title = Text("Закон Ципфа:", color=ACCENT_COLOR, font_size=72)
        subtitle = Text("Фундаментальний закон статистики", font_size=36).next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(subtitle))

        heading = Text("У чому суть?", color=ACCENT_COLOR).to_edge(UP)
        self.play(Write(heading))

        desc = Text(
            "Частота появи слова у тексті\nобернено пропорційна його рангу.",
            font_size=32, line_spacing=1.5
        )
        self.play(FadeIn(desc))
        self.wait(2)

        example = MarkupText(
            '1-ше місце: <span color="#deff9a">100%</span>\n'
            '2-ге місце: <span color="#deff9a">50%</span>\n'
            '3-тє місце: <span color="#deff9a">33%</span>',
            font_size=36, line_spacing=1.8
        ).shift(DOWN * 0.5)

        self.play(desc.animate.shift(UP * 1.5).scale(0.8))
        self.play(Write(example))
        self.wait(2)
        self.play(FadeOut(heading), FadeOut(desc), FadeOut(example))

        title_formula = Text("Математична модель", color=ACCENT_COLOR).to_edge(UP)
        self.play(Write(title_formula))

        formula = MarkupText(
            "<i>f</i>(<i>r</i>) = C / <i>r</i><sup><i>s</i></sup>",
            font_size=80, color=YELLOW_ACCENT, font="Times New Roman"
        )

        formula_labels = VGroup(
            Text("f — частота слова", font_size=24),
            Text("r — ранг (місце у списку за частотністю)", font_size=24),
            Text("s — показник степеня (зазвичай ≈ 1)", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).shift(DOWN * 2)

        self.play(Write(formula))
        self.play(FadeIn(formula_labels))
        self.wait(5)
        self.play(FadeOut(title_formula), FadeOut(formula), FadeOut(formula_labels))

        title_formula = Text("Закон Ципфа-Мандельброта", color=ACCENT_COLOR).to_edge(UP)
        self.play(Write(title_formula))

        formula = MarkupText(
            "<i>f</i>(<i>r</i>) = C / (<i>r</i> + <i>q</i>)<sup><i>s</i></sup>",
            font_size=80, color=YELLOW_ACCENT, font="Times New Roman"
        )

        formula_labels = VGroup(
            Text("f — частота слова", font_size=24),
            Text("r — ранг (місце у списку за частотністю)", font_size=24),
            Text("s — показник степеня (зазвичай ≈ 1)", font_size=24),
            Text("q — додатковий параметр", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).shift(DOWN * 2)

        self.play(Write(formula))
        self.play(FadeIn(formula_labels))
        self.wait(8)
        self.play(FadeOut(title_formula), FadeOut(formula), FadeOut(formula_labels))

        title_why = Text("Навіщо це досліджувати?", color=ACCENT_COLOR).to_edge(UP)
        self.play(Write(title_why))

        reasons = VGroup(
            Text("• Оптимізація пошукових систем та стиснення даних", font_size=28),
            Text("• Розуміння еволюції людської мови", font_size=28),
            Text("• Аналіз складних самоорганізованих систем", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.6)

        self.play(FadeIn(reasons, shift=RIGHT))
        self.wait(5.5)
        self.play(FadeOut(title_why), FadeOut(reasons))

        final_text = Text(
            "Закон Ципфа працює всюди:\nвід частоти слів до населення міст.",
            font_size=36, color=YELLOW_ACCENT, line_spacing=1.5
        )
        self.play(Write(final_text))
        self.wait(8)
        self.play(FadeOut(final_text))

        # =========================
        # Заголовок
        # =========================
        title = styled_text("Закон Ципфа для різних мов", font_size=64, color=ACCENT_COLOR).shift(UP * 3)

        with self.voiceover(
            text="Для прикладу візьмемо англійську мову."
        ) as tracker:
            self.play(Write(title))

        self.play(FadeOut(title))

        # ================================================

        pg = with_card(ImageMobject(asset("pg-logo.jpg")), pad=0.18).to_edge(LEFT)
        ws = with_card(ImageMobject(asset("wikisource.png")), pad=0.18).scale(0.35)
        ia = with_card(ImageMobject(asset("internet_archive.png")), pad=0.18).scale(0.37).to_edge(RIGHT)

        with self.voiceover(
                text="Для дослідження використаємо відкриті електронні бібліотеки: проект Гутенберг, Вікіджерела та Internet Archive. Це також дозволяє додати українські тексти."
        ) as tracker:
            self.play(FadeIn(pg,ws,ia))

        self.play(FadeOut(pg,ws,ia))

        # ================================================

        book1 = with_card(ImageMobject(asset("book.png")).shift(DOWN).scale(0.5), pad=0.10)

        self.play(FadeIn(book1))
        self.wait(13)

        # ================================================

        book2 = with_card(ImageMobject(asset("book.png")).shift(UP*1.5).scale(0.5), pad=0.10)

        self.play(FadeIn(book2))
        self.wait(3.5)

        self.play(FadeOut(book1, book2))

        # ================================================

        en_top = Text("the             13495\n\
of                7920\n\
and             6547\n\
to                6440\n\
that            4050\n\
in                3486\n\
is                 3012\n\
they           2411\n\
not             2406\n\
by               2338\n\
a                   2281\n\
which       2184\n\
it                 2158\n\
he                2086\n\
for              2008\n\
be                1997\n\
but             1787\n\
are               1713\n\
as                 1689\n\
this            1534\n\
...", font_size=20)

        self.play(Write(en_top))
        self.wait(4)

        self.play(FadeOut(en_top))

        # ================================================

        en_raw_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT*4+UP * 3)
        en_raw_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT*4+UP * 3)

        en_raw_zipf = with_card(ImageMobject(result("en_raw_zipf.png")).shift(4*LEFT).scale(1.3), pad=0.18)
        en_raw_zipf_mandelbrot = with_card(ImageMobject(result("en_raw_zipf_mandelbrot.png")).shift(4*RIGHT).scale(1.3), pad=0.18)

        self.play(FadeIn(en_raw_zipf_title, en_raw_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot))
        self.wait(13)

        self.play(FadeOut(en_raw_zipf_title, en_raw_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot))

        # ===============================================

        det_coef_1_title = styled_text("Коефіцієнт детермінації визначається наступним чином:", font_size=25).to_edge(UP)
        det_coef_1_formula = MathTex(r"R^2 = 1 - \frac{V(y|x)}{V(y)} = 1 - \frac{\sigma^2}{\sigma_y^2}")
        det_coef_1_text1 = styled_text("де V(y) — дисперсія випадкової величини y,", font_size=25)
        det_coef_1_formula1 = MathTex(r"V(y) = \sigma_y^2")
        det_coef_1_group1 = VGroup(det_coef_1_text1, det_coef_1_formula1).arrange(RIGHT, buff=0.5)
        det_coef_1_text2 = styled_text("V(y|x) — умовна дисперсія (дисперсія похибки моделі).", font_size=25)
        det_coef_1_formula2 = MathTex(r"V(y|x) = \sigma^2")
        det_coef_1_group2 = VGroup(det_coef_1_text2, det_coef_1_formula2).arrange(RIGHT, buff=0.5)
        det_coef_1_explanations = VGroup(det_coef_1_group1, det_coef_1_group2).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        det_coef_1_explanations.next_to(det_coef_1_formula, DOWN, buff=1)


        self.play(Write(det_coef_1_title))
        self.play(Write(det_coef_1_formula))
        self.play(Write(det_coef_1_explanations))

        self.play(FadeOut(det_coef_1_title, det_coef_1_formula, det_coef_1_explanations))

        # ================================================

        det_coef_2_title = styled_text("Для розрахунку вибіркового коефіцієнта детермінації\nвикористовують вибіркові оцінки дисперсій:", font_size=25).to_edge(UP)
        det_coef_2_formula = MathTex(r"R^2 = 1 - \frac{\hat{\sigma}^2}{\hat{\sigma}_y^2}= 1 - \frac{RSS/n}{TSS/n}= 1 - \frac{RSS}{TSS},", font_size=25)
        det_coef_2_rss_text = styled_text("де RSS — сума квадратів залишків регресії:", font_size=25)
        det_coef_2_rss_formula = MathTex(r"RSS = \sum_{t=1}^{n} e_t^2 = \sum_{t=1}^{n} (y_t - \hat{y}_t)^2,",font_size=25)
        det_coef_2_rss_desc = styled_text("y_t, ŷ_t — фактичні та оціночні значення змінної.", font_size=25)
        det_coef_2_tss_text = styled_text("TSS — загальна сума квадратів:", font_size=25)
        det_coef_2_tss_formula = MathTex(r"TSS = \sum_{t=1}^{n} (y_t - \overline{y})^2 = n \hat{\sigma}_y^2",font_size=25)
        det_coef_2_explanations = VGroup(det_coef_2_title, det_coef_2_formula, det_coef_2_rss_text, det_coef_2_rss_formula, det_coef_2_rss_desc, det_coef_2_tss_text, det_coef_2_tss_formula).arrange(DOWN, aligned_edge=LEFT)

        self.play(Write(det_coef_2_explanations))
        self.wait(3)

        self.play(FadeOut(det_coef_2_explanations))

        # ================================================

        en_raw_zipf_title_for_det_coef = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 4 + UP)
        en_raw_zipf_mandelbrot_title_for_det_coef = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP)

        en_raw_zipf_for_det_coef = MathTex(r"R^2 = 0.9803", font_size=25).shift(LEFT * 4 + DOWN)
        en_raw_zipf_mandelbrot_for_det_coef = MathTex(r"R^2 = 0.9827", font_size=25).shift(RIGHT * 4 + DOWN)


        self.play(Write(en_raw_zipf_title_for_det_coef))
        self.play(Write(en_raw_zipf_mandelbrot_title_for_det_coef))
        self.play(Write(en_raw_zipf_for_det_coef))
        self.play(Write(en_raw_zipf_mandelbrot_for_det_coef))
        self.wait(1.5)

        self.play(FadeOut(en_raw_zipf_title_for_det_coef, en_raw_zipf_mandelbrot_title_for_det_coef, en_raw_zipf_for_det_coef, en_raw_zipf_mandelbrot_for_det_coef))

        # ===============================================

        lemmatization = with_card(ImageMobject(asset("lemmatization.png")).shift(2*DOWN).scale(2), pad=0.25)
        stanza = with_card(ImageMobject(asset("stanza-logo.png")).shift(UP).scale(2), pad=0.25)


        self.play(FadeIn(lemmatization, stanza))
        self.wait(14.5)

        self.play(FadeOut(lemmatization, stanza))

        # ================================================

        en_lemma_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 4 + UP * 3)
        en_lemma_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP * 3)

        en_lemma_zipf = with_card(ImageMobject(result("en_lemma_zipf.png")).shift(4 * LEFT).scale(1.3), pad=0.18)
        en_lemma_zipf_mandelbrot = with_card(
            ImageMobject(result("en_lemma_zipf_mandelbrot.png")).shift(4 * RIGHT).scale(1.3),
            pad=0.18,
        )


        self.play(FadeIn(en_lemma_zipf_title, en_lemma_zipf_mandelbrot_title, en_lemma_zipf, en_lemma_zipf_mandelbrot))
        self.wait(4)

        self.play(FadeOut(en_lemma_zipf_title, en_lemma_zipf_mandelbrot_title, en_lemma_zipf, en_lemma_zipf_mandelbrot))

        # ================================================

        en_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 2 + UP * 3.7)
        en_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP * 3.7)

        en_raw_zipf = with_card(ImageMobject(result("en_raw_zipf.png")).shift(2 * LEFT+UP*1.7), pad=0.16)
        en_raw_zipf_mandelbrot = with_card(ImageMobject(result("en_raw_zipf_mandelbrot.png")).shift(
            4 * RIGHT+UP*1.7), pad=0.16)

        en_lemma_zipf = with_card(ImageMobject(result("en_lemma_zipf.png")).shift(2 * LEFT+DOWN*2.1), pad=0.16)
        en_lemma_zipf_mandelbrot = with_card(ImageMobject(result("en_lemma_zipf_mandelbrot.png")).shift(
            4 * RIGHT+DOWN*2.1), pad=0.16)

        raw = styled_text("Без лематизації", font_size=20).shift(UP * 1.7).to_edge(LEFT)
        lemma = styled_text("З лематизацією", font_size=20).shift(DOWN*2.1).to_edge(LEFT)

        with self.voiceover(
                text="Зручно порівняти всі 4 графіки."
        ) as tracker:
            self.play(FadeIn(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot, en_lemma_zipf, en_lemma_zipf_mandelbrot))

        self.play(FadeOut(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot, en_lemma_zipf, en_lemma_zipf_mandelbrot))

        # ================================================

        en_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 2 + UP * 3.7)
        en_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP * 3.7)

        en_raw_zipf = styled_text("0.9803", font_size=30).shift(2 * LEFT + UP * 1.7)
        en_raw_zipf_mandelbrot = styled_text("0.9827", font_size=30).shift(4 * RIGHT + UP * 1.7)

        en_lemma_zipf = styled_text("0.9784", font_size=30).shift(2 * LEFT + DOWN * 2.1)
        en_lemma_zipf_mandelbrot = styled_text("0.9828", font_size=30).shift(4 * RIGHT + DOWN * 2.1)

        raw = styled_text("Без лематизації", font_size=20).shift(UP * 1.7).to_edge(LEFT)
        lemma = styled_text("З лематизацією", font_size=20).shift(DOWN * 2.1).to_edge(LEFT)


        self.play(FadeIn(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                             en_lemma_zipf, en_lemma_zipf_mandelbrot))
        self.wait(10)

        self.play(FadeOut(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                          en_lemma_zipf, en_lemma_zipf_mandelbrot))

        # ================================================

        zipf_title = styled_text("s (закон Ципфа)", font_size=14).shift(LEFT * 3.2 + UP * 3.7)
        mand_s_title = styled_text("s (закон Ципфа-Мандельброта)", font_size=14).shift(RIGHT * 0.9 + UP * 3.7)
        mand_q_title = styled_text("q (закон Ципфа-Мандельброта)", font_size=14).shift(RIGHT * 5 + UP * 3.7)

        raw_s = with_card(ImageMobject(asset("raw_s.png")).shift(3.2 * LEFT + UP * 1.7).scale(0.6), pad=0.10)
        raw_mand_s = with_card(ImageMobject(asset("raw_mand_s.png")).shift(0.9 * RIGHT + UP * 1.7).scale(0.6), pad=0.10)
        raw_mand_q = with_card(ImageMobject(asset("raw_mand_q.png")).shift(5 * RIGHT + UP * 1.7).scale(0.6), pad=0.10)

        lemma_s = with_card(ImageMobject(asset("lemma_s.png")).shift(3.2 * LEFT + DOWN * 2.1).scale(0.6), pad=0.10)
        lemma_mand_s = with_card(ImageMobject(asset("lemma_mand_s.png")).shift(0.9 * RIGHT + DOWN * 2.1).scale(0.6), pad=0.10)
        lemma_mand_q = with_card(ImageMobject(asset("lemma_mand_q.png")).shift(5 * RIGHT + DOWN * 2.1).scale(0.6), pad=0.10)

        raw = styled_text("Без лематизації", font_size=14).shift(UP * 1.7+LEFT*6)
        lemma = styled_text("З лематизацією", font_size=14).shift(DOWN * 2.1+LEFT*6)

        with self.voiceover(
                text="Нанесемо на карту значення апроксимованих коефіцієнтів: s для закону Ципфа, s та q для закону Ципфа-Мандельброта."
        ) as tracker:
            self.play(FadeIn(zipf_title, mand_s_title, mand_q_title, raw_s, raw_mand_s, raw_mand_q, lemma_s, lemma_mand_s, lemma_mand_q, raw, lemma))

        self.play(FadeOut(zipf_title, mand_s_title, mand_q_title, raw_s, raw_mand_s, raw_mand_q, lemma_s, lemma_mand_s, lemma_mand_q, raw, lemma))

        # ================================================

        en_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 2 + UP * 3.7)
        en_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP * 3.7)

        en_raw_zipf = with_card(ImageMobject(asset("raw_det.png")).shift(2 * LEFT + UP * 1.7).scale(0.7), pad=0.12)
        en_raw_zipf_mandelbrot = with_card(ImageMobject(asset("raw_mand_det.png")).shift(
            4 * RIGHT + UP * 1.7).scale(0.7), pad=0.12)

        en_lemma_zipf = with_card(ImageMobject(asset("lemma_det.png")).shift(2 * LEFT + DOWN * 2.1).scale(0.7), pad=0.12)
        en_lemma_zipf_mandelbrot = with_card(ImageMobject(asset("lemma_mand_det.png")).shift(
            4 * RIGHT + DOWN * 2.1).scale(0.7), pad=0.12)

        raw = styled_text("Без лематизації", font_size=20).shift(UP * 1.7).to_edge(LEFT)
        lemma = styled_text("З лематизацією", font_size=20).shift(DOWN * 2.1).to_edge(LEFT)


        self.play(FadeIn(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                             en_lemma_zipf, en_lemma_zipf_mandelbrot))
        self.wait(10)

        self.play(FadeOut(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                          en_lemma_zipf, en_lemma_zipf_mandelbrot))


        title_1 = Text("Математичне обґрунтування похибки", color=ACCENT_COLOR).to_edge(UP)
        self.play(FadeIn(title_1))

        statement = Text(
            "Відхилення від теоретичного розподілу Ципфа\nє наслідком алгоритмічної похибки NLP-моделей.",
            font_size=32,
            line_spacing=1.5,
            color=TEXT_COLOR
        )
        self.play(Write(statement), run_time=2)
        self.wait(8)
        self.play(FadeOut(statement))

        title_2 = Text("Імовірнісна модель ідентифікації", color=ACCENT_COLOR).to_edge(UP)
        self.play(Transform(title_1, title_2))

        formula_group = VGroup()
        formula_intro = Text("Ймовірність коректної лематизації:", font_size=30).set_opacity(0.8)

        formula = MarkupText("<i>P</i><sub>acc</sub> = <i>p</i><sup>D</sup>", font_size=72, color=MATH_COLOR,
                             font="Times New Roman")

        definitions = VGroup(
            Text("p — ймовірність розпізнавання однієї ознаки", font_size=24),
            Text("D — вимірність вектора морфологічних ознак", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT).set_opacity(0.7)

        formula_group.add(formula_intro, formula, definitions).arrange(DOWN, buff=0.8)

        self.play(FadeIn(formula_intro))
        self.play(Write(formula))
        self.play(FadeIn(definitions))
        self.wait(4)
        self.play(FadeOut(formula_group))

        title_3 = Text("Порівняльний статистичний аналіз", color=ACCENT_COLOR).to_edge(UP)
        self.play(Transform(title_1, title_3))

        p_val = 0.95

        eng_group = VGroup()
        eng_name = Text("Аналітична мова (Англійська)", font_size=28, color=TEXT_COLOR)
        eng_dim = Text("D = 1", font_size=24).set_opacity(0.7)
        eng_math = MarkupText(f"<i>P</i><sub>eng</sub> = {p_val}<sup>1</sup> = 0.950", font_size=40, color=MATH_COLOR,
                              font="Times New Roman")
        eng_group.add(eng_name, eng_dim, eng_math).arrange(DOWN, buff=0.4).shift(LEFT * 3.5)

        lit_group = VGroup()
        lit_name = Text("Флективна мова (Литовська)", font_size=28, color=TEXT_COLOR)
        lit_dim = Text("D = 4", font_size=24).set_opacity(0.7)
        lit_math = MarkupText(f"<i>P</i><sub>lit</sub> = {p_val}<sup>4</sup> ≈ 0.814", font_size=40, color=MATH_COLOR,
                              font="Times New Roman")
        lit_group.add(lit_name, lit_dim, lit_math).arrange(DOWN, buff=0.4).shift(RIGHT * 3.5)

        divider = Line(UP * 1.5, DOWN * 1.5, color=GRAY).set_opacity(0.5)

        self.play(FadeIn(divider))
        self.play(FadeIn(eng_group, shift=RIGHT * 0.5), FadeIn(lit_group, shift=LEFT * 0.5))
        self.wait(4)

        sys_error = Text(
            "Систематична похибка класифікації: ~18.6%",
            font_size=30,
            color=ACCENT_COLOR
        ).to_edge(DOWN).shift(UP * 0.5)
        self.play(Write(sys_error))
        self.wait(3)
        self.play(FadeOut(eng_group), FadeOut(lit_group), FadeOut(divider), FadeOut(sys_error))

        title_4 = Text("Фундаментальність закону Ципфа", color=ACCENT_COLOR).to_edge(UP)
        self.play(Transform(title_1, title_4))

        final_conclusions = VGroup(
            Text("1. Штучне заниження частоти високорангових лексем.", font_size=28, color=TEXT_COLOR),
            Text("2. Гіпертрофоване збільшення дисперсії розподілу.", font_size=28, color=TEXT_COLOR),
            Text("Закон Ципфа є інваріантним. Аномалії — артефакт інструментів.", font_size=32, color=MATH_COLOR)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.8)

        final_conclusions.move_to(ORIGIN)

        for line in final_conclusions:
            self.play(FadeIn(line, shift=UP * 0.3), run_time=1.5)
            self.wait(1)

        self.wait(7)

        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=2
        )

        self.wait()
