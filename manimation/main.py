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
        RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results_2026-04-25_20-56-10"))

        def asset(path: str) -> str:
            return os.path.join(BASE_DIR, path)

        def result(path: str) -> str:
            return os.path.join(RESULTS_DIR, path)

        BG_COLOR = "#0B1020"
        TEXT_COLOR = "#EAF2FF"
        ACCENT_COLOR = "#7C5CFF"
        ACCENT_COLOR_2 = "#22C55E"

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

        # =========================
        # Заголовок
        # =========================
        title = styled_text("Закон Ципфа для різних мов", font_size=64).shift(UP * 3)

        with self.voiceover(
            text="Для прикладу візьмемо англійську мову."
        ) as tracker:
            self.play(Write(title))

        self.play(FadeOut(title))

        # ================================================

        pg = with_card(ImageMobject(asset("pg-logo.jpg")), pad=0.18)

        with self.voiceover(
                text="Для дослідження використаємо матеріали проекту Гутенберг - найстарішу універсальну електронну бібліотеку."
        ) as tracker:
            self.play(FadeIn(pg))

        self.play(FadeOut(pg))

        # ================================================

        book1 = with_card(ImageMobject(asset("book.png")).shift(DOWN).scale(0.5), pad=0.10)

        with self.voiceover(
                text="Для того щоб зібрати достатню кількість тексту для аналізу візьмемо випадкуву книгу на досліджуваній мові (наприклад ангійській). Якщо вона містить достатньо тексту (у нашому випадку достатньою вважалась кількість у 200000 токенів, тобто слів), то переходимо до наступного етапу."
        ) as tracker:
            self.play(FadeIn(book1))

        # ================================================

        book2 = with_card(ImageMobject(asset("book.png")).shift(UP*1.5).scale(0.5), pad=0.10)

        with self.voiceover(
                text="Якщо ж ні, то беремо наступні випадкові книги допоки не досягнемо необхідної кількості тексту."
        ) as tracker:
            self.play(FadeIn(book2))

        self.play(FadeOut(book1, book2))

        # ================================================

        en_top = Text("the             10282\n\
and             5434\n\
to              5174\n\
i               4908\n\
of              4749\n\
a               4414\n\
that            3094\n\
in              3025\n\
it              2993\n\
he              2944\n\
you             2858\n\
was             2704\n\
is              1897\n\
his             1892\n\
had             1654\n\
she             1646\n\
with            1479\n\
have            1463\n\
as              1437\n\
not             1382\n\
...", font_size=20)

        with self.voiceover(
                text="Далі впорядковуємо слова по частоті. Порядковий номер номер слова будемо називати його рангом."
        ) as tracker:
            self.play(Write(en_top))

        self.play(FadeOut(en_top))

        # ================================================

        en_raw_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT*4+UP * 3)
        en_raw_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT*4+UP * 3)

        en_raw_zipf = with_card(ImageMobject(result("en_raw_zipf.png")).shift(4*LEFT).scale(1.3), pad=0.18)
        en_raw_zipf_mandelbrot = with_card(ImageMobject(result("en_raw_zipf_mandelbrot.png")).shift(4*RIGHT).scale(1.3), pad=0.18)

        with self.voiceover(
                text="Будуємо графік залежності частоти слова від рангу у логарифмічному масштабі. Апроксимуємо за законом Ципфа та законом Ципфа-Мандельброта. Як бачимо апроксимація досить точно описує графік."
        ) as tracker:
            self.play(FadeIn(en_raw_zipf_title, en_raw_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot))

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

        with self.voiceover(
                text="Для оцінки точності апроксимації використаємо коефіцієнт детермінації. Його означення зараз перед вами."
        ) as tracker:
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

        with self.voiceover(
                text="Для вибіркового коефіцієнту детермінації використовують наступне означення."
        ) as tracker:
            self.play(Write(det_coef_2_explanations))

        self.play(FadeOut(det_coef_2_explanations))

        # ================================================

        en_raw_zipf_title_for_det_coef = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 4 + UP)
        en_raw_zipf_mandelbrot_title_for_det_coef = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP)

        en_raw_zipf_for_det_coef = MathTex(r"R^2 = 0.9810", font_size=25).shift(LEFT * 4 + DOWN)
        en_raw_zipf_mandelbrot_for_det_coef = MathTex(r"R^2 = 0.9831", font_size=25).shift(RIGHT * 4 + DOWN)

        with self.voiceover(
                text="У нашому дослідженні коефіцієнт детермінації досить блиський до 1, що означає високу точність апроксимації."
        ) as tracker:
            self.play(Write(en_raw_zipf_title_for_det_coef))
            self.play(Write(en_raw_zipf_mandelbrot_title_for_det_coef))
            self.play(Write(en_raw_zipf_for_det_coef))
            self.play(Write(en_raw_zipf_mandelbrot_for_det_coef))

        self.play(FadeOut(en_raw_zipf_title_for_det_coef, en_raw_zipf_mandelbrot_title_for_det_coef, en_raw_zipf_for_det_coef, en_raw_zipf_mandelbrot_for_det_coef))

        # ===============================================

        lemmatization = with_card(ImageMobject(asset("lemmatization.png")).shift(2*DOWN).scale(2), pad=0.25)
        stanza = with_card(ImageMobject(asset("stanza-logo.png")).shift(UP).scale(2), pad=0.25)

        with self.voiceover(
                text="У більшості мов одне і теж слово може мати декілька форм. Спробуємо дослідити закон Ципфа попередньо перевівши усі слова до початкової форми. Такий процес називається лематизацією. Для цього використаємо проєкт Stanza."
        ) as tracker:
            self.play(FadeIn(lemmatization, stanza))

        self.play(FadeOut(lemmatization, stanza))

        # ================================================

        en_lemma_zipf_title = styled_text("Закон Ципфа", font_size=24).shift(LEFT * 4 + UP * 3)
        en_lemma_zipf_mandelbrot_title = styled_text("Закон Ципфа-Мандельброта", font_size=24).shift(RIGHT * 4 + UP * 3)

        en_lemma_zipf = with_card(ImageMobject(result("en_lemma_zipf.png")).shift(4 * LEFT).scale(1.3), pad=0.18)
        en_lemma_zipf_mandelbrot = with_card(
            ImageMobject(result("en_lemma_zipf_mandelbrot.png")).shift(4 * RIGHT).scale(1.3),
            pad=0.18,
        )

        with self.voiceover(
                text="Тепер графік залежності частоти слова від рангу у логарифмічному масштабі набуває такого вигляду."
        ) as tracker:
            self.play(
                FadeIn(en_lemma_zipf_title, en_lemma_zipf_mandelbrot_title, en_lemma_zipf, en_lemma_zipf_mandelbrot))

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

        en_raw_zipf = styled_text("0.9810", font_size=30).shift(2 * LEFT + UP * 1.7)
        en_raw_zipf_mandelbrot = styled_text("0.9831", font_size=30).shift(4 * RIGHT + UP * 1.7)

        en_lemma_zipf = styled_text("0.9802", font_size=30).shift(2 * LEFT + DOWN * 2.1)
        en_lemma_zipf_mandelbrot = styled_text("0.9835", font_size=30).shift(4 * RIGHT + DOWN * 2.1)

        raw = styled_text("Без лематизації", font_size=20).shift(UP * 1.7).to_edge(LEFT)
        lemma = styled_text("З лематизацією", font_size=20).shift(DOWN * 2.1).to_edge(LEFT)

        with self.voiceover(
                text="Порівняємо також коефіцієнт детермінації. Як бачимо апроксимація за законом Ципфа-Мандельброта та лематизація справді збільшують коефіцієнт детермінації, тобто точність апроксимації."
        ) as tracker:
            self.play(FadeIn(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                             en_lemma_zipf, en_lemma_zipf_mandelbrot))

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

        with self.voiceover(
                text="Тепер нанесемо на мапу коефіцієнти детермінації. Як видно лематизація зазвичай справді збільшує коефіцієнт детермінації, проте є особливість в литовській мові, що потребує додаткового дослідження."
        ) as tracker:
            self.play(FadeIn(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                             en_lemma_zipf, en_lemma_zipf_mandelbrot))

        self.play(FadeOut(raw, lemma, en_zipf_title, en_zipf_mandelbrot_title, en_raw_zipf, en_raw_zipf_mandelbrot,
                          en_lemma_zipf, en_lemma_zipf_mandelbrot))

        self.wait()