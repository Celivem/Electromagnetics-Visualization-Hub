"""Smoke test: every page of app.py renders, and every button and mode on it runs without an error.

Uses Streamlit's headless AppTest, so no browser is needed. From the repository root:

    python -m unittest discover -s tests -v

The 3D simulations run at their default resolution; the whole suite takes about half a minute.
"""
import logging
import unittest
import warnings
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 300

# (module in the navigation selectbox, label of the sub-navigation radio, option)
PAGES = [
    ("首頁", None, None),
    ("函數近似", "方法", "傅立葉近似"),
    ("函數近似", "方法", "勒讓德近似"),
    ("電位+電場模擬 (2D)", "結構", "笛卡爾"),
    ("電位+電場模擬 (2D)", "結構", "球座標"),
    ("電位+電場模擬 (2D)", "結構", "點電荷"),
    ("電位+電場模擬 (3D)", "結構", "笛卡爾"),
    ("電位+電場模擬 (3D)", "結構", "球座標"),
    ("電位+電場模擬 (3D)", "結構", "點電荷"),
]


def open_page(cat, label, sub, mode=None):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.selectbox[0].set_value(cat).run()
    if sub is not None:
        next(r for r in at.sidebar.radio if r.label == label).set_value(sub).run()
    if mode is not None:
        at.main.radio[mode[0]].set_value(mode[1]).run()
    return at


def problems(at):
    return [f"exception: {e.value}" for e in at.exception] + [f"st.error: {e.value}" for e in at.error]


class TestEveryPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)

    def test_pages_buttons_and_modes(self):
        for cat, label, sub in PAGES:
            name = cat if sub is None else f"{cat} / {sub}"
            at = open_page(cat, label, sub)
            with self.subTest(page=name):
                self.assertEqual(problems(at), [])
            for i in range(len(at.button)):
                at_b = open_page(cat, label, sub)
                button = at_b.button[i]
                button.click().run()
                with self.subTest(page=name, button=button.label):
                    self.assertEqual(problems(at_b), [])
            for j, radio in enumerate(at.main.radio):
                for option in radio.options:
                    at_m = open_page(cat, label, sub, (j, option))
                    with self.subTest(page=name, mode=option):
                        self.assertEqual(problems(at_m), [])
                    for k in range(len(at_m.main.button)):
                        at_mb = open_page(cat, label, sub, (j, option))
                        button = at_mb.main.button[k]
                        button.click().run()
                        with self.subTest(page=name, mode=option, button=button.label):
                            self.assertEqual(problems(at_mb), [])

    def test_number_input_defaults(self):
        """Defaults must be values, not minimums (a past bug passed the default as min_value)."""
        at = open_page("函數近似", "方法", "傅立葉近似")
        inputs = {n.label: (n.value, n.min) for n in at.number_input}
        self.assertEqual(inputs["起點 a"], (-3.1415, None))
        self.assertEqual(inputs["終點 b"], (3.1415, None))
        self.assertEqual(inputs["最大項數"][0], 50)
        at = open_page("電位+電場模擬 (2D)", "結構", "笛卡爾")
        self.assertEqual([n.value for n in at.number_input], [10.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
