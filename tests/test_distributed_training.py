import io
import re
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class DistributedTrainingExampleTest(unittest.TestCase):
    def setUp(self):
        chapter_path = Path(__file__).resolve().parents[1] / "distributed-training.qmd"
        chapter = chapter_path.read_text(encoding="utf-8")
        blocks = [
            block
            for block in re.findall(r"```python\n(.*?)```", chapter, re.DOTALL)
            if "def static_gb" in block
        ]
        self.assertEqual(len(blocks), 1)
        namespace = {}
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(blocks[0], str(chapter_path), "exec"), namespace)
        self.static_gb = namespace["static_gb"]
        self.example_output = stdout.getvalue()

    def test_qmd_example_reproduces_documented_accounts(self):
        self.assertEqual(self.example_output, "16.0\n2.0\n")
        self.assertEqual(self.static_gb(8, 1, 1, 0), 16.0)
        self.assertEqual(self.static_gb(2, 2, 2, 3), 2.0)

    def test_qmd_example_requires_positive_integer_parallel_degrees(self):
        configuration = {"dp": 2, "tp": 2, "pp": 2, "zero": 3}
        for field in ("dp", "tp", "pp"):
            for value in (-2, 0, 1.5, 2.0):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.static_gb(**{**configuration, field: value})


if __name__ == "__main__":
    unittest.main()
