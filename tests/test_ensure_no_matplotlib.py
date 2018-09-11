import unittest
import sys

try:
    from cemc import *
    from cemc.mcmc import *
    from cemc.tools import *
    from cemc.wanglandau import *
    import_msg = ""
    available = True
except ImportError as exc:
    import_msg = str(exc)
    available = False


class TestNoMatplotlib(unittest.TestCase):
    def _imported(self, mods, name):
        for m in mods:
            if m.find(name) != -1:
                return True
        return False

    def test_no_matplotlib(self):
        if not available:
            self.skipTest(import_msg)

        not_allowed_modules = ["matplotlib", "pyplot", "tkinter", "Tkinter"]
        import_mod = list(sys.modules.keys())
        for mod in not_allowed_modules:
            self.assertFalse(self._imported(import_mod, mod))


if __name__ == "__main__":
    from cemc import TimeLoggingTestRunner
    unittest.main(testRunner=TimeLoggingTestRunner)
