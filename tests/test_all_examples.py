import unittest
import importlib
import os
import sys
try:
    from ase.ce import BulkCrystal
    has_CE = True
except Exception as exc:
    print (exc)
    has_CE = False

def test_generator(fname):
    def run_example( self):
        if ( not has_CE ):
            testcase.skipTest("ASE version needs CE to run the examples")
            return
        no_throw = True
        msg = ""
        try:
            mod = importlib.import_module(fname)
        except Exception as exc:
            msg = str(exc)
            no_throw = False
        self.assertTrue( no_throw, msg=msg )
    return run_example

class TestSequence( unittest.TestCase ):
    pass

# Get all files in the example folder
all_files = os.listdir("examples")
sys.path.append("examples")

example_files = [fname for fname in all_files if fname.startswith("ex") and fname.endswith(".py")]
for i in range(len(example_files)):
    modname = example_files[i].split(".")[0]
    testmethodname = "test_"+modname
    test_method = test_generator(modname)
    setattr(TestSequence, testmethodname, test_method)

if __name__ == "__main__":
    unittest.main()
