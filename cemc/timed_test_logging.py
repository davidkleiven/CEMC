"""
Convenient classes for unittest that also prints the timing

Based on:
https://hackernoon.com/timing-tests-in-python-for-fun-and-profit-1663144571
"""
import time
from unittest.runner import TextTestResult
import unittest


class TimeLoggingTestResult(TextTestResult):
    def __init__(self, *args, **kwargs):
        super(TimeLoggingTestResult, self).__init__(*args, **kwargs)
        self.test_timings = []

    def startTest(self, test):
        self._test_started_at = time.time()
        super(TimeLoggingTestResult, self).startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = self.getDescription(test)
        self.test_timings.append((elapsed, name))
        super(TimeLoggingTestResult, self).addSuccess(test)

    def getTestTimings(self):
        return self.test_timings


class TimeLoggingTestRunner(unittest.TextTestRunner):
    def __init__(self, *args, **kwargs):
        return super(TimeLoggingTestRunner, self).__init__(
            resultclass=TimeLoggingTestResult)

    def run(self, test):
        result = super(TimeLoggingTestRunner, self).run(test)
        self.stream.writeln("\nTiming results")
        test_timings = result.getTestTimings()
        test_timings.sort(reverse=True)

        for elapsed, name in test_timings:
            self.stream.writeln("{:<90} \t ({:>5}s)".format(
                name, round(elapsed, 2)))
        return result
