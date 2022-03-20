import subprocess
import pytest
import _pytest.skipping


def pytest_configure(config):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")
    subprocess.call(["make"], shell=True, cwd="tests")


def pytest_sessionfinish(session, exitstatus):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")

def pytest_addoption(parser):
    parser.addoption(
        "--no-skips",
        action="store_true",
        default=False, help="disable skip marks")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):
    if "--no-skips" not in args:
        return

    def no_skip(*args, **kwargs):
        return

    _pytest.skipping.skip = no_skip