import subprocess


def pytest_configure(config):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")
    subprocess.call(["make"], shell=True, cwd="tests")


def pytest_sessionfinish(session, exitstatus):
    subprocess.call(["make", "clean"], shell=True, cwd="tests")
