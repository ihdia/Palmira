from setuptools import find_packages, setup
from pathlib import Path


def parse_requirements_line(line):
    """Special case for git requirements"""
    if line.startswith("git+http"):
        assert "@" in line, "Branch should be specified with suffix (ex: @master)"
        assert (
            "#egg=" in line
        ), "Package name should be specified with suffix (ex: #egg=kraken)"
        package_name = line.split("#egg=")[-1]
        return f"{package_name} @ {line}"
    else:
        return line


def parse_requirements():
    path = Path(__file__).parent.resolve() / "requirements.txt"
    assert path.exists(), f"Missing requirements: {path}"
    return list(
        map(parse_requirements_line, map(str.strip, path.read_text().splitlines()))
    )


setup(
    name="palmira",
    version=0.1,
    description="Palm Leaf Manuscript Region Annotator",
    author="Nishanth artham",
    author_email="arthamnishanth123@gmail.com",
    install_requires=parse_requirements(),
    # entry_points={"console_scripts": [f"{COMMAND}={MODULE}.worker:main"]},
    packages=['palmira'],
)
