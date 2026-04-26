from setuptools import find_packages, setup


setup(
    name="terrascout",
    version="0.1.0",
    description="A compact autonomy demo for a crop-inspection rover in GPS-degraded orchards.",
    packages=find_packages(include=["terrascout", "terrascout.*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.26", "matplotlib>=3.8"],
    extras_require={"dev": ["pytest>=8.0", "ruff>=0.5", "mypy>=1.10"]},
)
