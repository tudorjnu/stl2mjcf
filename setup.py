from setuptools import setup, find_packages


setup(
    name="stl2mjcf",
    version="0.1.dev1",
    author="Tudor Jianu",
    author_email="tudorjnu@gmail.com",
    packages=find_packages(
        where="src",
        include=["stl2mjcf"],
    ),
    setup_requires=[
        "setuptools==58.0.0",
    ],
    install_requires=[
        "mujoco",
        "pyaml",
        "trimesh",
        "rtree",
        "lxml",
        "mergedeep",
    ],
    entry_points={
        "console_scripts": [
            "stl2mjcf=stl2mjcf.stl2mjcf:main",
            # "stl2mjcf=stl2mjcf.stl2mjcf:test.py",
        ],
    },
)
