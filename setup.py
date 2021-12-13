from pathlib import Path

from setuptools import find_packages, setup

# Get __version__ from rl_pong/version.py
exec(
    compile(
        open("rl_pong/version.py").read(),
        "rl_pong/version.py",
        "exec",
    )
)

setup(
    name="RL-Pong",
    version=__version__,
    author="Leon Shams-Schaal",
    description="Train an reinforcement learning agent to play the game Pong.",
    long_description=Path("readme.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/WLAC-CS131-Fall-2021/LeonShams",
    project_urls={
        "Bug Tracker": "https://github.com/WLAC-CS131-Fall-2021/LeonShams/issues",
    },
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=Path("requirements.txt").read_text().splitlines(),
    entry_points={"console_scripts": ["rl-pong=rl_pong.__main__:main"]},
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="pong dqn reinforcement learning rl",
)
