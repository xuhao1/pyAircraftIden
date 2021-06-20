from setuptools import setup, find_packages

setup(
    name="pyAircraftIden",
    version="1.0",
    author="xuhao3e8",
    author_email="xuhao3e8@gmail.com",
    description="An open-source aircraft identification library in python",

    url="https://www.xuhao1.me/open-source/", 

    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages(exclude=("data",)),
        install_requires=['scipy', "numpy", "matplotlib", "control", "sympy"],

)