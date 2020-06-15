from setuptools import setup

setup(name='synergyenvs',
      version="0.0.1",
      install_requires=[
          'pybullet>=1.7.8',
      ],
      package_data={'synergyenvs':[ "envs/assets/hand/*.xml"]}
)
