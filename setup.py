from setuptools import setup

setup(name='nettack',
      version='0.1',
      description='Adversarial Attacks on Neural Networks for Graph Data',
      author='Daniel Zügner, Amir Akbarnejad, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de',
      packages=['nettack'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'tensorflow', 'numba'],
zip_safe=False)