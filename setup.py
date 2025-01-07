from setuptools import setup, find_packages

setup(
    name='pyro_util',
    package_dir={'': 'src'},
    version='0.0.1',
    packages=find_packages(where="src"),
    install_requires=[
        "numpyro",
        "jax",
        "xarray",
        "pandas", "numpy", "patsy"
    ]
)
