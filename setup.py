from setuptools import setup
import io

with io.open('README.md', 'r', encoding='utf-8') as readme_file:
    readme = readme_file.read()

setup_args = {
    'name': 'pyspherical',
    'author': 'Adam E. Lanman',
    'url': 'https://github.com/aelanman/pyspherical',
    'download_url': 'https://github.com/aelanman/pyspherical/archive/v0.0.1.tar.gz',
    'license': 'MIT',
    'description': '',
    'long_description': readme,
    'long_description_content_type': 'text/markdown',
    'package_dir': {'pyspherical': 'pyspherical'},
    'packages': ['pyspherical', 'pyspherical.tests'],
    'version': '0.0.1',
    'include_package_data': True,
    'test_suite': 'pytest',
    'tests_require': ['pytest', 'sympy'],
    'setup_requires': ['pytest-runner'],
    'install_requires': ['numpy', 'numba', 'scipy'],
    'classifiers': ['Development Status :: 3 - Alpha',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 3.6',
                    'Topic :: Scientific/Engineering :: Physics'],
    'keywords': 'spin spherical harmonic transforms wigner'
}

if __name__ == '__main__':
    setup(**setup_args)
