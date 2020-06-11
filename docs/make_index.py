# -*- coding: utf-8 -*-

"""
Format the readme.md file into the sphinx index.rst file.

"""
import codecs
import inspect
import os

import pypandoc
from datetime import datetime

def write_index_rst(readme_file=None, write_file=None):
    t = datetime.now().isoformat()
    out = ('.. pyspherical documentation index, created by\n'
           '   make_index.py on {date}\n\n').format(date=t)

    if readme_file is None:
        main_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.stack()[0][1])))
        readme_file = os.path.join(main_path, 'README.md')

    pypandoc.convert_file(readme_file, 'md')
    readme_text = pypandoc.convert_file(readme_file, 'rst')

    out += readme_text

    split_point = out.find('\nReferences')

    insert = (
            """
Further Documentation
---------------------

.. toctree::
   :maxdepth: 2

   tutorial
   functions
            """
    )

    out = out[:split_point] + insert + out[split_point:]

    out.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\xa0", " ")

    if write_file is None:
        write_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        write_file = os.path.join(write_path, 'index.rst')
    F = codecs.open(write_file, 'w', 'utf-8')
    F.write(out)
    print("wrote " + write_file)


if __name__ == '__main__':
    readme = '../README.md'
    write_index_rst(readme)
