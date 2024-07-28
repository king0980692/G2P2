import sys
import setuptools
from distutils.core import setup, Extension
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext

import numpy
extensions = [
        Extension(
                "pysmore.utils.*",
                ["./pysmore/utils/*.pyx"],
               language="c++",             # generate C++ code
               extra_compile_args=["-std=c++11"],
               libraries=['stdc++'],

            ),
]
setuptools.setup(
    name="pysmore",
    version="0.0.2-dev",
    author="Leon, Chang",
    author_email="king0980692@gmail.com",
    description=("An pytorch version of SMORe"),
    long_description_content_type="text/markdown",
    url="https://github.com/cnclabs/pysmore",
    project_urls={
        "Bug Tracker": "https://github.com/cnclabs/pysmore/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],

    cmdclass={'build_ext':build_ext},
    ext_modules=cythonize(extensions, language_level=3),


    install_requires=["tqdm", "torch", "numpy", "pandas","cython", 'torchtext', 'torchinfo'],

    extras_require={
        'onnx':['onnxruntime', 'onnx']
    },

    packages=setuptools.find_packages(),
    python_requires=">=3.6",

    entry_points={
        "console_scripts": [
            "pysmore_run=pysmore.run:run_all",
            "pysmore_train=pysmore.train:entry_points",
            "pysmore_emb_pred=pysmore.emb_pred:recommendations",
            "pysmore_pred=pysmore.pred:entry_points",
            "pysmore_eval=pysmore.eval:pysmore_eval",
        ]
    }
)
