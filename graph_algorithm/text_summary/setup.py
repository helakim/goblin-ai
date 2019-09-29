# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
#
#            7''  Q..\
#         _7         (_
#       _7  _/    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /-~-/||'=.__  '::. '-~'' {             ___   /  //     ./{
#  V   V-~-~| ||   __''_   ':::.   ''~-~.___.-'' _/  // / {_   /  {  /
#   VV/-~-~-|/ \ .'__'. '.    '::                     _ _ _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (~-~-~\\.-' /    \'   \::::. | '_ ` _ \ / _ \_  / | | |/ _` | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from setuptools import find_packages, setup


CURRENT_PYTHON = sys.version_info[:2]
CERBERUS_REQUIRED_PYTHON = (3, 5)

SKLEARN_MIN_VERSION = '0.19'
GENSIM_MIN_VERSION = '3.6.0'
NUMPY_MIN_VERSION = '1.14.1'
SCIPY_MIN_VERSION = '1.0.0'
NETWORKX_MIN_VERSION = '2.1'  # PageRank Algorithm
KOLNPY_MIN_VERSION = '0.4.4'  # Corpus(mecab, kkma, etc...)

# This check and everything above must remain compatible with python 2.X.
##########################################################################
#                               INFO                                     #
#                         Unsupported Python                             #
##########################################################################

if CURRENT_PYTHON < CERBERUS_REQUIRED_PYTHON:
    sys.stderr.write("""
    This version of Module requires Python {} {}, but you're trying to
    install it on Python {} {}
    """).format(*(CERBERUS_REQUIRED_PYTHON + CURRENT_PYTHON))
    sys.exit(1)

REQUIREMENTS = [
    'scikit-learn>={0}'.format(SKLEARN_MIN_VERSION),
    'gensim>={0}'.format(GENSIM_MIN_VERSION),
    'numpy>={0}'.format(NUMPY_MIN_VERSION),
    'scipy>={0}'.format(SCIPY_MIN_VERSION),
    'networkx>={0}'.format(NETWORKX_MIN_VERSION),
    'konlpy>={0}'.format(KOLNPY_MIN_VERSION),
]

with open('README.md') as f:
    readme = f.read()


with open('LICENSE') as f:
    license = f.read()

setup(
    name="text_summary",
    version="0.1.0",
    author="kt.kim",
    author_email="firefoxdev0619@gmail.com",
    long_description=readme,
    license=license,
    description="A graph-based ranking model for text processing and show how this applications",
    packages=find_packages(exclude=('data, assert')),
    install_requires=REQUIREMENTS,
    classifiers=[
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: 3'
        'Programming Language :: 3.5',
        'Programming Language :: 3.6',
        'Programming Language :: 3 :: only',
        'Programming Language :: Python :: Implementation :: CPython'
    ]
)