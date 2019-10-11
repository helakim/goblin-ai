# *********************************************************************
# @Project    basic_model_server
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   10/10/2019
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
import sys
from setuptools import find_packages, setup

CURRENT_PYTHON = sys.version_info[:2]
CERBERUS_REQUIRED_PYTHON = (3, 5)

NUMPY_MIN_VERSION = '1.14.3'
WERKZEUG_MIN_VERSION = '0.14.1'  # Werkzug is a comprehensive WSGI web Application library.
WAITRESS_MIN_VERSION = '1.1.0'
GUNICORN_MIN_VERSION = '19.8.1'  # Gunicorn Not Supported for windows
TWISTED_MIN_VERSION = '18.4.0'

FLASK_MIN_VERSION = '1.0.2'  # Flask is a lighweight WSGI web application framework.
FLASK_JSON_MIN_VERSION = '0.3.2'
FLASK_CORS_MIN_VERSION = '3.0.4'
OPENCV_PYTHON_MIN_VERSION = '3.4.1.15'
PIL_MIN_VERSION_MIN_VERSION = '5.1.0'
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
    "numpy>={0}".format(NUMPY_MIN_VERSION),
    "Werkzeug>={0}".format(WERKZEUG_MIN_VERSION),
    "waitress>={0}".format(WAITRESS_MIN_VERSION),
    "gunicorn>={0}".format(GUNICORN_MIN_VERSION),
    "Twisted>={0}".format(TWISTED_MIN_VERSION),
    "Flask>={0}".format(FLASK_MIN_VERSION),
    "Flask-JSON>={0}".format(FLASK_JSON_MIN_VERSION),
    "Flask-Cors>={0}".format(FLASK_CORS_MIN_VERSION),
    "opencv-python>={0}".format(OPENCV_PYTHON_MIN_VERSION),
    "Pillow>={0}".format(PIL_MIN_VERSION_MIN_VERSION)
]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="goblin_ai_server",
    version="0.1.0",
    author="kt.kim",
    author_email="firefoxdev0619@gamil.com",
    long_description=readme,
    license=license,
    description="## Simple RestApi Model Server for Goblin-Ai Deploy",
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
