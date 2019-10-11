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
import fire
import pprint

from waitress import serve
from flask import Flask
from flask_json import json_response, as_json
from flask_cors import CORS

# Do not use the development server in a production environment.
# Create the application instance
app = Flask(__name__)
CORS(app)
app.config.from_object(__name__)  # Load config from app.py file
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_LENGTH'] = (1024 * 1024) * 5  # 1MB
app.config['ENV_DEFAULT_PORT'] = 8000
app.config['ENV_DEBUG_MODE'] = True
app.config['JSON_ADD_STATUS'] = True
app.config['JSON_STATUS_FIELD_NAME'] = 'cerberus_status'
app.config['JSON_JSONP_OPTIONAL'] = False
app.config['JSON_DECODE_ERROR_MESSAGE'] = True
app.config['APP_HOST_NAME'] = '127.0.0.1'
app.config['SECRET_KEY'] = b"\xacL\xcc\xc7\x0cRK\xfaYDZMN\xe3]]"


def response_json_ops(custom_status=200, status=200, res_msg='life is too short we need Goblin-AI'):
    return json_response(cerberus_status=custom_status, status_=status, message=res_msg)


def server_ops():
    p = pprint.PrettyPrinter(indent='4')
    # ------
    # Display server information :)
    p.pprint(app.config)
    # To allow aptana to receive errors, set use_debugger=False
    # app.run(port=app.config['ENV_DEFAULT_PORT'], debug=app.config['ENV_DEBUG_MODE'])
    # Deploy Server with Web Serve Gateway Interface
    serve(app=app, host=app.config['APP_HOST_NAME'], port=app.config['ENV_DEFAULT_PORT'])


@app.route('/goblin_ai_get_test', methods=["GET"])
@as_json
def test_api_get():
    response_header = response_json_ops()

    return response_header


@app.route('/goblin_ai_post_test', methods=["POST"])
@as_json
def test_api_post():
    response_header = response_json_ops()

    return response_header


if __name__ == '__main__':
    fire.Fire(server_ops)
