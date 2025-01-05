from flask import Blueprint

assistml_bp = Blueprint('assistml', __name__)
upload_bp = Blueprint('upload', __name__)


from assistml.api import assistml, upload
