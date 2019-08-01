import os.path

from flask import Flask, Response


app = Flask(__name__)
app.config.from_object(__name__)


def root_dir():  # pragma: no cover
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "html")


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except IOError as exc:
        return str(exc)


@app.route('/', methods=['GET'])
def mainpage():  # pragma: no cover
    content = get_file('index.html')
    return Response(content, mimetype="text/html")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    complete_path = os.path.join(root_dir(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


if __name__ == '__main__':  # pragma: no cover
    app.run(  # ssl_context=('cert.pem', 'key.pem'),
        port=8877,
        # host="",
    )
