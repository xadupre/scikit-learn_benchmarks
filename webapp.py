from starlette.routing import Router, Mount
from starlette.staticfiles import StaticFiles


app = Router(routes=[
    Mount('/', app=StaticFiles(directory='html'), name="html"),
])
