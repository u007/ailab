import uvicorn

from locateanything_api.app import create_app
from locateanything_api.config import get_settings


def log_registered_routes() -> None:
    routes = []
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/v1/"):
            methods = ",".join(sorted(getattr(route, "methods", []) or []))
            routes.append(f"{methods} {path}")
    if routes:
        print("Registered OpenAI-compatible routes: " + "; ".join(sorted(routes)), flush=True)


def main() -> None:
    settings = get_settings()
    log_registered_routes()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


app = create_app()


if __name__ == "__main__":
    main()
