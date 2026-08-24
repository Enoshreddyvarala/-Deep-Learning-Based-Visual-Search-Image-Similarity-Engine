import os
from flask import Flask, render_template

from config import Config, BASE_DIR
from extensions import db, login_manager, bcrypt


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)

    from auth import auth_bp
    from main import main_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    with app.app_context():
        import models  # noqa: F401  ensure models are registered
        db.create_all()

        from ml.inference import ModelService
        app.model_service = ModelService(
            app_root=BASE_DIR,
            artifacts_dir=app.config["MODEL_ARTIFACTS_DIR"],
            demo_catalog_dir=app.config["DEMO_CATALOG_DIR"],
        )
        print(f"[app] Visual search model loaded in '{app.model_service.mode}' mode "
              f"({app.model_service.info()['catalog_size']} catalog images).")

    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(e):
        return render_template("404.html", message="File too large (max 8 MB)."), 413

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
