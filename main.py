import os
import json
import uuid
from flask import (
    Blueprint, render_template, request, redirect, url_for,
    flash, current_app, send_from_directory
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from extensions import db
from models import SearchHistory

main_bp = Blueprint("main", __name__)


def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


@main_bp.route("/")
def home():
    model_info = current_app.model_service.info()
    return render_template("home.html", model_info=model_info)


@main_bp.route("/search", methods=["GET", "POST"])
@login_required
def search():
    results = None
    query_image_url = None
    inference_mode = None

    if request.method == "POST":
        file = request.files.get("shoe_image")
        top_k = int(request.form.get("top_k", current_app.config["TOP_K_DEFAULT"]))

        if not file or file.filename == "":
            flash("Please choose an image to upload.", "danger")
            return redirect(url_for("main.search"))

        if not _allowed_file(file.filename):
            flash("Unsupported file type. Please upload a PNG or JPG image.", "danger")
            return redirect(url_for("main.search"))

        filename = secure_filename(file.filename)
        unique_name = f"{current_user.id}_{uuid.uuid4().hex[:8]}_{filename}"
        save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        results, inference_mode = current_app.model_service.find_similar(save_path, top_k=top_k)
        query_image_url = url_for("static", filename=f"uploads/{unique_name}")

        history_entry = SearchHistory(
            user_id=current_user.id,
            query_image_path=f"uploads/{unique_name}",
            top_k=top_k,
            results_json=json.dumps(results),
            inference_mode=inference_mode,
        )
        db.session.add(history_entry)
        db.session.commit()

    model_info = current_app.model_service.info()
    return render_template(
        "search.html",
        results=results,
        query_image_url=query_image_url,
        inference_mode=inference_mode,
        model_info=model_info,
    )


@main_bp.route("/history")
@login_required
def history():
    entries = (
        SearchHistory.query.filter_by(user_id=current_user.id)
        .order_by(SearchHistory.created_at.desc())
        .all()
    )
    parsed = []
    rewrite = current_app.model_service._to_static_catalog_path
    for e in entries:
        results = json.loads(e.results_json)
        for r in results:
            if "path" in r:
                r["path"] = rewrite(r["path"])
        parsed.append({
            "id": e.id,
            "query_image_url": url_for("static", filename=e.query_image_path),
            "top_k": e.top_k,
            "results": results,
            "inference_mode": e.inference_mode,
            "created_at": e.created_at,
        })
    return render_template("history.html", entries=parsed)


@main_bp.route("/history/<int:entry_id>/delete", methods=["POST"])
@login_required
def delete_history(entry_id):
    entry = SearchHistory.query.filter_by(id=entry_id, user_id=current_user.id).first_or_404()
    db.session.delete(entry)
    db.session.commit()
    flash("Search removed from history.", "info")
    return redirect(url_for("main.history"))


# Serve demo catalog images (they live outside static/uploads but under static/,
# so Flask's default static route already handles static/catalog_demo/** — this
# route only exists for symmetry if the catalog folder is later moved).
