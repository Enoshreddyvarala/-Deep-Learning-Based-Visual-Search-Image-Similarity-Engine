from datetime import datetime
from flask_login import UserMixin
from extensions import db, login_manager


class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    searches = db.relationship(
        "SearchHistory", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"


class SearchHistory(db.Model):
    __tablename__ = "search_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    query_image_path = db.Column(db.String(255), nullable=False)
    top_k = db.Column(db.Integer, default=8)
    results_json = db.Column(db.Text, nullable=False)  # serialized list of result dicts
    inference_mode = db.Column(db.String(50), default="demo")  # "trained_model" or "demo"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SearchHistory {self.id} user={self.user_id}>"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
