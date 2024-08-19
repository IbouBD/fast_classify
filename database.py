from flask import Flask, request, redirect, url_for, render_template, session, flash, send_file, jsonify
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin
from flask_login import AnonymousUserMixin

from flask_security.utils import hash_password
import uuid
from flask_sqlalchemy import SQLAlchemy

from celery import Celery
from celery.schedules import crontab
import redis
from redis import StrictRedis
from utils import*


def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)

    # Configuration de la base de données
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Configuration de Flask-Security
    app.config['SECURITY_REGISTERABLE'] = True
    app.config['SECURITY_PASSWORD_SALT'] = 'supersecretsalt'
    app.config['WTF_CSRF_SECRET_KEY'] = 'somesupercsrf'
    # app.config['SECURITY_PASSWORD_HASH'] = 'bcrypt'
    # app.config['SECURITY_PASSWORD_SINGLE_HASH'] = ['bcrypt']
    # app.config['SECURITY_SEND_REGISTER_EMAIL'] = False
    # app.config['SECURITY_SEND_PASSWORD_CHANGE_EMAIL'] = False
    # app.config['SECURITY_SEND_PASSWORD_RESET_EMAIL'] = False
    # app.config['SECURITY_CHANGEABLE'] = True
    # app.config['SECURITY_RECOVERABLE'] = True

    # Configuration de l'upload de fichiers
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SORTED_FOLDER'] = SORTED_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
    app.config['ZIP_FOLDER'] = ARCHIVE_NAME

    # Configuration de Flask-Session
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SESSION_REDIS'] = StrictRedis(host='redis', port=6379)  # Update this line
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # Configuration de Celery
    celery = Celery(app.import_name, backend='redis://redis:6379/0', broker='redis://redis:6379/0')  # Update this line
    app.config.from_mapping(
        CELERY=dict(
            broker_url="redis://redis:6379/0",
            result_backend="redis://redis:6379/0",
            task_ignore_result=True,
            beat_schedule={
                "task-every-10-seconds": {
                    "task": "app.tasks.del_file",
                    "schedule": 10,
                }
            },
        ),
    )
    celery.conf.update(app.config)

    # Configuration de sécurité supplémentaire
    # csrf = CSRFProtect(app)
    # talisman = Talisman(app, content_security_policy={
    #     'default-src': [
    #         '\'self\'',
    #         'cdnjs.cloudflare.com'
    #     ],
    #     'img-src': '*',
    #     'style-src': [
    #         '\'self\'',
    #         '\'unsafe-inline\'',
    #         'cdnjs.cloudflare.com'
    #     ],
    #     'script-src': [
    #         '\'self\'',
    #         'cdnjs.cloudflare.com',
    #         'ajax.googleapis.com'
    #     ]
    # })

    return app, celery

app, celery = create_app()
db = SQLAlchemy(app)

# Définition des modèles
roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255))
    active = db.Column(db.Boolean(), default=True)
    confirmed_at = db.Column(db.DateTime)
    fs_uniquifier = db.Column(db.String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    roles = db.relationship(
        'Role', 
        secondary=roles_users, 
        backref=db.backref('users', lazy='dynamic')
    )

    def __repr__(self):
        return f'<User {self.username}>'
    
class AnonymousUser(AnonymousUserMixin):
    @property
    def id(self):
        if 'anonymous_id' not in session:
            session['anonymous_id'] = str(uuid.uuid4())
        return session['anonymous_id']

    def __repr__(self):
        return f'<User Anonymous {self.id}>'

class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)
    description = db.Column(db.String(255))

user_datastore = SQLAlchemyUserDatastore(db, User, Role)