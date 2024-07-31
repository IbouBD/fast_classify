import shutil
from flask import Flask, request, redirect, url_for, render_template, session, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from database import create_app, db, User, Role, user_datastore, AnonymousUser
from flask_security.utils import hash_password
from forms import UploadForm, RegistrationForm, LoginForm
from tasks import  process_images, celery
from utils import *
from celery.result import AsyncResult
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
from flask_talisman import Talisman
import random
from apscheduler.schedulers.background import BackgroundScheduler
import uuid
import os,time,stat

csrf = CSRFProtect()
# Configuration de Flask-Talisman
csp = {
    'default-src': [
        '\'self\'',
        'cdnjs.cloudflare.com',
        'ajax.googleapis.com'
    ],
    'script-src': [
        '\'self\'',
        'cdnjs.cloudflare.com',
        'ajax.googleapis.com',
        '\'strict-dynamic\'',
        "data: https://ssl.gstatic.com 'unsafe-inline' 'unsafe-eval'",
        # The nonce placeholder will be replaced by Talisman
    ],
    'style-src': [
        '\'self\'',
        '\'unsafe-inline\'',
        'cdnjs.cloudflare.com'
        
    ],
    'img-src': '*',
}

app, celery = create_app()
db.init_app(app=app)
bcrypt = Bcrypt(app)
csrf.init_app(app)
#talisman = Talisman(
#      app,
#      content_security_policy=csp,
#      content_security_policy_nonce_in=['script-src']
#  )
login_manager = LoginManager()

login_manager.init_app(app)  
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
login_manager.anonymous_user = AnonymousUser

def generate_nonce(length=8):
    """Generate pseudorandom number."""
    return ''.join([str(random.randint(0, 9)) for i in range(length)])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)



@app.route("/", methods=["GET", "POST"])
async def index():
    form = UploadForm()
    if form.validate_on_submit():
        files = request.files.getlist('files')  # Get the list of files
        nb_cluster = form.nb_cluster.data
        print("nb_cluster", nb_cluster)
        image_data = []

        if current_user.is_authenticated:
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(current_user.id))
            user_data = os.path.join(app.config['SORTED_FOLDER'], str(current_user.id))
            os.makedirs(user_folder, exist_ok=True)
            os.makedirs(user_data, exist_ok=True)
        else:
            
            anonymous_user_id= session.get('anonymous_id')
            
            user_folder = os.path.join(app.config['UPLOAD_FOLDER'], current_user.id)
            user_data = os.path.join(app.config['SORTED_FOLDER'], current_user.id)
            os.makedirs(user_folder, exist_ok=True)
            os.makedirs(user_data, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(user_folder, filename)
                file.save(file_path)
                image_data.append((filename, file_path))

        if len(image_data) < nb_cluster:
            flash(f"You can't have more groups than images, number of groups : {nb_cluster}, number of images : {len(image_data)}", 'danger')
            image_data = []
            return redirect(url_for('index'))
    
        task = process_images.apply_async(args=[image_data, nb_cluster, user_folder, user_data])
        session['task_id'] = task.id
        
        return redirect(url_for('show_plot', task_id=task.id))

    return render_template("index.html", form=form, plot_url=None)

@app.route('/task_status/<task_id>')
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'SUCCESS':
        return jsonify({'status': 'SUCCESS', 'result': url_for('static', filename='cluster_plot.html')})
    elif task.state == 'PENDING':
        return jsonify({'status': 'PENDING'})
    else:
        return jsonify({'status': 'FAILURE'})

@app.route('/show_plot/<task_id>', methods=["GET", "POST"])
def show_plot(task_id):
    nonce=generate_nonce()
    return render_template('show_plot.html', task_id=task_id, nonce=nonce)


@app.route('/download_zip/<task_id>', methods=["GET"])
def download_zip(task_id):
    if current_user.is_authenticated:
        user_sorted_folder = os.path.join(app.config['SORTED_FOLDER'], str(current_user.id))
        user_zip_folder = os.path.join(app.config['ZIP_FOLDER'], str(current_user.id))
   
    else:
        user_sorted_folder = os.path.join(app.config['SORTED_FOLDER'], current_user.id)
        user_zip_folder = os.path.join(app.config['ZIP_FOLDER'], current_user.id)

    os.makedirs(user_zip_folder, exist_ok=True)
    try :
        zip_path = shutil.make_archive(
            base_name=os.path.join(user_zip_folder, 'sorted_images'),
            format='zip',
            root_dir=user_sorted_folder
        )
    
    except Exception as e:
        flash('The zip file does not exist.', 'danger')
        return redirect(url_for('index'))

    if os.path.exists(zip_path):
        try:
            response = send_file(zip_path, as_attachment=True)

            # Ensure file is closed after sending
            response.call_on_close(cleanup_files(zip_path, user_sorted_folder, user_zip_folder))
            return response
        except Exception as e:
            print(e)
            flash('The zip file does not exist.', 'danger')
            return redirect(url_for('show_plot', task_id=task_id))
            
    else:
        flash('The zip file does not exist.', 'danger')
        return redirect(url_for('show_plot', task_id=task_id))
    

    
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        hash_password = bcrypt.generate_password_hash(password).decode('utf-8')
        #print(f"Username: {username}, Email: {email}, Password: {password}")

        # Vérifier si l'utilisateur ou l'email existe déjà
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user is None:
            try:
                new_user = User(username=username, email=email, password=hash_password)
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful. You can now log in.','success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                flash('An error occurred while creating your account. Please try again.', 'danger')
                print(f"Error: {e}")
                return redirect(url_for('register'))
        else:
            flash('Username or Email already exists.', 'danger')
            return redirect(url_for('register'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            flash('Login successful.', 'success')
            session['loggedin'] = True
            session['email'] = email
            session['username'] = user.username
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def  logout():
    logout_user()
    session.pop('loggedin', None)
    session.pop('email', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
