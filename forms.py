from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, IntegerField, validators, BooleanField, Form
from wtforms.validators import DataRequired, Email, EqualTo

class UploadForm(FlaskForm):
    files = FileField('Upload Images', validators=[DataRequired()], render_kw={"webkitdirectory": "webkitdirectory", "directory": "directory", "multiple": "multiple"})
    nb_cluster = IntegerField('Number of Clusters', validators=[DataRequired()])
    submit = SubmitField('Submit', render_kw={"class":"submit_btn"})

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()],render_kw={"class":"field", 'placeholder': 'username'})
    email = StringField('Email', validators=[DataRequired(), Email()],render_kw={"class":"field", 'placeholder': 'email'})
    password = PasswordField('Password', validators=[DataRequired()],render_kw={"class":"field", 'placeholder': 'password'})
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')],render_kw={"class":"field", 'placeholder': 'confirm password'})
    submit = SubmitField('Register', render_kw={"class":"submit_btn"})

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()],render_kw={"class":"field", 'placeholder': 'email'})
    password = PasswordField('Password', validators=[DataRequired()],render_kw={"class":"field", 'placeholder': 'password'})
    submit = SubmitField('Login', render_kw={"class":"submit_btn"})