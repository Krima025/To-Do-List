import os
import json
import numpy as np
import joblib
from dotenv import load_dotenv 
load_dotenv()
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, render_template, redirect, url_for, request, session, jsonify, has_request_context, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from sqlalchemy.orm import relationship
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from flask_dance.contrib.google import make_google_blueprint, google

# --- 1. CONFIGURATION & ENVIRONMENT FIXES ---
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'

app = Flask(__name__) # You already have this line

# --- ADD THESE LINES TO FIX THE LOGIN LOOP ---
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['PREFERRED_URL_SCHEME'] = 'https'
# ---------------------------------------------
app.secret_key = os.getenv("FLASK_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app.config['GOOGLE_OAUTH_CLIENT_ID'] = os.getenv("GOOGLE_CLIENT_ID")
app.config['GOOGLE_OAUTH_CLIENT_SECRET'] = os.getenv("GOOGLE_CLIENT_SECRET")

# In todo.py, find the configuration section and ensure this block is present:


# --- Database Configuration (PROJECT FOLDER PATH) ---
# This path points to: to-do-app/data/todo.db
db_path = os.path.join(os.path.dirname(__file__), 'data', 'todo.db')

# Ensure the 'data' directory exists before SQLAlchemy tries to write the file
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# --- Database Configuration (SQLite for development) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
# --- FIX FOR RENDER DATABASE ---

# --- 2. AUTHENTICATION SETUP ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# --- 3. MODELS ---
DEFAULT_LABELS = ["Projects", "Works", "School", "Others", "Unlabeled"]
PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
PRIORITY_MAP_INV = {0: 'Low', 1: 'Medium', 2: 'High'}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    google_id = db.Column(db.String(256), unique=True, nullable=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100))
    profile_pic = db.Column(db.String(255))
    _labels_json = db.Column('labels_json', db.Text, default=json.dumps(DEFAULT_LABELS))
    
    tasks = db.relationship('Task', backref='user', lazy=True, cascade="all, delete-orphan")
    
    @property
    def labels(self): return json.loads(self._labels_json)
    @labels.setter
    def labels(self, value): self._labels_json = json.dumps(value)

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(20))
    time = db.Column(db.String(10))
    label = db.Column(db.String(50))
    completed = db.Column(db.Boolean, default=False)
    
    # NEW ML FIELDS
    user_priority = db.Column(db.String(10), default='Medium') # Priority set by user (feature)
    ai_priority = db.Column(db.String(10), default='Medium')   # Priority predicted by model (output)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return { 
            'id': self.id, 
            'title': self.title, 
            'date': self.date, 
            'time': self.time, 
            'label': self.label, 
            'completed': self.completed,
            'user_priority': self.user_priority,
            'ai_priority': self.ai_priority
        }
with app.app_context():
    db.create_all()
    print("--- DEBUG: Database Tables Created Successfully ---")

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- 4. MODEL TRAINING & PREDICTION LOGIC ---
# --- ADD THIS BLOCK ---

# Define the location where the trained SVM model will be saved
db_folder = 'data'
MODEL_PATH = os.path.join(db_folder, 'svm_priority_model.joblib')

# --- END OF BLOCK ---
def train_and_save_model(user_id, tasks_data):
    """Trains an SVM model on the user's task data and saves it."""
    
    now = datetime.now()
    features = []
    targets = []
    
    for task in tasks_data:
        try:
            deadline_str = f"{task.date} {task.time or '00:00'}"
            deadline = datetime.strptime(deadline_str, '%Y-%m-%d %H:%M')
            
            time_difference = (deadline - now).total_seconds() / 3600 # Time remaining in hours
            
            normalized_urgency = 1 / (1 + abs(time_difference) / 5) 
            
            target = PRIORITY_MAP.get(task.user_priority, 1) # Target is 0 (Low), 1 (Medium), or 2 (High)
            
            label_index = DEFAULT_LABELS.index(task.label) if task.label in DEFAULT_LABELS else 4
            title_length = len(task.title)
            
            features.append([time_difference, normalized_urgency, label_index, title_length])
            targets.append(target)
            
        except ValueError:
            continue

    if len(features) < 3:
        # Placeholder training for when data is insufficient
        clf = SVC(kernel='linear')
        X_placeholder = np.array([[0, 0, 0, 0], [10, 0, 1, 1], [-10, 0, 2, 2]])
        y_placeholder = np.array([0, 1, 2])
        clf.fit(X_placeholder, y_placeholder)
        print("DEBUG: Trained stable placeholder model (3 classes).")
    else:
        X = np.array(features)
        y = np.array(targets)
        
        clf = SVC(kernel='linear', C=1.0, random_state=42)
        clf.fit(X, y)
        print(f"DEBUG: Trained custom SVM model with {len(features)} samples.")

    # CRITICAL: Save the model using the defined path
    joblib.dump(clf, MODEL_PATH)
# --- 5. OAUTH & VIEW SETUP ---

google_bp = make_google_blueprint(
    scope=["profile", "email"],
    redirect_to="index"
)
app.register_blueprint(google_bp, url_prefix="/login")

@app.before_request
def before_request():
    if not has_request_context() or current_user.is_authenticated: return

    try:
        if google.authorized:
            # ... (Login Hook remains the same) ...
            resp = google.get("/oauth2/v2/userinfo")
            if resp.ok:
                info = resp.json()
                google_id = str(info['id'])
                user = db.session.execute(db.select(User).filter_by(google_id=google_id)).scalar_one_or_none()
                
                if not user:
                    user = User(google_id=google_id, email=info['email'], name=info.get('name', 'User'))
                    db.session.add(user)
                    db.session.commit()
                
                login_user(user, remember=True)
    except Exception as e:
        print(f"OAuth Error: {e}")

# --- 6. API ENDPOINTS ---
def predict_task_priority(task_data):
    """Loads the user's custom SVM model and predicts priority."""
    if not os.path.exists(MODEL_PATH):
        # Fallback: model hasn't been created yet (or failed training)
        return 'Medium'
        
    try:
        # Load the model
        clf = joblib.load(MODEL_PATH)
        
        # Prepare feature vector (same structure as training)
        now = datetime.now()
        
        # We need to handle two date formats: YYYY-MM-DD HH:MM
        deadline_str = f"{task_data['date']} {task_data.get('time') or '00:00'}"
        deadline = datetime.strptime(deadline_str, '%Y-%m-%d %H:%M')
        
        # 1. Time Difference (in hours)
        time_difference = (deadline - now).total_seconds() / 3600
        
        # 2. Normalized Urgency (0 to 1)
        normalized_urgency = 1 / (1 + abs(time_difference) / 5)
        
        # 3. Label Index
        try:
            label_index = DEFAULT_LABELS.index(task_data['label'])
        except ValueError:
            label_index = 4 # Default index for non-default labels
            
        # 4. Title Length
        title_length = len(task_data['title'])
        
        # Create feature vector
        feature_vector = np.array([[time_difference, normalized_urgency, label_index, title_length]])
        
        # Make prediction
        prediction_int = clf.predict(feature_vector)[0]
        
        # Convert prediction (0, 1, 2) back to priority string (Low, Medium, High)
        return PRIORITY_MAP_INV.get(prediction_int, 'Medium')
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        # Ensure a safe fallback to prevent crashing the app
        return 'Medium'
    
@app.route('/api/predict-priority', methods=['POST'])
@login_required
def predict_priority_api():
    task_data = request.json
    priority = predict_task_priority(task_data)
    return jsonify({'ai_priority': priority})


@app.route('/api/task', methods=['POST'])
@login_required
def create_task():
    data = request.json
    task_id = data.get('id')
    
    # 1. Get AI Prediction for new tasks or update existing tasks
    ai_priority = predict_task_priority(data)
    
    if task_id:
        task = db.session.get(Task, task_id)
        if task and task.user_id == current_user.id:
            # Update existing task
            task.title = data.get('title')
            task.date = data.get('date')
            task.time = data.get('time', '')
            task.label = data.get('label', 'Unlabeled')
            task.completed = data.get('completed', task.completed)
            task.user_priority = data.get('user_priority', 'Medium')
            task.ai_priority = ai_priority # Update AI prediction
            db.session.commit()
            return jsonify(task.to_dict())
        return jsonify({'error': 'Task not found or unauthorized'}), 404
    else:
        # Create new task
        new_task = Task(
            title=data.get('title'),
            date=data.get('date'),
            time=data.get('time', ''),
            label=data.get('label', 'Unlabeled'),
            user_priority = data.get('user_priority', 'Medium'),
            ai_priority = ai_priority, # Store initial prediction
            completed=False,
            user_id=current_user.id
        )
        db.session.add(new_task)
        db.session.commit()
        
        # Trigger re-train after adding a new data point
        train_and_save_model(current_user.id, current_user.tasks) 
        
        return jsonify(new_task.to_dict()), 201

# --- PASTE THIS INTO YOUR PYTHON FILE ---

# --- ENSURE THIS IS IN APP.PY ---
@app.route('/api/task/<int:task_id>/delete', methods=['DELETE'])
@login_required
def delete_task(task_id):
    print(f"--- ATTEMPTING DELETE ON TASK ID: {task_id} ---") # Debug print
    
    task = db.session.get(Task, task_id)
    
    if not task:
        print("--- ERROR: Task not found in DB ---")
        return jsonify({'error': 'Task not found'}), 404
        
    if task.user_id != current_user.id:
        print(f"--- ERROR: User ID mismatch. Task User: {task.user_id}, Current: {current_user.id} ---")
        return jsonify({'error': 'Unauthorized'}), 404

    db.session.delete(task)
    db.session.commit()
    print("--- SUCCESS: Task deleted ---")
    return jsonify({'success': True})

# --- 7. VIEW ROUTES & APP STARTUP ---
@app.route('/', endpoint='index')
def index():
    # ... (view logic remains the same) ...
    labels = DEFAULT_LABELS 
    initial_tasks = []
    session_data = {'isAuthenticated': False, 'user': None}
    
    if current_user.is_authenticated:
        db.session.expire(current_user)
        
        labels = current_user.labels
        
        session_data = {
            'isAuthenticated': True,
            'user': {'name': current_user.name, 'email': current_user.email}
        }
        # Tasks are automatically fetched with new priority fields
        initial_tasks = [t.to_dict() for t in current_user.tasks]
    
    elif session.get('is_guest'):
        labels = DEFAULT_LABELS
        session_data = {
            'isAuthenticated': True,
            'user': {'name': 'Guest User', 'email': 'guest@local'}
        }
        
    return render_template('home.html', 
                           initial_tasks=initial_tasks, 
                           initial_labels=labels, 
                           session_data=session_data)

@app.route('/guest')
def guest_login():
    session['is_guest'] = True
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    # ... (logout logic remains the same) ...
    if current_user.is_authenticated:
        logout_user()
    session.clear() 

    response = make_response(redirect(url_for('index')))
    
    response.delete_cookie('session')
    response.delete_cookie(app.config.get('SESSION_COOKIE_NAME'))
    response.delete_cookie('remember_token') 
    response.set_cookie('remember_token', '', expires=0) 

    return response

@app.after_request
def add_security_headers(response):
    # ... (security headers remain the same) ...
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Initial training on startup if no model exists
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            print("Training initial placeholder model...")
            train_and_save_model(None, []) # Train a medium-priority placeholder model
            
    app.run(debug=True, port=5001)