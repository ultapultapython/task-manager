import os
import sys

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def write_file(path, content):
    """Write content to file"""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def setup_project():
    # Create project structure
    directories = [
        'middleware',
        'models',
        'auth',
    ]
    
    for directory in directories:
        create_directory(directory)

    # Create requirements.txt
    requirements_content = '''fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic-settings>=2.0.0
sqlalchemy>=2.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
alembic>=1.12.0
python-dotenv>=1.0.0
psycopg2-binary>=2.9.9'''

    # Create test_requirements.txt
    test_requirements_content = '''requests>=2.31.0'''

    # Create settings.py
    settings_content = '''from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Rate limiting settings
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./taskmanager.db"
    
    # JWT settings
    SECRET_KEY: str = "your-secret-key-here"  # Change this in production!
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()'''

    # Create database.py
    database_content = '''from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from settings import Settings

settings = Settings()

SQLALCHEMY_DATABASE_URL = "sqlite:///./taskmanager.db"  # Using SQLite for simplicity

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()'''

    # Create models/models.py
    models_content = '''from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    projects = relationship("Project", back_populates="owner")
    tasks = relationship("Task", back_populates="assignee")

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="projects")
    tasks = relationship("Task", back_populates="project")

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    status = Column(String, default="pending")  # pending, in_progress, completed
    priority = Column(String, default="medium")  # low, medium, high
    due_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    project_id = Column(Integer, ForeignKey("projects.id"))
    assignee_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    project = relationship("Project", back_populates="tasks")
    assignee = relationship("User", back_populates="tasks")'''

    # Create auth/utils.py
    auth_utils_content = '''from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
from models.models import User

# Configuration
SECRET_KEY = "your-secret-key-here"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user'''

    # Create middleware/rate_limit.py
    rate_limit_content = '''from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from settings import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()

        # Remove old requests outside the window
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time <= self.window_seconds
        ]

        # Check if rate limit is exceeded
        if len(self.requests[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Too many requests")

        # Add current request
        self.requests[client_ip].append(now)

        response = await call_next(request)
        return response'''

    # Create middleware/logging.py
    logging_content = '''from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = (time.time() - start_time) * 1000
        logger.info(f"{request.method} {request.url.path} {response.status_code} {process_time:.2f}ms")
        
        return response'''

    # Create schemas.py
    schemas_content = '''from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: str
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Project schemas
class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    created_at: datetime
    owner_id: int

    class Config:
        from_attributes = True

# Task schemas
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    status: Optional[str] = "pending"
    priority: Optional[str] = "medium"
    due_date: Optional[datetime] = None

class TaskCreate(TaskBase):
    project_id: int

class Task(TaskBase):
    id: int
    created_at: datetime
    project_id: int
    assignee_id: Optional[int] = None

    class Config:
        from_attributes = True

# Response schemas
class ProjectWithTasks(Project):
    tasks: List[Task] = []

class UserResponse(User):
    projects: List[Project] = []
    tasks: List[Task] = []'''

    # Create main.py
    main_content = '''from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List

from database import engine, get_db
from models import models
from settings import settings
from schemas import UserCreate, User, Token, Project, ProjectCreate, Task, TaskCreate, ProjectWithTasks
from auth.utils import (
    get_current_active_user,
    create_access_token,
    verify_password,
    get_password_hash,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from middleware.rate_limit import RateLimitMiddleware
from middleware.logging import LoggingMiddleware

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Task Management API")

# Add middleware
app.add_middleware(RateLimitMiddleware,max_requests=settings.MAX_REQUESTS, window_seconds=settings.WINDOW_SECONDS)
app.add_middleware(LoggingMiddleware)

# Auth routes
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# User routes
@app.post("/users/", response_model=User)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Project routes
@app.post("/projects/", response_model=Project)
async def create_project(
    project: ProjectCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    db_project = models.Project(**project.dict(), owner_id=current_user.id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@app.get("/projects/", response_model=List[ProjectWithTasks])
async def read_projects(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(models.Project).filter(models.Project.owner_id == current_user.id).all()

# Task routes
@app.post("/tasks/", response_model=Task)
async def create_task(
    task: TaskCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Verify project exists and user has access
    project = db.query(models.Project).filter(
        models.Project.id == task.project_id,
        models.Project.owner_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db_task = models.Task(**task.dict(), assignee_id=current_user.id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.get("/tasks/", response_model=List[Task])
async def read_tasks(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return db.query(models.Task).filter(models.Task.assignee_id == current_user.id).all()

@app.put("/tasks/{task_id}/status")
async def update_task_status(
    task_id: int,
    status: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    task = db.query(models.Task).filter(
        models.Task.id == task_id,
        models.Task.assignee_id == current_user.id
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if status not in ["pending", "in_progress", "completed"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    task.status = status
    db.commit()
    return {"message": "Task status updated successfully"}'''

    # Create test_api.py
    test_api_content = '''import requests
import time
import random

BASE_URL = "http://localhost:8000"

def print_response(response, operation):
    print(f"\n=== {operation} ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json() if response.text else 'No content'}")
    print("=" * 50)

def simulate_user_actions():
    # 1. Register a new user
    print("\n🚀 Starting API simulation...")
    
    user_data = {
        "email": f"test{random.randint(1000,9999)}@example.com",
        "username": f"testuser{random.randint(1000,9999)}",
        "password": "testpassword123"
    }
    
    print(f"\n📝 Registering user: {user_data['username']}")
    register_response = requests.post(f"{BASE_URL}/users/", json=user_data)
    print_response(register_response, "User Registration")

    # 2. Login to get access token
    print("\n🔐 Logging in...")
    login_data = {
        "username": user_data["username"],
        "password": user_data["password"]
    }
    login_response = requests.post(f"{BASE_URL}/token", data=login_data)
    print_response(login_response, "Login")

    if login_response.status_code != 200:
        print("❌ Login failed! Exiting...")
        return

    access_token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # 3. Create projects (demonstrating rate limit)
    print("\n📊 Creating multiple projects rapidly to hit rate limit...")
    project_names = [
        "Website Redesign",
        "Mobile App Development",
        "Database Migration",
        "Cloud Infrastructure",
        "Security Audit",
        "API Integration",
        "UI/UX Improvement",
        "Performance Optimization",
        "Data Analytics",
        "Machine Learning Model",
        "DevOps Pipeline",
        "Documentation"
    ]

    for i, name in enumerate(project_names, 1):
        project_data = {
            "title": name,
            "description": f"Project description for {name}"
        }
        
        print(f"\n📁 Creating project {i}: {name}")
        response = requests.post(
            f"{BASE_URL}/projects/",
            json=project_data,
            headers=headers
        )
        print_response(response, f"Create Project {i}")

        if response.status_code == 429:  # Rate limit hit
            print("\n⚠️ Rate limit hit! Waiting for 60 seconds...")
            time.sleep(60)  # Wait for rate limit window to reset
            print("⏰ Resuming operations after rate limit cooldown...")
            
            # Try again after waiting
            response = requests.post(
                f"{BASE_URL}/projects/",
                json=project_data,
                headers=headers
            )
            print_response(response, f"Create Project {i} (Retry)")

    # 4. List all projects
    print("\n📋 Listing all projects...")
    projects_response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    print_response(projects_response, "List Projects")

    # 5. Create tasks for a project
    if projects_response.status_code == 200 and projects_response.json():
        project_id = projects_response.json()[0]["id"]
        
        task_titles = [
            "Research Phase",
            "Planning",
            "Implementation",
            "Testing",
            "Deployment"
        ]

        print("\n✅ Creating tasks for the first project...")
        for title in task_titles:
            task_data = {
                "title": title,
                "description": f"Task description for {title}",
                "project_id": project_id,
                "priority": random.choice(["low", "medium", "high"])
            }
            
            response = requests.post(
                f"{BASE_URL}/tasks/",
                json=task_data,
                headers=headers
            )
            print_response(response, f"Create Task: {title}")

    # 6. List all tasks
    print("\n📋 Listing all tasks...")
    tasks_response = requests.get(f"{BASE_URL}/tasks/", headers=headers)
    print_response(tasks_response, "List Tasks")

if __name__ == "__main__":
    try:
        simulate_user_actions()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
    finally:
        print("\n✨ Simulation completed!")'''

    # Create README.md
    readme_content = '''# Task Management API

A FastAPI-based Task Management API with user authentication, rate limiting, and logging middleware.

## Features

- User authentication with JWT tokens
- Project management
- Task management with priorities and status
- Rate limiting middleware
- Request logging
- SQLite database (easily configurable for PostgreSQL)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn main:app --reload
```

4. To test the API:
```bash
pip install -r test_requirements.txt
python test_api.py
```

## API Endpoints

### Authentication
- POST `/token` - Login and get access token
- POST `/users/` - Register new user
- GET `/users/me/` - Get current user info

### Projects
- POST `/projects/` - Create new project
- GET `/projects/` - List all user's projects

### Tasks
- POST `/tasks/` - Create new task
- GET `/tasks/` - List all user's tasks
- PUT `/tasks/{task_id}/status` - Update task status

## Rate Limiting

The API implements rate limiting of 10 requests per minute per IP address.

## Testing

The `test_api.py` script simulates a real user:
1. Registers a new user
2. Logs in to get an access token
3. Creates multiple projects (demonstrates rate limiting)
4. Creates tasks for a project
5. Lists all projects and tasks'''

    # Write all files
    files = {
        'requirements.txt': requirements_content,
        'test_requirements.txt': test_requirements_content,
        'settings.py': settings_content,
        'database.py': database_content,
        'models/models.py': models_content,
        'auth/utils.py': auth_utils_content,
        'middleware/rate_limit.py': rate_limit_content,
        'middleware/logging.py': logging_content,
        'schemas.py': schemas_content,
        'main.py': main_content,
        'test_api.py': test_api_content,
        'README.md': readme_content
    }

    # Create __init__.py files
    init_directories = ['models', 'middleware', 'auth']
    for directory in init_directories:
        write_file(os.path.join(directory, '__init__.py'), '')

    for file_path, content in files.items():
        write_file(file_path, content)

    print("\n✨ Project setup completed successfully!")
    print("\nTo get started:")
    print("1. Create a virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Run the application:")
    print("   uvicorn main:app --reload")
    print("\n4. In another terminal, install test requirements and run the test script:")
    print("   pip install -r test_requirements.txt")
    print("   python test_api.py")

if __name__ == "__main__":
    try:
        setup_project()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 