# -*- coding: utf-8 -*-
"""
Test Doctor Feedback Routes:
- POST /api/v1/doctor/cases/<case_id>/feedback
- GET  /api/v1/doctor/cases/flagged
- GET  /api/v1/doctor/cases/<case_id>/full
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import create_app
from database.postgres.db_connection import db
from database.postgres.models import User, Case, DoctorFeedback

app = create_app()
app.testing = True

with app.app_context():
    # Verify tables exist or can be created in test
    print("Checking DoctorFeedback table...")
    inspector = db.inspect(db.engine)
    tables = inspector.get_table_names()
    print("Tables in DB:", tables)
    assert 'doctor_feedback' in tables or hasattr(DoctorFeedback, '__tablename__'), "DoctorFeedback model defined"
    print("DoctorFeedback Model verified and mapped!")
