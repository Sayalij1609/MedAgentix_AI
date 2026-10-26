from datetime import datetime, timezone
from database.postgres.db_connection import db

class User(db.Model):
    """
    User ORM Model representing the 'users' table.
    Contains profile, credentials, and roles for access control.
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='patient')
    date_of_birth = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<User id={self.id} email='{self.email}' role='{self.role}'>"

    def to_dict(self):
        """
        Serializes user details to a dictionary.
        Safely excludes sensitive password hash.
        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "gender": self.gender,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Case(db.Model):
    """
    Case ORM Model representing the 'cases' table.
    Stores clinical intake, vitals, symptoms, history, and diagnostic output.
    """
    __tablename__ = 'cases'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    status = db.Column(db.String(50), nullable=False, default='pending')  # 'pending', 'processing', 'completed', 'reviewed'
    triage_level = db.Column(db.Integer, nullable=True)
    
    # JSON columns
    vitals = db.Column(db.JSON, nullable=False)
    symptoms = db.Column(db.JSON, nullable=False)
    history_and_lifestyle = db.Column(db.JSON, nullable=False)
    diagnostic_output = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    patient = db.relationship('User', foreign_keys=[patient_id], backref=db.backref('cases_as_patient', lazy='dynamic', cascade='all, delete-orphan'))
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref=db.backref('cases_as_doctor', lazy='dynamic'))

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "doctor_id": self.doctor_id,
            "status": self.status,
            "triage_level": self.triage_level,
            "vitals": self.vitals,
            "symptoms": self.symptoms,
            "history_and_lifestyle": self.history_and_lifestyle,
            "diagnostic_output": self.diagnostic_output,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DoctorFeedback(db.Model):
    """
    DoctorFeedback ORM Model representing the 'doctor_feedback' table.
    Captures physician review, validation, diagnosis overrides, rating,
    and clinical notes for continuous model evaluation and human-in-the-loop audit.
    """
    __tablename__ = 'doctor_feedback'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    case_id = db.Column(db.Integer, db.ForeignKey('cases.id', ondelete='CASCADE'), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)

    # Feedback on AI diagnosis
    ai_diagnosis_correct = db.Column(db.Boolean, nullable=True)
    corrected_diagnosis = db.Column(db.String(255), nullable=True)
    severity_correct = db.Column(db.Boolean, nullable=True)
    corrected_severity = db.Column(db.String(50), nullable=True)

    # Feedback on recommendations
    medications_appropriate = db.Column(db.Boolean, nullable=True)
    medication_notes = db.Column(db.Text, nullable=True)
    tests_appropriate = db.Column(db.Boolean, nullable=True)
    test_notes = db.Column(db.Text, nullable=True)

    # Overall review
    overall_rating = db.Column(db.Integer, nullable=True)  # 1 to 5
    clinical_notes = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    case = db.relationship('Case', foreign_keys=[case_id], backref=db.backref('doctor_feedbacks', lazy='dynamic', cascade='all, delete-orphan'))
    doctor = db.relationship('User', foreign_keys=[doctor_id], backref=db.backref('feedbacks_given', lazy='dynamic'))

    def __repr__(self):
        return f"<DoctorFeedback id={self.id} case_id={self.case_id} doctor_id={self.doctor_id} correct={self.ai_diagnosis_correct}>"

    def to_dict(self):
        return {
            "id": self.id,
            "case_id": self.case_id,
            "doctor_id": self.doctor_id,
            "ai_diagnosis_correct": self.ai_diagnosis_correct,
            "corrected_diagnosis": self.corrected_diagnosis,
            "severity_correct": self.severity_correct,
            "corrected_severity": self.corrected_severity,
            "medications_appropriate": self.medications_appropriate,
            "medication_notes": self.medication_notes,
            "tests_appropriate": self.tests_appropriate,
            "test_notes": self.test_notes,
            "overall_rating": self.overall_rating,
            "clinical_notes": self.clinical_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


