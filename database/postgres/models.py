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

