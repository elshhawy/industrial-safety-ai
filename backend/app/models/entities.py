from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class EventType(StrEnum):
    PERSON_DETECTED = "person_detected"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    PPE_VIOLATION = "ppe_violation"
    DANGER_ZONE_VIOLATION = "danger_zone_violation"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(StrEnum):
    MISSING_HELMET = "missing_helmet"
    MISSING_VEST = "missing_vest"
    MISSING_GLOVES = "missing_gloves"
    MISSING_GOGGLES = "missing_goggles"
    MISSING_PPE = "missing_ppe"
    DANGER_ZONE_ENTRY_WITHOUT_PPE = "danger_zone_entry_without_ppe"
    DANGER_ZONE_UNAUTHORIZED = "danger_zone_unauthorized"
    SAFETY_RULE_VIOLATION = "safety_rule_violation"


class ViolationStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class AlertStatus(StrEnum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


def enum_column(enum: type[StrEnum], name: str) -> Enum:
    return Enum(
        enum,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda x: [e.value for e in x],
    )


class SafetyEvent(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    event_type: Mapped[EventType] = mapped_column(enum_column(EventType, "event_type"), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    person_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    zone_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_violation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    violation_type: Mapped[ViolationType | None] = mapped_column(enum_column(ViolationType, "event_violation_type"), nullable=True)
    severity: Mapped[Severity | None] = mapped_column(enum_column(Severity, "event_severity"), nullable=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    ppe_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    frame_reference: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    violation: Mapped["Violation | None"] = relationship("Violation", back_populates="event", cascade="all, delete-orphan", uselist=False)


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    event_id: Mapped[UUID] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), unique=True, nullable=False)
    violation_type: Mapped[ViolationType] = mapped_column(enum_column(ViolationType, "violation_type_record"), nullable=False, index=True)
    severity: Mapped[Severity] = mapped_column(enum_column(Severity, "violation_severity_record"), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[ViolationStatus] = mapped_column(enum_column(ViolationStatus, "violation_status"), server_default=ViolationStatus.OPEN, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    event: Mapped["SafetyEvent"] = relationship("SafetyEvent", back_populates="violation")
    alert: Mapped["Alert | None"] = relationship("Alert", back_populates="violation", cascade="all, delete-orphan", uselist=False)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    violation_id: Mapped[UUID] = mapped_column(ForeignKey("violations.id", ondelete="CASCADE"), unique=True, nullable=False)
    message: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[AlertStatus] = mapped_column(enum_column(AlertStatus, "alert_status"), server_default=AlertStatus.ACTIVE, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    violation: Mapped["Violation"] = relationship("Violation", back_populates="alert")
