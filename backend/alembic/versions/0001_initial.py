"""Initial industrial safety events, violations and alerts schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("event_id", sa.String(128), nullable=False),
        sa.Column(
            "event_type",
            sa.Enum(
                "person_detected",
                "zone_entry",
                "zone_exit",
                "ppe_violation",
                "danger_zone_violation",
                name="event_type",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("track_id", sa.String(64), nullable=True),
        sa.Column("person_id", sa.String(64), nullable=True),
        sa.Column("zone_id", sa.String(64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_violation", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "violation_type",
            sa.Enum(
                "missing_helmet",
                "missing_vest",
                "missing_gloves",
                "missing_goggles",
                "missing_ppe",
                "danger_zone_entry_without_ppe",
                "danger_zone_unauthorized",
                "safety_rule_violation",
                name="event_violation_type",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=True,
        ),
        sa.Column(
            "severity",
            sa.Enum(
                "low",
                "medium",
                "high",
                "critical",
                name="event_severity",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=True,
        ),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("ppe_details", sa.JSON(), nullable=True),
        sa.Column("frame_reference", sa.String(512), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("event_id", name="events_event_id_key"),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="confidence_range"),
    )
    op.create_index("ix_events_event_type", "events", ["event_type"])
    op.create_index("ix_events_occurred_at", "events", ["occurred_at"])
    op.create_index("ix_events_camera_id", "events", ["camera_id"])
    op.create_index("ix_events_is_violation", "events", ["is_violation"])

    op.create_table(
        "violations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("event_id", sa.Uuid(), sa.ForeignKey("events.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "violation_type",
            sa.Enum(
                "missing_helmet",
                "missing_vest",
                "missing_gloves",
                "missing_goggles",
                "missing_ppe",
                "danger_zone_entry_without_ppe",
                "danger_zone_unauthorized",
                "safety_rule_violation",
                name="violation_type_record",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "severity",
            sa.Enum(
                "low",
                "medium",
                "high",
                "critical",
                name="violation_severity_record",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "open",
                "resolved",
                name="violation_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
            server_default="open",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("event_id", name="violations_event_id_key"),
    )
    op.create_index("ix_violations_violation_type", "violations", ["violation_type"])
    op.create_index("ix_violations_severity", "violations", ["severity"])
    op.create_index("ix_violations_camera_id", "violations", ["camera_id"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("violation_id", sa.Uuid(), sa.ForeignKey("violations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("message", sa.String(512), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "active",
                "acknowledged",
                "resolved",
                name="alert_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
            server_default="active",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("violation_id", name="alerts_violation_id_key"),
    )


def downgrade() -> None:
    op.drop_table("alerts")
    op.drop_table("violations")
    op.drop_table("events")
