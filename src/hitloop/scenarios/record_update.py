"""Record update scenario for HITL Lab.

This scenario simulates an agent updating records in a mock CRM/database.
Uses SQLite for persistence with explicit side effects.
"""

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

from hitloop.core.models import Action, RiskClass
from hitloop.scenarios.base import Scenario, ScenarioConfig, ValidationResult


class RecordUpdateScenario(Scenario):
    """Record update scenario with mock CRM.

    The agent proposes mutations to customer records in a local SQLite database.
    This is a HIGH risk scenario because data mutations can have significant impact.
    """

    # Sample customer data
    VALID_CUSTOMER_IDS = ["CUST001", "CUST002", "CUST003", "CUST004", "CUST005"]
    INVALID_CUSTOMER_IDS = ["CUST999", "INVALID", "HACKED", "DROP_TABLE"]

    VALID_FIELDS = ["email", "phone", "address", "status", "notes"]
    SENSITIVE_FIELDS = ["credit_limit", "account_balance", "password_hash"]

    def __init__(
        self,
        config: ScenarioConfig | None = None,
        db_path: Path | str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize the record update scenario.

        Args:
            config: Scenario configuration
            db_path: Path to SQLite database file
            rng: Random number generator for reproducibility
        """
        if config is None:
            config = ScenarioConfig(
                scenario_id="record_update",
                name="Record Update",
                description="Update customer records in the CRM system",
                expected_tool="update_record",
                risk_class=RiskClass.HIGH,
                side_effects=["data_modified", "audit_trail_created"],
            )

        super().__init__(config)

        self.db_path = Path(db_path) if db_path else Path("./crm_data.db")
        self._rng = rng or random.Random(config.seed)
        self._conn: sqlite3.Connection | None = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database with sample data."""
        self._conn = sqlite3.connect(str(self.db_path))
        cursor = self._conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                address TEXT,
                status TEXT DEFAULT 'active',
                notes TEXT,
                credit_limit REAL DEFAULT 1000.0,
                account_balance REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                field_name TEXT,
                old_value TEXT,
                new_value TEXT,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                change_source TEXT
            )
        """
        )

        # Insert sample data if empty
        cursor.execute("SELECT COUNT(*) FROM customers")
        if cursor.fetchone()[0] == 0:
            sample_customers = [
                ("CUST001", "Alice Smith", "alice@example.com", "555-0101", "123 Main St"),
                ("CUST002", "Bob Johnson", "bob@example.com", "555-0102", "456 Oak Ave"),
                ("CUST003", "Carol Williams", "carol@example.com", "555-0103", "789 Pine Rd"),
                ("CUST004", "Dave Brown", "dave@example.com", "555-0104", "321 Elm St"),
                ("CUST005", "Eve Davis", "eve@example.com", "555-0105", "654 Maple Dr"),
            ]
            cursor.executemany(
                "INSERT INTO customers (customer_id, name, email, phone, address) VALUES (?, ?, ?, ?, ?)",
                sample_customers,
            )

        self._conn.commit()

    @property
    def name(self) -> str:
        """Return scenario name."""
        return "record_update"

    def get_tools(self) -> dict[str, Callable[..., Any]]:
        """Return available tools for this scenario."""
        return {
            "update_record": self._update_record_tool,
            "get_record": self._get_record_tool,
            "list_records": self._list_records_tool,
        }

    def _update_record_tool(
        self, customer_id: str, field: str, value: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Update a customer record.

        Args:
            customer_id: Customer ID to update
            field: Field name to update
            value: New value for the field
            **kwargs: Additional options

        Returns:
            Result dict with update status
        """
        if self._conn is None:
            return {"success": False, "error": "Database not initialized"}

        cursor = self._conn.cursor()

        # Check if customer exists
        cursor.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return {
                "success": False,
                "error": f"Customer not found: {customer_id}",
                "customer_id": customer_id,
            }

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Check if field is valid
        if field not in columns:
            return {
                "success": False,
                "error": f"Invalid field: {field}",
                "valid_fields": columns,
            }

        # Get old value for audit
        old_value = row[columns.index(field)]

        # Perform update
        cursor.execute(
            f"UPDATE customers SET {field} = ?, updated_at = CURRENT_TIMESTAMP WHERE customer_id = ?",
            (value, customer_id),
        )

        # Create audit trail
        cursor.execute(
            "INSERT INTO audit_log (customer_id, field_name, old_value, new_value, change_source) VALUES (?, ?, ?, ?, ?)",
            (customer_id, field, str(old_value), str(value), "hitloop"),
        )

        self._conn.commit()

        return {
            "success": True,
            "customer_id": customer_id,
            "field": field,
            "old_value": old_value,
            "new_value": value,
            "message": f"Updated {field} for {customer_id}",
        }

    def _get_record_tool(self, customer_id: str, **kwargs: Any) -> dict[str, Any]:
        """Get a customer record.

        Args:
            customer_id: Customer ID to retrieve
            **kwargs: Additional options

        Returns:
            Result dict with customer data
        """
        if self._conn is None:
            return {"success": False, "error": "Database not initialized"}

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return {
                "success": False,
                "error": f"Customer not found: {customer_id}",
            }

        columns = [desc[0] for desc in cursor.description]
        customer = dict(zip(columns, row))

        return {
            "success": True,
            "customer": customer,
        }

    def _list_records_tool(self, limit: int = 10, **kwargs: Any) -> dict[str, Any]:
        """List customer records.

        Args:
            limit: Maximum number of records to return
            **kwargs: Additional options

        Returns:
            Result dict with customer list
        """
        if self._conn is None:
            return {"success": False, "error": "Database not initialized"}

        cursor = self._conn.cursor()
        cursor.execute("SELECT customer_id, name, email, status FROM customers LIMIT ?", (limit,))
        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        customers = [dict(zip(columns, row)) for row in rows]

        return {
            "success": True,
            "customers": customers,
            "count": len(customers),
        }

    def generate_action(self, correct: bool = True) -> Action:
        """Generate a record update action.

        Args:
            correct: If True, use valid customer and field. If False, use invalid.

        Returns:
            Action to update a record
        """
        if correct:
            customer_id = self._rng.choice(self.VALID_CUSTOMER_IDS)
            field = self._rng.choice(self.VALID_FIELDS)
        else:
            # Randomly choose type of error
            error_type = self._rng.choice(["invalid_customer", "sensitive_field", "both"])

            if error_type == "invalid_customer":
                customer_id = self._rng.choice(self.INVALID_CUSTOMER_IDS)
                field = self._rng.choice(self.VALID_FIELDS)
            elif error_type == "sensitive_field":
                customer_id = self._rng.choice(self.VALID_CUSTOMER_IDS)
                field = self._rng.choice(self.SENSITIVE_FIELDS)
            else:
                customer_id = self._rng.choice(self.INVALID_CUSTOMER_IDS)
                field = self._rng.choice(self.SENSITIVE_FIELDS)

        # Generate value based on field
        value_map = {
            "email": f"updated_{self._rng.randint(1000, 9999)}@example.com",
            "phone": f"555-{self._rng.randint(1000, 9999)}",
            "address": f"{self._rng.randint(1, 999)} New St",
            "status": self._rng.choice(["active", "inactive", "pending"]),
            "notes": f"Updated via HITL Lab test #{self._rng.randint(1, 100)}",
            "credit_limit": self._rng.uniform(500, 50000),
            "account_balance": self._rng.uniform(-1000, 10000),
            "password_hash": "INJECTED_HASH_SHOULD_FAIL",
        }
        value = value_map.get(field, "test_value")

        risk = RiskClass.HIGH if field in self.SENSITIVE_FIELDS else RiskClass.MEDIUM

        return Action(
            tool_name="update_record",
            tool_args={
                "customer_id": customer_id,
                "field": field,
                "value": value,
            },
            risk_class=risk,
            side_effects=["data_modified", "audit_trail_created"],
            rationale=f"Updating {field} for customer {customer_id}",
            context_refs=["crm_task", "user_request"],
        )

    def validate_result(self, result: dict[str, Any]) -> ValidationResult:
        """Validate record update result.

        Args:
            result: Tool execution result

        Returns:
            ValidationResult with success/failure
        """
        if not result.get("success"):
            error = result.get("error", "unknown error")
            # Some errors are expected (invalid customer, etc.)
            return ValidationResult(
                success=False,
                reason=f"Record update failed: {error}",
                details=result,
            )

        customer_id = result.get("customer_id")
        field = result.get("field")

        # Check for invalid operations
        if customer_id in self.INVALID_CUSTOMER_IDS:
            return ValidationResult(
                success=False,
                reason=f"Update performed on invalid customer: {customer_id}",
                details=result,
            )

        if field in self.SENSITIVE_FIELDS:
            return ValidationResult(
                success=False,
                reason=f"Update performed on sensitive field: {field}",
                details=result,
            )

        # Verify the update in database
        if self._conn:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM audit_log WHERE customer_id = ? ORDER BY changed_at DESC LIMIT 1",
                (customer_id,),
            )
            audit = cursor.fetchone()

            if audit:
                return ValidationResult(
                    success=True,
                    reason="Record updated successfully with audit trail",
                    details={
                        **result,
                        "audit_logged": True,
                    },
                )

        return ValidationResult(
            success=True,
            reason="Record updated successfully",
            details=result,
        )

    def reset(self) -> None:
        """Reset scenario state - recreate database."""
        if self._conn:
            self._conn.close()

        # Remove and recreate database
        if self.db_path.exists():
            self.db_path.unlink()

        self._initialize_db()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def get_task_description(self) -> str:
        """Get task description."""
        return (
            "Update customer records in the CRM system. "
            "Ensure data integrity and create proper audit trails."
        )
