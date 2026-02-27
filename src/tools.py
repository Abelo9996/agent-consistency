"""
Tool definitions for the agent consistency benchmark.

We define a set of simulated tools that agents can call. Tools are deterministic
(same input → same output) so any variance comes from the LLM, not the tools.
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Any


# ============================================================
# Tool Registry
# ============================================================

TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, parameters: dict):
    """Decorator to register a tool function and its schema."""
    def decorator(fn):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": fn,
        }
        return fn
    return decorator


def get_tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TOOLS.values()
    ]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return deterministic result."""
    if name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOLS[name]["function"](**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================
# Deterministic Data Store (seeded, reproducible)
# ============================================================

# Simulated database of contacts
CONTACTS = {
    "alice": {"name": "Alice Chen", "email": "alice@example.com", "phone": "+1-555-0101", "role": "Engineering Manager", "company": "TechCorp"},
    "bob": {"name": "Bob Smith", "email": "bob@example.com", "phone": "+1-555-0102", "role": "Product Designer", "company": "DesignLab"},
    "carol": {"name": "Carol Davis", "email": "carol@example.com", "phone": "+1-555-0103", "role": "Data Scientist", "company": "DataInc"},
    "dave": {"name": "Dave Wilson", "email": "dave@example.com", "phone": "+1-555-0104", "role": "Sales Director", "company": "SalesCo"},
    "eve": {"name": "Eve Martinez", "email": "eve@example.com", "phone": "+1-555-0105", "role": "CEO", "company": "StartupXYZ"},
    "frank": {"name": "Frank Lee", "email": "frank@example.com", "phone": "+1-555-0106", "role": "CTO", "company": "StartupXYZ"},
    "grace": {"name": "Grace Kim", "email": "grace@example.com", "phone": "+1-555-0107", "role": "Marketing Lead", "company": "BrandCo"},
    "henry": {"name": "Henry Brown", "email": "henry@example.com", "phone": "+1-555-0108", "role": "Backend Engineer", "company": "TechCorp"},
}

# Simulated calendar
CALENDAR = [
    {"id": "evt-001", "title": "Team Standup", "date": "2026-03-01", "start": "09:00", "end": "09:30", "attendees": ["alice", "henry"]},
    {"id": "evt-002", "title": "Product Review", "date": "2026-03-01", "start": "14:00", "end": "15:00", "attendees": ["alice", "bob", "carol"]},
    {"id": "evt-003", "title": "1:1 with Dave", "date": "2026-03-02", "start": "10:00", "end": "10:30", "attendees": ["dave"]},
    {"id": "evt-004", "title": "Board Meeting", "date": "2026-03-03", "start": "13:00", "end": "15:00", "attendees": ["eve", "frank"]},
    {"id": "evt-005", "title": "Design Sprint", "date": "2026-03-03", "start": "09:00", "end": "12:00", "attendees": ["bob", "grace"]},
    {"id": "evt-006", "title": "Lunch with Carol", "date": "2026-03-04", "start": "12:00", "end": "13:00", "attendees": ["carol"]},
    {"id": "evt-007", "title": "Investor Call", "date": "2026-03-05", "start": "16:00", "end": "17:00", "attendees": ["eve", "frank", "dave"]},
    {"id": "evt-008", "title": "Sprint Planning", "date": "2026-03-05", "start": "09:00", "end": "10:30", "attendees": ["alice", "henry", "bob"]},
]

# Simulated products/inventory
PRODUCTS = [
    {"id": "prod-001", "name": "Widget A", "price": 29.99, "stock": 150, "category": "electronics"},
    {"id": "prod-002", "name": "Widget B", "price": 49.99, "stock": 75, "category": "electronics"},
    {"id": "prod-003", "name": "Gadget Pro", "price": 199.99, "stock": 30, "category": "electronics"},
    {"id": "prod-004", "name": "Office Chair", "price": 299.99, "stock": 20, "category": "furniture"},
    {"id": "prod-005", "name": "Standing Desk", "price": 599.99, "stock": 10, "category": "furniture"},
    {"id": "prod-006", "name": "Notebook Pack", "price": 12.99, "stock": 500, "category": "office_supplies"},
    {"id": "prod-007", "name": "Pen Set", "price": 8.99, "stock": 300, "category": "office_supplies"},
    {"id": "prod-008", "name": "Monitor 27\"", "price": 349.99, "stock": 45, "category": "electronics"},
]

# Simulated emails
EMAILS = [
    {"id": "email-001", "from": "alice@example.com", "to": "me@example.com", "subject": "Q1 Budget Review", "date": "2026-02-20", "body": "Hi, please review the Q1 budget spreadsheet and send your comments by Friday. Key items: engineering headcount (+2), cloud costs ($45K/mo), and marketing budget ($20K)."},
    {"id": "email-002", "from": "dave@example.com", "to": "me@example.com", "subject": "New Lead: Acme Corp", "date": "2026-02-22", "body": "Great news! Acme Corp wants a demo next week. They're interested in our enterprise plan. Revenue potential: $50K ARR. Contact: John Doe (john@acme.com)."},
    {"id": "email-003", "from": "eve@example.com", "to": "me@example.com", "subject": "Board Deck Updates", "date": "2026-02-24", "body": "Please update slides 3-7 with latest metrics. Board meeting is March 3rd. Need: MRR ($125K), churn rate (3.2%), and pipeline ($400K)."},
    {"id": "email-004", "from": "grace@example.com", "to": "me@example.com", "subject": "Marketing Campaign Results", "date": "2026-02-25", "body": "February campaign results: 2,500 signups, 180 trials, 12 conversions. Cost per acquisition: $167. Top channel: LinkedIn (45%), Google Ads (30%), organic (25%)."},
    {"id": "email-005", "from": "bob@example.com", "to": "me@example.com", "subject": "Design System v2", "date": "2026-02-25", "body": "Design system v2 is ready for review. Major changes: new color palette, updated typography, and 15 new components. Preview link: https://design.example.com/v2"},
]

# Simulated weather data
WEATHER = {
    "san_francisco": {"temp_f": 58, "condition": "Partly Cloudy", "humidity": 72, "wind_mph": 12},
    "new_york": {"temp_f": 35, "condition": "Snow", "humidity": 85, "wind_mph": 20},
    "los_angeles": {"temp_f": 72, "condition": "Sunny", "humidity": 40, "wind_mph": 5},
    "chicago": {"temp_f": 28, "condition": "Cloudy", "humidity": 60, "wind_mph": 18},
    "seattle": {"temp_f": 45, "condition": "Rain", "humidity": 90, "wind_mph": 8},
    "austin": {"temp_f": 68, "condition": "Sunny", "humidity": 55, "wind_mph": 10},
}


# ============================================================
# Tool Implementations
# ============================================================

@tool(
    name="search_contacts",
    description="Search contacts by name, role, or company. Returns matching contacts.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (name, role, or company)"},
        },
        "required": ["query"],
    },
)
def search_contacts(query: str) -> dict:
    query_lower = query.lower()
    matches = [
        c for c in CONTACTS.values()
        if query_lower in c["name"].lower()
        or query_lower in c["role"].lower()
        or query_lower in c["company"].lower()
    ]
    return {"results": matches, "count": len(matches)}


@tool(
    name="get_contact",
    description="Get a specific contact by their first name (lowercase).",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "First name (lowercase), e.g. 'alice'"},
        },
        "required": ["name"],
    },
)
def get_contact(name: str) -> dict:
    contact = CONTACTS.get(name.lower())
    if contact:
        return contact
    return {"error": f"Contact '{name}' not found"}


@tool(
    name="list_calendar_events",
    description="List calendar events, optionally filtered by date range.",
    parameters={
        "type": "object",
        "properties": {
            "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
            "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
        },
        "required": [],
    },
)
def list_calendar_events(start_date: str = None, end_date: str = None) -> dict:
    events = CALENDAR
    if start_date:
        events = [e for e in events if e["date"] >= start_date]
    if end_date:
        events = [e for e in events if e["date"] <= end_date]
    return {"events": events, "count": len(events)}


@tool(
    name="create_calendar_event",
    description="Create a new calendar event.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
            "start_time": {"type": "string", "description": "Start time (HH:MM)"},
            "end_time": {"type": "string", "description": "End time (HH:MM)"},
            "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee names"},
        },
        "required": ["title", "date", "start_time", "end_time"],
    },
)
def create_calendar_event(title: str, date: str, start_time: str, end_time: str, attendees: list = None) -> dict:
    event_id = "evt-" + hashlib.md5(f"{title}{date}{start_time}".encode()).hexdigest()[:6]
    return {
        "id": event_id,
        "title": title,
        "date": date,
        "start": start_time,
        "end": end_time,
        "attendees": attendees or [],
        "status": "created",
    }


@tool(
    name="send_email",
    description="Send an email to a recipient.",
    parameters={
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"},
        },
        "required": ["to", "subject", "body"],
    },
)
def send_email(to: str, subject: str, body: str) -> dict:
    email_id = "sent-" + hashlib.md5(f"{to}{subject}".encode()).hexdigest()[:6]
    return {"id": email_id, "status": "sent", "to": to, "subject": subject}


@tool(
    name="search_emails",
    description="Search inbox emails by keyword in subject or body.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search keyword"},
        },
        "required": ["query"],
    },
)
def search_emails(query: str) -> dict:
    query_lower = query.lower()
    matches = [
        e for e in EMAILS
        if query_lower in e["subject"].lower() or query_lower in e["body"].lower()
    ]
    return {"results": matches, "count": len(matches)}


@tool(
    name="search_products",
    description="Search products by name or category.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (name or category)"},
        },
        "required": ["query"],
    },
)
def search_products(query: str) -> dict:
    query_lower = query.lower()
    matches = [
        p for p in PRODUCTS
        if query_lower in p["name"].lower() or query_lower in p["category"].lower()
    ]
    return {"results": matches, "count": len(matches)}


@tool(
    name="calculate",
    description="Evaluate a mathematical expression. Supports +, -, *, /, **, %, and parentheses.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '(29.99 * 150) + (49.99 * 75)'"},
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> dict:
    # Safe eval with only math operations
    allowed = set("0123456789.+-*/%(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": round(float(result), 2)}
    except Exception as e:
        return {"error": str(e)}


@tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name (e.g. 'san_francisco', 'new_york')"},
        },
        "required": ["city"],
    },
)
def get_weather(city: str) -> dict:
    city_key = city.lower().replace(" ", "_")
    weather = WEATHER.get(city_key)
    if weather:
        return {"city": city, **weather}
    return {"error": f"Weather data not available for '{city}'"}


@tool(
    name="create_note",
    description="Create a note/memo with a title and content.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title"},
            "content": {"type": "string", "description": "Note content"},
        },
        "required": ["title", "content"],
    },
)
def create_note(title: str, content: str) -> dict:
    note_id = "note-" + hashlib.md5(f"{title}".encode()).hexdigest()[:6]
    return {"id": note_id, "title": title, "status": "created"}
