"""
Task definitions for the consistency benchmark.

Each task is a natural language instruction that requires multi-step tool calling.
Tasks are categorized by type and difficulty.
"""

TASKS = [
    # ============================================================
    # Category 1: Data Retrieval & Aggregation
    # ============================================================
    {
        "id": "retrieve-001",
        "category": "retrieval",
        "difficulty": "easy",
        "task": "Find Alice's email address and send her a message saying 'Meeting moved to 3pm tomorrow'.",
        "expected_tools": ["get_contact", "send_email"],
        "min_steps": 2,
    },
    {
        "id": "retrieve-002",
        "category": "retrieval",
        "difficulty": "medium",
        "task": "Find all contacts at StartupXYZ and send each of them an email inviting them to a demo on March 10th at 2pm.",
        "expected_tools": ["search_contacts", "send_email"],
        "min_steps": 3,
    },
    {
        "id": "retrieve-003",
        "category": "retrieval",
        "difficulty": "medium",
        "task": "Search my emails for anything about 'budget' and summarize the key financial figures mentioned.",
        "expected_tools": ["search_emails"],
        "min_steps": 1,
    },
    {
        "id": "retrieve-004",
        "category": "retrieval",
        "difficulty": "hard",
        "task": "Find all emails from the last week, identify any that mention dollar amounts, and calculate the total dollar amounts mentioned across all emails.",
        "expected_tools": ["search_emails", "calculate"],
        "min_steps": 2,
    },

    # ============================================================
    # Category 2: Planning & Scheduling
    # ============================================================
    {
        "id": "schedule-001",
        "category": "scheduling",
        "difficulty": "easy",
        "task": "Schedule a 30-minute meeting with Bob on March 2nd at 2pm titled 'Design Review'.",
        "expected_tools": ["create_calendar_event"],
        "min_steps": 1,
    },
    {
        "id": "schedule-002",
        "category": "scheduling",
        "difficulty": "medium",
        "task": "Check my calendar for March 3rd and find a free 1-hour slot between 9am and 5pm for a meeting with Eve.",
        "expected_tools": ["list_calendar_events", "create_calendar_event"],
        "min_steps": 2,
    },
    {
        "id": "schedule-003",
        "category": "scheduling",
        "difficulty": "hard",
        "task": "Look at my calendar for the entire week of March 1-5. Find the day with the most free time and schedule a 2-hour 'Strategy Session' with Eve and Frank on that day.",
        "expected_tools": ["list_calendar_events", "create_calendar_event"],
        "min_steps": 2,
    },
    {
        "id": "schedule-004",
        "category": "scheduling",
        "difficulty": "hard",
        "task": "Check if there are any scheduling conflicts on March 3rd. If there are overlapping events, send an email to all affected attendees notifying them of the conflict.",
        "expected_tools": ["list_calendar_events", "search_contacts", "send_email"],
        "min_steps": 3,
    },

    # ============================================================
    # Category 3: Data Transformation & Computation
    # ============================================================
    {
        "id": "compute-001",
        "category": "computation",
        "difficulty": "easy",
        "task": "What is the total value of all electronics products in stock? (price × stock for each electronics item, then sum)",
        "expected_tools": ["search_products", "calculate"],
        "min_steps": 2,
    },
    {
        "id": "compute-002",
        "category": "computation",
        "difficulty": "medium",
        "task": "Find all products under $50 and calculate what the total revenue would be if we sold 50% of their current stock.",
        "expected_tools": ["search_products", "calculate"],
        "min_steps": 2,
    },
    {
        "id": "compute-003",
        "category": "computation",
        "difficulty": "hard",
        "task": "Calculate the total inventory value for each product category (electronics, furniture, office_supplies) and determine which category has the highest total value.",
        "expected_tools": ["search_products", "calculate"],
        "min_steps": 4,
    },

    # ============================================================
    # Category 4: Multi-Tool Composition
    # ============================================================
    {
        "id": "compose-001",
        "category": "composition",
        "difficulty": "medium",
        "task": "Find the email about 'Acme Corp', look up Dave's contact info (he sent it), and schedule a 1-hour 'Acme Demo Prep' meeting with him on March 4th at 3pm.",
        "expected_tools": ["search_emails", "get_contact", "create_calendar_event"],
        "min_steps": 3,
    },
    {
        "id": "compose-002",
        "category": "composition",
        "difficulty": "hard",
        "task": "Check the weather in San Francisco and New York. If either city is below 40°F, send an email to Alice warning about cold weather for her upcoming trip. Also check my calendar for March 1st and include any scheduled events in the email.",
        "expected_tools": ["get_weather", "get_contact", "list_calendar_events", "send_email"],
        "min_steps": 4,
    },
    {
        "id": "compose-003",
        "category": "composition",
        "difficulty": "hard",
        "task": "Search for the email about marketing campaign results. Extract the key metrics, calculate the total marketing spend (signups × cost per acquisition), and create a note summarizing the findings.",
        "expected_tools": ["search_emails", "calculate", "create_note"],
        "min_steps": 3,
    },
    {
        "id": "compose-004",
        "category": "composition",
        "difficulty": "hard",
        "task": "I need to prepare for the board meeting. Find the email from Eve about board deck updates, check when the board meeting is on my calendar, look up all attendees' contact info, and send each attendee a reminder email with the meeting details.",
        "expected_tools": ["search_emails", "list_calendar_events", "get_contact", "send_email"],
        "min_steps": 5,
    },

    # ============================================================
    # Category 5: Ambiguous / Underspecified Tasks
    # ============================================================
    {
        "id": "ambig-001",
        "category": "ambiguous",
        "difficulty": "medium",
        "task": "Help me prepare for my meetings tomorrow.",
        "expected_tools": ["list_calendar_events"],
        "min_steps": 1,
    },
    {
        "id": "ambig-002",
        "category": "ambiguous",
        "difficulty": "hard",
        "task": "I need to follow up on important things from this week.",
        "expected_tools": ["search_emails", "list_calendar_events"],
        "min_steps": 1,
    },
    {
        "id": "ambig-003",
        "category": "ambiguous",
        "difficulty": "hard",
        "task": "Get me ready for the investor call.",
        "expected_tools": ["search_emails", "list_calendar_events", "search_contacts"],
        "min_steps": 2,
    },
    {
        "id": "ambig-004",
        "category": "ambiguous",
        "difficulty": "hard",
        "task": "What should I focus on this week?",
        "expected_tools": ["list_calendar_events", "search_emails"],
        "min_steps": 1,
    },
]


def get_tasks(category: str = None, difficulty: str = None) -> list[dict]:
    """Filter tasks by category and/or difficulty."""
    tasks = TASKS
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if difficulty:
        tasks = [t for t in tasks if t["difficulty"] == difficulty]
    return tasks
