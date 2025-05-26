🗓 Anime Boston Scheduler Planner
Scrapes the Anime Boston 2025 schedule using pandas/beautifulsoup and optimizes event choices using linear programming.

Features:
- Extracts full event grid with time, room, category, and color
- Filters low-utility events and early low-value ones

Solves the schedule as an assignment problem via LP

•	Optimization is figuring out how to get an objective given constraints and decisions.
•	Decisions are the actions to take toward an objective within constraints. 
o	x[event] is a yes/no decision variable we want to solve: 
	1 = schedule event
	0 = not to schedule event
•	An objective is a goal to minimize or maximize (e.g., maximize utility/enjoyment given events selected)
•	Constraints are what the rules are. For example:
o	Focus on interesting events only.
	Not considering events below a Utility threshold, say, 5
o	Arrive at 10 am, or earlier for a major event.
	Arrive at 9:45 am, or earlier if it's a great morning event, say 8:00 am (if the utility score >= 7 to justify less sleep)
o	No double-booking events
	Pick only one event per 15-minute timeslot.
	Locks in until the end (currently no partials, but we could leave 15 mins early if we want, for example)
o	Only one meal for lunch and one meal for dinner
	I assumed flexible 45-minute meal 'events' between, say, 11:30 am and 1:30 pm; 5:00 pm and 7:30 pm)


----
✅ What's done:
- ✅ Completed the web scraping in a notebook that captures the data, processes it (including subevents like Maid Cafe going into 45-min block options), and tidies it.
- ✅ Write a `WebScrape.py` script for translating data engineering into a CSV (optionally: store timestamp when the script queries data).
- ✅ After the CSV is fully ready, manually utility score the information.
- ✅ With the data prepped, write the LP mathematical model in the notebook.
- ✅ After the LP model is defined, write the code.
- ✅ Observe results in a displayed itinerary.
- ✅ Standardize code in `MathOptModel.py`, visualize in itineraries.
- ✅ Circle actual schedules (+with backup events)
- ✅ Make refinements to the LP model/visualizations/etc based on feedback from party members.
- ✅ Finalize actual schedules with party (+with backup events)
