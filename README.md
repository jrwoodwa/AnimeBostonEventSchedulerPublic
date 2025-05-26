# 🗓 Anime Boston Scheduler Planner
Scrapes the Anime Boston 2025 schedule using pandas/beautifulsoup and optimizes event choices using linear programming (LP).

Features:
- Extracts full event grid with time, room, category, and color
- Filters low-utility events and early low-value ones

Solves the schedule as an assignment problem via LP optimization
 
# 🔁 Nice to haves for next year
- Refined scoring: Update utility ratings now that we know which events over‑ or under‑deliver.
- Cross‑building travel time: Model in security checks and walks between Hynes vs. Sheraton to avoid tight transfers.
- Daily shopping slot: Reserve an hour each day for Artist Alley and Dealers’ Hall (respectively), with enjoyment modeled to drop when it’s predicted to be crowded. But pick up on first day (before stock outs) or last day for closing (discount season!).

----
# ✅ What's done:
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
