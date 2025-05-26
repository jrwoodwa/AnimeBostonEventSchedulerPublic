import numpy as np
import pandas as pd
import ast
import pickle
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint, Binary, value
)
from pyomo.opt import SolverFactory
from collections import namedtuple
from datetime import datetime, timedelta
import imgkit



# processing problem

# --- split event into overlapping subevents ---
def split_event_to_overlapping_subevents(df, event_col, time_col, room_col, chunk_size=3, target_event=None):
    """
    Slide over time slots to create overlapping subevents.
    """
    if target_event:
        df = df[df[event_col] == target_event]

    exploded_df = (df.explode(time_col)
                   .sort_values([room_col, time_col])
                   .reset_index(drop=True))

    result_rows = []

    for (room, event), group in exploded_df.groupby([room_col, event_col]):
        times = list(group[time_col])
        for i in range(len(times) - chunk_size + 1):
            sub_times = times[i:i + chunk_size]
            sub_label = chr(65 + i)  # A, B, C ...
            event_id = f"{event} {sub_label}"
            result_rows.append({
                event_col: event,
                'Subevent': sub_label,
                'EventID': event_id,
                room_col: room,
                time_col: sub_times,
                'Category': group['Category'].iloc[0],
                'Color': group['Color'].iloc[0],
                'Utility': group['Utility'].iloc[0]
            })

    return pd.DataFrame(result_rows)


# --- generate meal subevents ---
def generate_meal_subevents_multiple(meal_windows, chunk_size=3, color='#FFFF00', room='Meal', date=pd.Timestamp.today()):
    """
    Generate meal time chunks for given windows.
    """
    all_rows = []

    for meal_name, start, end in meal_windows:
        dt_start = pd.Timestamp.combine(date.date(), start.time())
        dt_end = pd.Timestamp.combine(date.date(), end.time())

        full_slots = pd.date_range(start=dt_start, end=dt_end, freq='15min')

        chunks = [full_slots[i:i + chunk_size] for i in range(len(full_slots) - chunk_size + 1)]

        for i, chunk in enumerate(chunks):
            time_only = [t.strftime('%H:%M') for t in chunk]
            all_rows.append({
                'Event': f'{meal_name.upper()}',
                'EventID': f'{meal_name} {chr(65 + i)}',
                'Subevent': f'{time_only[0]}-{time_only[-1]}',
                'Category': meal_name,
                'Color': color,
                'Room': room,
                'TimeSlot': time_only,
                'start_time': time_only[0],
                'end_time': time_only[-1]
            })

    return pd.DataFrame(all_rows)


# --- explode timeslots to find overlapping events ---
def explode_timeslots(df, sorting_columns=['Room', 'Event'], agg_column='EventID'):
    """
    Explode time slots and group overlapping events.
    """
    df_exploded = df.explode('TimeSlot').sort_values(sorting_columns).reset_index(drop=True)

    df_grouped = (
        df_exploded
        .groupby('TimeSlot')[[agg_column]]
        .agg(set)
        .reset_index()
        .assign(
            Event_tuple=lambda d: d[agg_column].apply(tuple),
            Event_tuple_length=lambda d: d['Event_tuple'].apply(len)
        )
    )

    df_dedup = (
        df_grouped
        .drop_duplicates(subset='Event_tuple')
        .drop(columns='Event_tuple')
        .reset_index(drop=True)
    )

    df_constraints = (
        df_dedup[df_dedup['Event_tuple_length'] > 1]
        .reset_index(drop=True)
        .rename(columns={agg_column: f'Overlapping{agg_column}s'})
    )

    return df_constraints


# --- main processing for each day ---
def process_day(day_index, filename_prefix='AnimeBoston_day', meal_windows=None, mealtimeslots = 4, timeslots_per_splitting=3, U_threshold=5, U_min=7,
                desired_arrival_time_str='09:45'):
    """
    Process a single day's schedule and export pkl files.
    """

    # Load data
    df_raw = pd.read_csv(f"{filename_prefix}{day_index}_schedule.csv", 
                         encoding='utf-8', 
                         converters={'TimeSlot': ast.literal_eval})

    # Filter by utility threshold
    df = df_raw[df_raw['Utility'] > U_threshold]

    # # Split overlapping subevents for a specific event
    amv_events = df[df['AMVs'] == True]['Event'].unique()
    
    subevent_dfs = []
    for event in amv_events:
        temp = split_event_to_overlapping_subevents(
            df, 'Event', 'TimeSlot', 'Room',
            chunk_size=timeslots_per_splitting,
            target_event=event
        )
        subevent_dfs.append(temp)

    subevent_df = pd.concat(subevent_dfs) if subevent_dfs else pd.DataFrame()
    
    df = df[~df['Event'].isin(amv_events)]

    # Generate meal subevents for this day
    date = pd.to_datetime('2025-05-23') + pd.Timedelta(days=day_index)  # base date + day offset
    df_meal_subevents = generate_meal_subevents_multiple(meal_windows, chunk_size=mealtimeslots, date=date)

    df_meal_subevents['Utility'] = 0.0  # meals have zero utility

    # Marking events
    df_final = df.copy()
    df_final['EventID'] = df_final.index

    # Combine all events
    df_final = pd.concat([df_final, subevent_df]) # AMV Contest Overflows
    df_final = pd.concat([df_final, df_meal_subevents]) # meal subevents

    # Extract start/end time strings
    df_final['start_time'] = df_final['TimeSlot'].str[0]
    df_final['end_time'] = df_final['TimeSlot'].str[-1]

    # Parse times to datetime
    df_final['start_dt'] = pd.to_datetime(df_final['start_time'], format='%H:%M')

    # Add day offset to times before 6 AM (next day)
    df_final['start_dt'] = df_final['start_dt'].apply(lambda t: t + pd.Timedelta(days=1) if t.hour < 6 else t)

    # Filter by desired arrival or min utility
    desired_arrival_time = pd.to_datetime(desired_arrival_time_str, format='%H:%M')
    df_final = df_final[
        (df_final['start_dt'] > desired_arrival_time) |
        ((df_final['start_dt'] <= desired_arrival_time) & (df_final['Utility'] >= U_min))
    ]

    # Sort and drop helper columns
    df_final = df_final.sort_values('start_dt').drop(columns=['start_dt'])
    # Prepare problem dataframe
    df_problem = df_final#[['EventID', 'TimeSlot', 'Utility']]

    # Prepare constraints dataframe
    df_problem_constraints = explode_timeslots(df_final)

    # Save outputs
    with open(f"df_problem_day{day_index}.pkl", "wb") as f:
        pickle.dump(df_problem, f)

    with open(f"df_problem_constraints_day{day_index}.pkl", "wb") as f:
        pickle.dump(df_problem_constraints, f)

    # Save meal constraints separately
    meal_constraints = df_meal_subevents.groupby('Category')['EventID'].apply(set)
    with open(f"df_problem_constraints_meals_day{day_index}.pkl", "wb") as f:
        pickle.dump(meal_constraints, f)

    return df_problem, df_problem_constraints, meal_constraints


# import timeit

# # Time original set-based
# time_orig = timeit.timeit(lambda: greedy_warm_start(event_dataframe), number=1000)

# # Time bitmap-based (assuming you have greedy_warm_start_bitmap_pandas)
# time_bitmap = timeit.timeit(lambda: greedy_warm_start_bitmap(event_dataframe), number=1000)

# print(f"Original iterrows time: {time_orig:.2f}s")
# print(f"Bitmap method time: {time_bitmap:.2f}s")

# # Original iterrows time: 2.40s
# # Bitmap method time: 1.26s

# def greedy_warm_start(df):
#     used_time_slots = set()
#     solution = {}
#     selected_ids = set()
#     df = df.copy()
#     df['GreedyScore'] = df['Utility'] * df['TimeSlot'].str.len()

#     # Force-select Lunch A and Dinner A
#     for meal in ['Lunch A', 'Dinner A']:
#         row = df[df['EventID'] == meal]
#         if not row.empty:
#             solution[meal] = 1
#             selected_ids.add(meal)
#             used_time_slots |= set(row.iloc[0]['TimeSlot'])

#     # Greedy selection for others (skip selected ones)
#     for _, row in df.sort_values('GreedyScore', ascending=False).iterrows():
#         event_id = row['EventID']
#         if event_id in selected_ids:
#             continue
#         time_slots = set(row['TimeSlot'])
#         if used_time_slots & time_slots:
#             solution[event_id] = 0
#         else:
#             solution[event_id] = 1
#             used_time_slots |= time_slots

#     return solution


def greedy_warm_start_bitmap(df, debug=False):
    """
    Efficient bit-wise warm start greedy heuristic... time slots are 15-minute increments and 24 hours is 96 (sometimes goes overnight events, but not every event uses up all the timeslots)

    Event 11
  Time slots: ['21:00', '21:15', '21:30', '21:45', '22:00', '22:15', '22:30', '22:45', '23:00', '23:15']
  Used mask : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  Bitmask   : 000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000
  Overlap   : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  → Selected

Event 122
  Time slots: ['13:30', '13:45', '14:00', '14:15', '15:00', '15:15', '15:30', '15:45', '19:30', '19:45', '20:00', '20:15', '21:00', '21:15', '21:30', '21:45']
  Used mask : 000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000
  Bitmask   : 000000000000000000000000000000000000000000001111001111000000000000111100111100000000000000000000
  Overlap   : 000000000000000000000000000000000000000000001111000000000000000000000000000000000000000000000000
  → Skipped

Event 123
  Time slots: ['14:00', '14:15', '14:30', '14:45', '15:30', '15:45', '16:00', '16:15', '20:00', '20:15', '20:30', '20:45', '21:30', '21:45', '22:00', '22:15']
  Used mask : 000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000
  Bitmask   : 000000000000000000000000000000000000000000111100111100000000000011110011110000000000000000000000
  Overlap   : 000000000000000000000000000000000000000000111100000000000000000000000000000000000000000000000000
  → Skipped

Event 22
  Time slots: ['20:00', '20:15', '20:30', '20:45', '21:00', '21:15', '21:30', '21:45']
  Used mask : 000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000
  Bitmask   : 000000000000000000000000000000000000000000001111111100000000000000000000000000000000000000000000
  Overlap   : 000000000000000000000000000000000000000000001111000000000000000000000000000000000000000000000000
  → Skipped

Event 108
  Time slots: ['10:00', '10:15', '10:30', '10:45', '11:00', '11:15']
  Used mask : 000000000000000000000000000000000000001111111111000000000000000000000000000000000000000000000000
  Bitmask   : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111000000
  Overlap   : 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
  → Selected
  .
  .
  .
    """
    # convert slots to bitmask
    def to_bitmask(slots):
        bitmask = 0
        for s in slots:
            bitmask |= 1 << slot_to_bit[s]
        return bitmask

    # map all unique slots to bit positions
    all_slots = np.concatenate(df['TimeSlot'].to_numpy())
    unique_slots = np.unique(all_slots).tolist()
    slot_to_bit = {slot: i for i, slot in enumerate(unique_slots)}

    # bitmask & greedy score
    df = df.copy()
    df['Bitmask'] = df['TimeSlot'].apply(to_bitmask)
    df['GreedyScore'] = df['Utility'] * df['TimeSlot'].str.len()
    df_sorted = df.sort_values('GreedyScore', ascending=False).reset_index(drop=True)

    used_slots = 0
    solution = {}
    bitmasks = df_sorted['Bitmask'].to_numpy(dtype=object)
    event_ids = df_sorted['EventID'].tolist()

    # helper: force one meal A, skip the rest
    def force_meal_selection(meal_name):
        ''' With a dict, force Lunch A and Dinner A as set, and everything else as not set
        '''
        target_id = f"{meal_name} A" # always 'Meal A'
        match = df_sorted['EventID'].str.startswith(meal_name)
        is_target = df_sorted['EventID'] == target_id
        bitmask = df_sorted.loc[is_target, 'Bitmask']

        if not bitmask.empty:
            nonlocal used_slots
            used_slots |= bitmask.iloc[0]
            solution[target_id] = 1
            if debug:
                print(f"Forced select: {target_id}")
        else:
            solution[target_id] = 0
            if debug:
                print(f"Warning: {target_id} not found")

        skipped = df_sorted.loc[match & ~is_target, 'EventID']
        solution.update({eid: 0 for eid in skipped})
        if debug:
            for eid in skipped:
                print(f"Forced skip: {eid}")

    # force meals
    force_meal_selection('Lunch')
    force_meal_selection('Dinner')

    # greedy selection
    for i, event_id in enumerate(event_ids):
        if event_id in solution:
            continue  # already handled

        event_mask = bitmasks[i]
        overlap = used_slots & event_mask

        if debug:
            print(f"\nEvent {event_id}")
            print(f"  Time slots: {df_sorted.at[i, 'TimeSlot']}")
            print(f"  Used mask : {used_slots:096b}")
            print(f"  Bitmask   : {event_mask:096b}")
            print(f"  Overlap   : {overlap:096b}")

        if overlap == 0:
            solution[event_id] = 1
            used_slots |= event_mask
            if debug:
                print("  → Selected")
        else:
            solution[event_id] = 0
            if debug:
                print("  → Skipped")

    return solution

# --- return structure ---

def run_event_selection_lp(event_pkl, overlap_pkl, meal_pkl, warm_start_fn = None, 
                           EventSelectionResult = namedtuple("EventSelectionResult", 
                                                             ["selected_events", "total_utility", "variable_values", "model"])
                          ):
    # --- load data ---
    with open(event_pkl, "rb") as f:
        event_df = pickle.load(f)

    with open(overlap_pkl, "rb") as f:
        overlap_df = pickle.load(f)

    with open(meal_pkl, "rb") as f:
        meal_df = pickle.load(f)

    # --- boost utility ---
    def boost_utility(u):
        if u >= 9:
            return u * 1000
        elif u >= 8:
            return u * 100
        else:
            return u * 1

    event_df['BoostedUtility'] = event_df['Utility'].apply(boost_utility)

    # --- build dicts ---
    utility = dict(zip(event_df['EventID'], event_df['BoostedUtility']))
    slot_lens = dict(zip(event_df['EventID'], event_df['TimeSlot'].apply(len)))
    event_ids = list(utility.keys())
    overlap_groups = list(overlap_df['OverlappingEventIDs'])
    meal_group_map = {k: sorted(list(v)) for k, v in meal_df.to_dict().items()}

    # --- init model ---
    model = ConcreteModel()
    model.Events = Set(initialize=event_ids)
    model.MealGroups = Set(initialize=meal_group_map.keys())
    model.GroupEvents = Set(model.MealGroups, initialize=meal_group_map)
    model.Utility = Param(model.Events, initialize=utility)
    model.TimeSlotLen = Param(model.Events, initialize=slot_lens)
    model.x = Var(model.Events, domain=Binary)

    # --- objective ---
    def obj_rule(m):
        return sum(m.Utility[e] * m.TimeSlotLen[e] * m.x[e] for e in m.Events)
    model.TotalUtility = Objective(rule=obj_rule, sense='maximize')

    # --- overlap constraints ---
    model.OverlapGroupIndex = Set(initialize=range(len(overlap_groups)))
    def overlap_rule(m, i):
        return sum(m.x[e] for e in overlap_groups[i]) <= 1
    model.NoOverlap = Constraint(model.OverlapGroupIndex, rule=overlap_rule)

    # --- meal constraints ---
    def meal_rule(m, g):
        return sum(m.x[e] for e in m.GroupEvents[g]) == 1
    model.OneMeal = Constraint(model.MealGroups, rule=meal_rule)

    # --- warm start ---
    if warm_start_fn is not None:
        initial_solution = warm_start_fn(event_df)
        for e, val in initial_solution.items():
            model.x[e].value = val
            model.x[e].fixed = False
    
    # --- solve ---
    solver = SolverFactory('gurobi')
    solver.solve(model, 
                 tee=True, 
                 warmstart=(warm_start_fn is not None) # depends if warm start algorithm provided
                ) 

    # --- extract results ---
    selected = [e for e in model.Events if model.x[e].value >= 0.5]
    total_utility = value(model.TotalUtility)
    variable_values = {e: model.x[e].value for e in model.Events}

    return EventSelectionResult(
        selected_events=selected,
        total_utility=total_utility,
        variable_values=variable_values,
        model=model  # optional, can be dropped if not needed
    )



#---------- ## visualizing
def normalize_room(room):
    if isinstance(room, tuple) and len(room) == 2:
        return room
    if isinstance(room, str) and room.lower() == "meal":
        return ("Meal", "Meal")
    try:
        parsed = ast.literal_eval(room)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            return parsed
    except:
        pass
    return ("Other", str(room))

def create_multiindex_schedule_matrix(df_full, selected_event_ids):
    exploded = df_full.explode("TimeSlot").copy()
    exploded["Room"] = exploded["Room"].apply(normalize_room)

    # build the cell tuples
    exploded["cell"] = exploded.apply(
        lambda r: (f"{r['Event']} ({r['Utility']})", r["EventID"], r["Utility"]),
        axis=1
    )

    # pivot to gather lists of tuples per (TimeSlot, Room)
    pivot = exploded.pivot_table(
        index="TimeSlot",
        columns="Room",
        values="cell",
        aggfunc=list,
        fill_value=None
    )

    # sort index chronologically
    def sort_key(t):
        dt = datetime.strptime(t, "%H:%M")
        if dt.hour < 6:
            dt += timedelta(hours=24)
        return dt

    pivot = pivot.reindex(sorted(pivot.index, 
                                 key=sort_key),
                          axis=0)

    # now convert column Index-of-tuples into a true MultiIndex
    pivot.columns = pd.MultiIndex.from_tuples(pivot.columns, names=["Venue", "Room"])

    # sort columns by Venue then Room
    pivot = pivot.reindex(
        sorted(pivot.columns, key=lambda x: (x[0], x[1])),
        axis=1
    )

    return pivot

def style_multiindex_matrix(matrix, selected_event_ids):
    def style_cell(cell):
        css = "border:1px solid #000;"
        if isinstance(cell, list):
            if any(eid in selected_event_ids for _, eid, _ in cell):
                css += "border:2px solid red;"
            if any(util >= 8 for _, _, util in cell):
                css += "background-color:#CCCCCC;"
        return css

    def format_cell(cell):
        if not isinstance(cell, list):
            return ""
        seen, lines = set(), []
        for lbl, _, _ in cell:
            if lbl not in seen:
                lines.append(lbl)
                seen.add(lbl)
        return "<br>".join(lines)

    # compute where Venue changes to apply thick right border down all rows
    venues = matrix.columns.get_level_values(0)
    boundaries = [
        i for i in range(1, len(venues))
        if venues[i] != venues[i-1]
    ]
    boundary_styles = [
        {
            'selector': f'td:nth-child({idx+1})',
            'props': [('border-right', '3px solid #000')]
        }
        for idx in boundaries
    ]

    return (
        matrix.style
              .map(style_cell)
              .format(format_cell)
              .set_table_styles(
                  # collapse once
                  [{'selector': 'table',
                    'props': [('border-collapse', 'collapse')]}]
                  # add thick separators at each boundary
                  + boundary_styles
              )
    )



def save_styled_png(styler, filename, width=1200):
    """
    Render a pandas Styler to a landscape PNG, preserving CSS.
    Requires wkhtmltoimage + imgkit.
    """
    html = styler.to_html()
    options = {
        'format': 'png',
        'width': str(width),
        'encoding': "UTF-8",
    }
    imgkit.from_string(html, filename, options=options)
    print(f"Saved styled table to {filename}")