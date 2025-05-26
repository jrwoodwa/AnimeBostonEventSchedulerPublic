# standard libraries
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import requests


# --- clean html tables ---
def preprocess_tables(tables, ii, FirstColumnName='TimeSlot'):
    table_df = tables[ii]
    
    table_df.dropna(subset=[table_df.columns[0]], inplace=True)
    table_df = table_df.iloc[:, :-1].copy()
    table_df.columns = [FirstColumnName] + table_df.columns[1:].tolist()

    # convert to 24-hr format
    vconvert = np.vectorize(lambda x: datetime.strptime(x, '%I:%M %p').strftime('%H:%M'))
    table_df[FirstColumnName] = vconvert(table_df[FirstColumnName])
    return table_df


# --- chunk events like Maid Cafe ---
def split_event_to_subevents(df, event_col, time_col, room_col, chunk_size=3, target_event=None):
    """
    deprecated in 2025 most likely
    """
    if target_event:
        df = df[df[event_col] == target_event]

    exploded_df = (df.explode(time_col)
                   .sort_values(by=[room_col, time_col])
                   .reset_index(drop=True))

    exploded_df['Subevent'] = exploded_df.groupby(room_col).cumcount() // chunk_size
    exploded_df['Subevent'] = exploded_df['Subevent'].apply(lambda x: f"{chr(65 + x)}")

    return (exploded_df
            .groupby([event_col, 'Subevent', 'Utility', 'Category', 'Color', room_col])[time_col]
            .agg(list)
            .reset_index())


# --- scrape and build category and events info ---
def fetch_schedule_data(url):
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    events = []
    events = [
        {
            "category_number": td.get("class")[1].split('-')[-1],
            "title": td.get("title"),
            "title_table": td.get_text(strip=True)
        }
        for td in soup.select("td.schedule-event")
    ]


    df_events = pd.DataFrame(events).drop_duplicates().sort_values(by='title')
    df_events['title'] = df_events['title'].replace("", np.nan).fillna(df_events['title_table']) #???

    df_events['category_number'] = df_events['category_number'].astype(int)

    legend_items = soup.select("div.schedule-legend label.schedule-category-label")
    category_colors = {
        label.text.strip(): {
            'Category': label.text.strip(),
            'Color': label.get("style").split("background-color:")[1].strip(),
            'Data Value': label.find('input').get('data-value')
        }
        for label in legend_items if "background-color:" in label.get("style", "")
    }

    df_categories = pd.DataFrame.from_dict(category_colors, orient='index')
    df_categories['Data Value'] = df_categories['Data Value'].astype(int)
    # fix missing accent in category name
    df_categories['Category'] = df_categories['Category'].replace("Maid Cafe", "Maid Café")

    # ------- if we don't have anything for information
    df_categories['Utility'] = 0
    # df_categories.to_csv('Event_Categories.csv', index=False, encoding='utf-8-sig')

    df_all_events = pd.merge(df_events, df_categories,
                             left_on='category_number',
                             right_on='Data Value',
                             how='left') \
                      .sort_values(by=['category_number', 'title']) \
                      .drop(columns=['category_number', 'Data Value'])

    # --- clean title_table into a new column ---
    df_all_events['title_id'] = df_all_events['title_table'].str.lower().str.replace(r'[\s\n\r\t]+', '', regex=True)

    return df_all_events, df_categories


# --- load and melt table ---
def process_event_table(url, all_events_df, table = 0):
    tables = pd.read_html(url)
    df = preprocess_tables(tables, table)

    df["RowOrder"] = df.index
    long_df = df.melt(id_vars=["TimeSlot", "RowOrder"], var_name="Room", value_name="Event")
    events_df = long_df.dropna(subset=["Event"]).sort_values("RowOrder").drop(columns="RowOrder").reset_index(drop=True)

    # --- clean title_table into a new column ---
    events_df['title_id'] = events_df['Event'].str.lower().str.replace(r'[\s\n\r\t]+', '', regex=True)

    events_df.columns = events_df.columns.str.strip()
    all_events_df.columns = all_events_df.columns.str.strip()

    #events_df.to_csv(f'eventsday_{table}.csv', index=False)
    ## changed to inner table just because we want to ensure events exist, and need dropped events to not show up
    merged_df = pd.merge(events_df, all_events_df,
                         on='title_id',
                         how='inner') \
                  .drop(columns=['Event']) \
                  .rename(columns={'title': 'Event'}) # careful on line above
    
    merged_df['AMVs'] = merged_df['AMVs'].astype(bool)
    grouped_df = merged_df.groupby(['Event', 'Room', 'Category', 'Color', 'Utility', 'AMVs'])['TimeSlot'].agg(list).reset_index()

    return grouped_df


# --- main pipeline ---
def main():
    url = "https://www.animeboston.com/schedule/index/2025"

    all_events_df,_ = fetch_schedule_data(url) # all_events_df, category_colors_df
    # These were used for a first pass, but afterwards, the table titles were quite dirty, so I had to scrub and tidy manually to match between days and all events
    all_events_df.to_excel('webscraped_all_events.xlsx')

    all_events_df = pd.read_excel('0_all_events.xlsx')
    all_events_df = all_events_df[all_events_df['Dropped']!= True] # do not pass through dropped events

    
    # --- clean title_table into a new column ---
    all_events_df['title_id'] = all_events_df['title_bestguess'].str.lower().str.replace(r'[\s\n\r\t]+', '', regex=True)
    
    for ii in range(0,3):
        grouped_df = process_event_table(url, all_events_df, table = ii)
    
        # handle Speed Friending splitting
        subevent_df = split_event_to_subevents(grouped_df, 'Event', 'TimeSlot', 'Room', 
                                               chunk_size=4, target_event="Speed Friending")
        
        grouped_df = grouped_df[grouped_df["Event"] != "Speed Friending"]
        final_df = pd.concat([grouped_df, subevent_df], ignore_index=True)
    
        # exclude non-schedulable items
        # Considering using anything with a utility of 0.0 as this
        exclude_from_scheduling = [
            "Room Clear", "Seating", "ID Check Seating (18+)"
        ]
        df_filtered = final_df[~final_df["Event"].isin(exclude_from_scheduling)].reset_index(drop=True)

        # convert unhashable list-type columns to strings
        # drop duplicate times
        time_df = df_filtered[['Event', 'Room', 'TimeSlot']]
        
        # drop TimeSlot column
        df_notslot = df_filtered.drop(columns=['TimeSlot'])
        
        # group without slot
        df_grouped = df_notslot.groupby(['Event', 'Room', 'Utility', 'AMVs'], as_index=False).agg({
            'Category': lambda x: '|'.join(sorted(set(x))),
            'Color': lambda x: '|'.join(sorted(set(x)))
        })

        df_grouped = df_grouped.drop_duplicates()
        
        # final merge result
        df_final = pd.merge(df_grouped, time_df, on=['Event', 'Room'], how='inner')

        df_final['TimeSlot']= df_final['TimeSlot'].astype(str)

        df_final = df_final.drop_duplicates()
    
        df_final.to_csv(f'AnimeBoston_day{ii}_schedule.csv', index=False, encoding='utf-8-sig')


# --- run the main function ---
if __name__ == "__main__":
    main()
    print(f"Last webscraped: {datetime.now():%Y-%m-%d %H:%M:%S}")
