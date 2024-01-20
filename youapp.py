import requests
import pandas as pd
import streamlit as st
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px


def get_channel_info(api_key, channel_name):
    base_url = "https://www.googleapis.com/youtube/v3/search"

    # Step 1: Search for the channel by name to get the channel ID
    params = {
        'part': 'id',
        'q': channel_name,
        'type': 'channel',
        'key': api_key,
        'maxResults': 1
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
       
        json_data = response.json()

        channel_id = json_data['items'][0]['id']['channelId']

        # Step 2: Fetch channel statistics using the obtained channel ID
        stats_url = "https://www.googleapis.com/youtube/v3/channels"
        stats_params = {
            'part': 'statistics',
            'id': channel_id,
            'key': api_key
        }

        stats_response = requests.get(stats_url, params=stats_params)
        stats_response.raise_for_status()
        stats_data = stats_response.json()
        channel_statistics = stats_data['items'][0]['statistics']

        # Step 3: Fetch channel snippet using the obtained channel ID
        snippet_url = "https://www.googleapis.com/youtube/v3/channels"
        snippet_params = {
            'part': 'snippet',
            'id': channel_id,
            'key': api_key
        }

        snippet_response = requests.get(snippet_url, params=snippet_params)
        snippet_response.raise_for_status()
        snippet_data = snippet_response.json()
        channel_snippet = snippet_data['items'][0]['snippet']

        return {
            'channel_id': channel_id,
            'statistics': channel_statistics,
            'snippet': channel_snippet
        }
        
    except requests.exceptions.RequestException as e:
        st.write(f"An error occurred: {e}")
        return None

def extract_minutes(duration_str):
    duration_timedelta = pd.to_timedelta(duration_str)
    return duration_timedelta.total_seconds() / 60


def get_channel_videos(api_key, channel_id):
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=200&type=video"


    videos = []
    count = 0
    while True:

        response = requests.get(url)

        if response.status_code != 200:
            st.write("Error: Unable to fetch data from YouTube API")
            st.write(response)
            break

        data = response.json()

        for item in data['items']:
            video_info = item['snippet']
            video_title = video_info['title']
            video_publish_time = video_info['publishedAt']

            # Get video statistics
            video_id = item['id']['videoId']
            video_stats_url = f"https://www.googleapis.com/youtube/v3/videos?key={api_key}&id={video_id}&part=statistics"
            stats_response = requests.get(video_stats_url)
            stats_data = stats_response.json()
            view_count = stats_data['items'][0]['statistics']['viewCount']

            video_cot_url = f"https://www.googleapis.com/youtube/v3/videos?key={api_key}&id={video_id}&part=contentDetails"
            stats_response = requests.get(video_cot_url)
            stats_data = stats_response.json()
            duration = stats_data['items'][0]['contentDetails']['duration']

            videos.append({
                "Title": video_title,
                "Published At": video_publish_time,
                "View Count": view_count,
                "duration": duration
            })
        
        count += 1
        if count == 2:
            break
       # Check if there are more pages of results
        if 'nextPageToken' in data:
            next_page_token = data['nextPageToken']
            url = f"{url}&pageToken={next_page_token}"
            
            
        else:
            break
    return pd.DataFrame(videos)

def clean_title(title):
    # Remove numbers
    title_no_numbers = re.sub(r'\d+', '', title)
    
    # Remove punctuation (keeping only letters and spaces)
    title_no_punctuation = re.sub(r'[^\w\s]', '', title_no_numbers)
    
    return title_no_punctuation

def word_imp(df):
    if len(df) > 10:
        df['Title'] = df['Title'].apply(clean_title)

      
        vectorizer = TfidfVectorizer(max_df = 0.9)
        X = vectorizer.fit_transform(df['Title'])
        y = df['View Count']

        svr_model = SVR(kernel='linear')
        svr_model.fit(X, y)

        # Get feature names (words corresponding to BERT embeddings)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Get the most important positive and negative words
        coefficients = svr_model.coef_.toarray().reshape(len(feature_names))
        positive_indices = np.argsort(coefficients)[-20:]
        negative_indices = np.argsort(coefficients)[:20]

        most_positive_coefficients = coefficients[positive_indices]
        most_negative_coefficients = coefficients[negative_indices]
        most_positive_words = feature_names[positive_indices]
        most_negative_words = feature_names[negative_indices]

        return most_positive_coefficients, most_negative_coefficients, most_positive_words, most_negative_words
    else:
        return None

#api
api_key = st.secrets["API_KEY"]

channel_name = st.text_input('Enter Channel Name')

if st.button("Submit"):
    
    if channel_name != None:
        ch_info = get_channel_info(api_key, channel_name)

        ch_id = ch_info["channel_id"]
        st.image(ch_info['snippet']['thumbnails']['default']['url'])
        st.write(f"Channel Title: {ch_info['snippet']['title']}")
        st.write(f"Channel active since: {ch_info['snippet']['publishedAt'][:10]}")
        st.write(f"Channel Total View: {int(ch_info['statistics']['viewCount']):,}")
        st.write(f"Total Subs: {int(ch_info['statistics']['subscriberCount']):,}")
        st.write(f"Total Video: {ch_info['statistics']['videoCount']}")

        if ch_id != None:
            df = get_channel_videos(api_key, ch_id)

            df['View Count'] = df['View Count'].astype(int)
            df['duration'] = df.duration.apply(lambda x: extract_minutes(x))
            df['Published At'] = pd.to_datetime(df['Published At'])
            df['date'] = df['Published At'].dt.date
            df['hour'] = df['Published At'].dt.hour
            df['day'] = df['Published At'].dt.day_name()
       
            df_s = df[df.duration <= 1 ]
            df = df[df.duration > 1 ]

            
            if len(df)> 10:
                pos_co, neg_co, pos_w, neg_w = word_imp(df)
            if len(df_s) > 10:
                pos_co_s, neg_co_s, pos_w_s, neg_w_s = word_imp(df_s)
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Table', 'Relation View and Duration', 'Views among Published Hours', 'Views among Published Days of Week', 'Views among Published Days of Week & Hours','Postive Words', 'Negative Words'])
            with tab1:
                st.write(df[['Title', 'View Count', 'date']].sort_values('View Count',ascending
=False).reset_index(drop=True))
                expander = st.expander('See for Shorts')
                expander.write(df_s[['Title', 'View Count', 'date']].sort_values('View Count',ascending
=False).reset_index(drop=True))
                
            with tab2:
                if len(df) > 10:                 
                    fig2 = px.scatter(df, x='duration', y='View Count', trendline="ols", labels={'duration': 'Duration', 'View Count': 'View Count'})
                    fig2.update_layout(
                        title="Scatter Plot with Trend Line by Duration and View Count",
                        xaxis_title="Duration",
                        yaxis_title="View Count",
                        
                    )

                    st.plotly_chart(fig2)
                else:
                    st.write('Not enough data')
                    
                expander = st.expander('See for Shorts')
                if len(df_s) > 10:
                    fig2_1 = px.scatter(df_s, x='duration', y='View Count', trendline="ols", labels={'duration': 'Duration', 'View Count': 'View Count'})
                    fig2_1.update_layout(
                        title="Scatter Plot with Trend Line by Duration and View Count for Shorts",
                        xaxis_title="Duration",
                        yaxis_title="View Count",
                        
                    )

                    expander.plotly_chart(fig2_1)
                else:
                    expander.write('Not enough data')
                
            with tab3:
                if len(df) > 10:   
                   # fig2, ax2= plt.subplots()
                    fig3 = px.box(df, x="hour", y="View Count")

                    
                    fig3.update_layout(
                        title="Box Plot of View Count by Hour",
                        xaxis_title="Hour",
                        yaxis_title="View Count",
                        
                    )
                    st.plotly_chart(fig3)
                else:
                    st.write('Not enough data')
                expander = st.expander('See for Shorts')
                if len(df_s) > 10:
                    fig3_1 = px.box(df_s, x="hour", y="View Count")
                    
                    fig3_1.update_layout(
                        title="Box Plot of View Count by Hour for Shorts",
                        xaxis_title="Hour",
                        yaxis_title="View Count",
                       
                    )
                    expander.plotly_chart(fig3_1)
                else:
                    expander.write('Not enough data')


            desired_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']    
            with tab4:
                
                
                if len(df) > 10: 
                    fig4 = px.box(df, x="day", y="View Count")
                    fig4.update_xaxes(categoryorder='array', categoryarray=desired_order)
                    fig4.update_layout(
                        title="Box Plot of View Count by Day",
                        xaxis_title="Day",
                        yaxis_title="View Count",
                       
                    )
                    
                    st.plotly_chart(fig4)
     
                else:
                    st.write('Not enough data')
                expander = st.expander('See for Shorts')
                if len(df_s) > 10:
                    fig4_1 = px.box(df_s, x="day", y="View Count")
                    fig4_1.update_xaxes(categoryorder='array', categoryarray=desired_order)
                    fig4_1.update_layout(
                        title="Box Plot of View Count by Day for Shorts",
                        xaxis_title="Day",
                        yaxis_title="View Count",
                   
                    )
                    
                    expander.plotly_chart(fig4_1)
                else:
                    expander.write('Not enough data')
                    
            with tab5:
 
                fig5 = px.box(df, x='day', y='View Count', color='hour')
                fig5.update_xaxes(categoryorder='array', categoryarray=desired_order)
                fig5.update_layout(
                        title="Box Plot of View Count by Days & Hours ",
                        xaxis_title="Day",
                        yaxis_title="View Count",
                   
                    )
                st.plotly_chart(fig5)
#                 pivot_table = df.pivot_table(values='View Count', index='day', columns='hour', aggfunc='mean').fillna(0)
#                 st.write(pivot_table)
                expander = st.expander('See for Shorts')
                if len(df_s) > 10:
                    fig5_1 = px.box(df_s, x='day', y='View Count', color='hour')
                    fig5_1.update_xaxes(categoryorder='array', categoryarray=desired_order)
                    fig5_1.update_layout(
                        title="Box Plot of View Count by Days & Hours for Shorts",
                        xaxis_title="Day",
                        yaxis_title="View Count",
                   
                    )
                    
                    expander.plotly_chart(fig5_1)
                else:
                    expander.write('Not enough data')

                    
            with tab6:
                if len(df) > 10:                  
                    fig4 = px.bar(x=pos_co, y=pos_w, labels={'x': 'Values', 'y':'Words'}, title='Bar Plot for Positive Words')
                    fig4.update_layout(yaxis=dict(tickmode='array', tickvals=pos_w, ticktext=pos_w))
                    fig4.update_xaxes(title_text='Values')
                    fig4.update_yaxes(title_text='Positive Words')
                    st.plotly_chart(fig4)
                else:
                    st.write('Not enough data')
                expander = st.expander('See for Shorts')
                if len(df_s) > 10:                    
                    fig4 = px.bar(x=pos_co_s, y=pos_w_s, labels={'x': 'Values', 'y':'Words'}, title='Bar Plot for Positive Words')
                    fig4.update_layout(yaxis=dict(tickmode='array', tickvals=pos_w_s, ticktext=pos_w_s))
                    fig4.update_xaxes(title_text='Values')
                    fig4.update_yaxes(title_text='Positive Words')
                    expander.plotly_chart(fig4)
                else:
                    expander.write('Not enough data')
                    


                    
            with tab7:
                if len(df) > 10:
                    fig5 = px.bar(x=neg_co, y=neg_w, labels={'x': 'Values', 'y':'Words'}, title='Bar Plot for Negative Words')
                    fig5.update_layout(yaxis=dict(tickmode='array', tickvals=neg_w, ticktext=neg_w))
                    fig5.update_xaxes(title_text='Values')
                    fig5.update_yaxes(title_text='Negative Words')
                    st.plotly_chart(fig5)
                else:
                    st.write('Not enough data')
                    
                expander = st.expander('See for Shorts')
                
                if len(df_s) > 10:
                    fig5 = px.bar(x=neg_co_s, y=neg_w_s, labels={'x': 'Values', 'y':'Words'}, title='Bar Plot for Negative Words')
                    fig5.update_layout(yaxis=dict(tickmode='array', tickvals=neg_w_s, ticktext=neg_w_s))
                    fig5.update_xaxes(title_text='Values')
                    fig5.update_yaxes(title_text='Negative Words')
                    expander.plotly_chart(fig5)
                else:
                    expander.write('Not enough data')
                
                
           

