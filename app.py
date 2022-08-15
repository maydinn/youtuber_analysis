import streamlit as st
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver import ActionChains
import time 
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import os
#nltk.download('punkt') 






def cleaning(data):
    
    #1. Tokenize
    text_tokens = word_tokenize(data.lower())   #removed the .lower intentionaly to keep NNP s
    
    #2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    
    #joining
    return " ".join(tokens_without_punc)





def main():
    #st.write(st.__version__)
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">enter link</h2>
    </div><br><br>"""
    
    
    
     
    link = st.text_input('link', placeholder = 'https://www.youtube.com/c/<channelname>/videos', key = '1')
    
    sentence = st.text_input('title for upcoming video', placeholder = 'Trends App in 2022', key = '2')
    
    if st.button('Submit'):
     
        
        
        if link.endswith( '/videos'):

            chrome_options = webdriver.ChromeOptions()
            chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)



            driver.get(link)
            time.sleep(2)

            driver.find_element_by_xpath("//button[@aria-label = 'Reject all']").click()
            time.sleep(3)


            while True:
                scroll_height = 12000
                document_height_before = driver.execute_script("return document.documentElement.scrollHeight")
                driver.execute_script(f"window.scrollTo(0, {document_height_before + scroll_height});")
                time.sleep(3)
                document_height_after = driver.execute_script("return document.documentElement.scrollHeight")
                if document_height_after == document_height_before:
                    break


            element = WebDriverWait(driver, 30).until(
                            EC.presence_of_all_elements_located((By.XPATH, "//*[@id='meta']"))
                            )




            text = []
            view = []
            date = []
            st.write(element[0].text)
            for i in element:

                a = i.text.split('\n')

                if (len(a) == 3) and ('waiting' not in (i.text)):
                    for j in a:
                        if 'CC' != j:
                            if ' views' in j:
                                view.append(j)

                            elif ' ago' in j:
                                if 'Streamed' in j:
                                    date.append(' '.join(j.split()[1:]))
                                else:
                                    date.append(j)                   

                            else:
                                text.append(j)



            df=pd.DataFrame(zip(text, view, date), columns=['text','views','date'])
            driver.close()
            def views(x):
                a = x.split(' ')[0]

                try:
                    a = int(a)

                except:
                    if '.' in a:
                        a = a.replace('.','')
                        if 'K' in a:
                            a = a.replace('K','00')
                        elif 'M' in a:
                            a = a.replace('M', '00000')
                        else:
                            print(a)
                            a ='##################'


                    else:
                        if 'K' in a:
                            a = a.replace('K','000')
                        elif 'M' in a:
                            a = a.replace('M', '000000')
                        else:
                            print(a)
                            a = '################'

                return int(a)


            df.views = df.views.apply(views)

            df['years_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'year' in x else 0 )
            df['months_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'month' in x else 0 if 'day' in x or 'week' in x or 'minute' in x else -1 )
            df['weeks_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'week' in x else 1 if ('day' in x and int(x.split(' ')[0]) > 6) else 0 if 'day' in x or 'minute' in x else -1 )
            df['days_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'day' in x else 0 if 'minute' in x else -1 )


            df_y = df[['views', 'years_ago']].groupby('years_ago').mean().reset_index()
            df_y.years_ago = df_y.years_ago.astype(str)
            df_m = df[df.months_ago != -1][['views', 'months_ago']].groupby('months_ago').mean().reset_index()
            df_m.months_ago = df_m.months_ago.astype(str)
            df_w = df[df.weeks_ago != -1][['views', 'weeks_ago']].groupby('weeks_ago').mean().reset_index()
            df_w.weeks_ago = df_w.weeks_ago.astype(str)
            df_d = df[df.days_ago != -1][['views', 'days_ago']].groupby('days_ago').mean().reset_index()
            df_d.days_ago = df_d.days_ago.astype(str)

            df["text2"] = df.text.apply(cleaning)

            X = df['text2']
            y = df['views']

            tf_idf_word_vectorizer =TfidfVectorizer(max_df=0.95, min_df=3)
            tf_idf_word_vectorizer.fit(X)

            X_tf_idf_word = tf_idf_word_vectorizer.transform(X)
            model = RandomForestRegressor()
            model.fit(X_tf_idf_word, y)

            feature_imp = feature_imp= pd.DataFrame(zip(model.feature_importances_,
                           tf_idf_word_vectorizer.get_feature_names_out()), columns=['value','word']).sort_values('value',ascending=False).head(20)



            fig_y = px.bar(df_y, x='years_ago', y='views',height=500)
            fig_m = px.bar(df_m, x='months_ago', y='views',height=500)
            fig_w = px.bar(df_w, x='weeks_ago', y='views',height=500)
            fig_d = px.bar(df_d, x='days_ago', y='views',height=500)
            fig_wo = px.bar(feature_imp.sort_values('value'), x='value', y= 'word', orientation='h'  )






          #  tab_time, tab_word = st.tabs(['time analysis', 'word analysis'])

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Year", "Month ", "Week", "Day", 'Effective words', 'Potential Views based on Title',"All"])


            with tab1:
                st.plotly_chart(fig_y, sharing="streamlit")

            with tab2:
                st.plotly_chart(fig_m, sharing="streamlit")

            with tab3:
                st.plotly_chart(fig_w, sharing="streamlit")

            with tab4:
                st.plotly_chart(fig_d, sharing="streamlit")

            with tab5:
                st.plotly_chart(fig_wo, sharing="streamlit") 

            with tab6:
                pot_views = model.predict(tf_idf_word_vectorizer.transform(pd.Series((cleaning(sentence)))))[0]
                text = f'Your pontential views with this title is {pot_views}'
                st.write(text)

            with tab7:
                st.plotly_chart(fig_y, sharing="streamlit")
                st.plotly_chart(fig_m, sharing="streamlit")
                st.plotly_chart(fig_w, sharing="streamlit")
                st.plotly_chart(fig_d, sharing="streamlit")
                st.plotly_chart(fig_wo, sharing="streamlit")
                st.write(text)
                
                
        else:
            st.write("""please enter the right version of link as following;
            https://www.youtube.com/c/<channelname>/videos""")

            
            
            
            
            


                


        
        
              
       

        


                        
                    

                
            

                    

if __name__ == '__main__':
    main()