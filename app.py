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
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import os



word_list = ['english', 'arabic', 'azerbaijani', 'basque', 'bengali', 'catalan', 'chinese', 'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'greek', 'hebrew', 'hinglish', 'hungarian', 'indonesian', 'italian', 'kazakh', 'nepali', 'norwegian', 'portuguese', 'romanian', 'russian', 'slovene', 'spanish', 'swedish', 'tajik', 'turkish']



def cleaning(data, word):
    stop_words = stopwords.words(word)
    #1. Tokenize
    text_tokens = word_tokenize(data.lower())   #removed the .lower intentionaly to keep NNP s
    
    #2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    
    #3. Remove stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    
    #joining
    return " ".join(tokens_without_sw)

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
            
            
                




def main():
    
   
    st.title("A Snapshot of Given Youtube Channel ")
    st.markdown("This application is a Streamlit dashboard used "
                "to provide a brief shapshot of the a given channel 🐦")
    
    
     
    link = st.text_input('link', placeholder = 'https://www.youtube.com/c/<channelname>/videos', key = '1')
    
    sentence = st.text_input('title for upcoming video', placeholder = 'Trends App in 2022', key = '2')
    
    word = st.selectbox('language for the given youtube channel', word_list)
    
    if st.button('Submit'):
        with st.spinner('Wait for it...'):
            time.sleep(1)
        
        
        if link.endswith( '/videos') and link.startswith('https://www.youtube.com/c/'):
            with st.spinner('Getting data...'):
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
                channel_info = element[0].text
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

                
                driver.close()
                df=pd.DataFrame(zip(text, view, date), columns=['text','views','date'])
                st.write(channel_info)
            with st.spinner('Analysing...'):
                df.views = df.views.apply(views)
                df['years_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'year' in x else 0 )
                df['months_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'month' in x else 0 if 'day' in x or 'week' in x or 'minute' in x else -1 )
                df['weeks_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'week' in x else 1 if ('day' in x and int(x.split(' ')[0]) > 6) else 0 if 'day' in x or 'minute' in x else -1 )
                df['days_ago']=df.date.apply(lambda x: int(x.split(' ')[0]) if 'day' in x else 0 if 'minute' in x else -1 )
            
                #df = pd.read_csv('df.csv')
                df_y = df[['views', 'years_ago']].groupby('years_ago').mean().reset_index()
                df_y.years_ago = df_y.years_ago.astype(str)
                df_m = df[df.months_ago != -1][['views', 'months_ago']].groupby('months_ago').mean().reset_index()
                df_m.months_ago = df_m.months_ago.astype(str)
                df_w = df[df.weeks_ago != -1][['views', 'weeks_ago']].groupby('weeks_ago').mean().reset_index()
                df_w.weeks_ago = df_w.weeks_ago.astype(str)
                df_d = df[df.days_ago != -1][['views', 'days_ago']].groupby('days_ago').mean().reset_index()
                df_d.days_ago = df_d.days_ago.astype(str)
                df['lan'] = word
                
                df["text2"] = df.apply(lambda x: cleaning( x.text, x.lan), 1)

                X = df['text2']
                y = df['views']

                tf_idf_word_vectorizer =TfidfVectorizer(max_df=0.95, min_df=3)
                tf_idf_word_vectorizer.fit(X)

                X_tf_idf_word = tf_idf_word_vectorizer.transform(X)
                model = RandomForestRegressor()
                model.fit(X_tf_idf_word, y)
                
                k = KMeans(n_clusters=4)
                
                k.fit(df[['views']]) 
                df['label'] = k.labels_
                df['label_str']= df.label.astype(str)
                
                
                feature_imp = feature_imp= pd.DataFrame(zip(model.feature_importances_,
                               tf_idf_word_vectorizer.get_feature_names_out()), columns=['value','word']).sort_values('value',ascending=False).head(20)


                df_g = df[['views','label']].groupby('label').agg(['count','mean','min', 'max'])
                df_g.reset_index(inplace=True)
                df_g.columns = ['Cluster','The Number Videos','Mean Value of Views','Minimum Value of View', 'Max Value of View']
                df_g = df_g.sort_values('Mean Value of Views', ascending=False)
                df_g.reset_index(inplace=True, drop = True)
                

                fig_y = px.bar(df_y, x='years_ago', y='views',height=500)
                fig_m = px.bar(df_m, x='months_ago', y='views',height=500)
                fig_w = px.bar(df_w, x='weeks_ago', y='views',height=500)
                fig_d = px.bar(df_d, x='days_ago', y='views',height=500)
                fig_wo = px.bar(feature_imp.sort_values('value'), x='value', y= 'word', orientation='h'  )
                fig_cl = px.scatter(df, 'views', 'label', color='label_str', symbol="label_str" )





            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Year", "Month ", "Week", "Day", 'Clusters', 'Effective words', 'Potential Views based on Title',"All"])


            with tab1:
                st.plotly_chart(fig_y, sharing="streamlit")

            with tab2:
                st.plotly_chart(fig_m, sharing="streamlit")

            with tab3:
                st.plotly_chart(fig_w, sharing="streamlit")

            with tab4:
                st.plotly_chart(fig_d, sharing="streamlit")
                
            with tab5:
                st.plotly_chart(fig_cl, sharing="streamlit")
                st.dataframe(df_g )

            with tab6:
                st.plotly_chart(fig_wo, sharing="streamlit") 

            with tab7:
                pot_views = model.predict(tf_idf_word_vectorizer.transform(pd.Series((cleaning(sentence, word)))))[0]
                text = f'Your pontential views with this title is {pot_views}'
                st.write(text)

            with tab8:
                st.plotly_chart(fig_y, sharing="streamlit")
                st.plotly_chart(fig_m, sharing="streamlit")
                st.plotly_chart(fig_w, sharing="streamlit")
                st.plotly_chart(fig_d, sharing="streamlit")
                st.plotly_chart(fig_wo, sharing="streamlit")
                st.write(text)
                
                
        else:
            with st.spinner('Wait for it...'):
                time.sleep(2)
            
#             'st.write("please enter the right version of link as following;")
#             st.write()'
           
            st.error("""please enter the right version of link as in link box""")
            

            
            
            
            
            


                


        
        
              
       

        


                        
                    

                
            

                    

if __name__ == '__main__':
    main()