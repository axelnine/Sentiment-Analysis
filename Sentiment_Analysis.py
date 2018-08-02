# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:57:40 2017

@author: Eight
"""
#import pymysql,sys,global_variable as gv
from itertools import chain
from functools import partial
import re,string,os,pandas as pd
from nltk import bigrams,trigrams
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
#from sklearn.metrics import confusion_matrix

###############################################################################

#def connect():

#    try:
#        con = pymysql.connect(host=gv.host,user=gv.user, password=gv.password, db=gv.dbname)
#    except Exception as e:
#        print('Error in connecting to DB ',e)
#    return con

#def facebook(con):
#    cursor = con.cursor()
#    query = 'SELECT sentiment_type,comment_id, message, categorized_status FROM cingulariti.facebook_comments where categorized_status = "Active"'
#    try:
#        cursor.execute(query)
#    except Exception as e:
#        print('Error in query execution ',e)
#
#    df = pd.DataFrame(list(cursor.fetchall()),columns=['sentiment_type','comment_id','message','categorized_status'])
##    temp_df = df.copy()
##    print temp_df
#    if df.empty:
#        return
#    df = generate_sentiment(df,df['message'])
#    df = df [['sentiment_type','categorized_status','comment_id']]
##    predicted = df['sentiment_type']
##    actual = temp_df['sentiment_type']
##    accuracy = accuracy_score(actual,predicted)
##    print accuracy
#    for rows in df.itertuples():
#        query1 = '''Update facebook_comments Set `sentiment_type`=%s,`categorized_status`=%s where comment_id=%s'''
#        try:
#            cursor.execute(query1,(str(rows[1]),str(rows[2]),str(rows[3])))
#        except Exception , e:
#            print('Error in Updates query execution :', e)
#            return None
#        con.commit()
#    print("Succesful")

######################################################################################################################################

#def googleplus(con):
#    print("in gplus")
#    cursor = con.cursor()
#    query = 'SELECT comment_id, comment, categorized_status FROM cingulariti.googleplus_comments where categorized_status = "Active"'
#    try:
#        cursor.execute(query)
#    except Exception as e:
#        print('Error in query execution ',e)
#
#    df = pd.DataFrame(list(cursor.fetchall()),columns=['comment_id','comment','categorized_status'])
#    if df.empty:
#        return
#    df = generate_sentiment(df,df['comment'])
#    df = df [['sentiment_type','categorized_status','comment_id']]
#    print df
#    for rows in df.itertuples():
#        query1 = '''Update googleplus_comments Set `sentiment_type`=%s,`categorized_status`=%s where comment_id=%s'''
#        try:
#            cursor.execute(query1,(str(rows[1]),str(rows[2]),str(rows[3])))
#        except Exception , e:
#            print('Error in Updates query execution :', e)
#            return None
#        con.commit()
#    print("Succesful")

######################################################################################################################################

#def linkedin(con):
#    cursor = con.cursor()
#    query = 'SELECT comment_id, comment, categorized_status FROM cingulariti.linkedin_comments where categorized_status = "Active"'
#    try:
#        cursor.execute(query)
#    except Exception as e:
#        print('Error in query execution ',e)
#
#    df = pd.DataFrame(list(cursor.fetchall()),columns=['comment_id','comment','categorized_status'])
#    if df.empty:
#        return
#    df = generate_sentiment(df,df['comment'])
#    df = df [['sentiment_type','categorized_status','comment_id']]
#
#    for rows in df.itertuples():
#        query1 = '''Update linkedin_comments Set `sentiment_type`=%s,`categorized_status`=%s where comment_id=%s'''
#        try:
#            cursor.execute(query1,(str(rows[1]),str(rows[2]),str(rows[3])))
#        except Exception , e:
#            print('Error in Updates query execution :', e)
#            return None
#        con.commit()
#    print("Succesful")

######################################################################################################################################

#def youtube(con):
#    cursor = con.cursor()
#    query = 'SELECT comment_id, comment, categorized_status FROM cingulariti.youtube_comments where categorized_status = "Active"'
#    try:
#        cursor.execute(query)
#    except Exception as e:
#        print('Error in query execution ',e)
#
#    df = pd.DataFrame(list(cursor.fetchall()),columns=['comment_id','comment','categorized_status'])
#    if df.empty:
#        return
#    df = generate_sentiment(df,df['comment'])
#    df = df [['sentiment_type','categorized_status','comment_id']]
#
#    for rows in df.itertuples():
#        query1 = '''Update youtube_comments Set `sentiment_type`=%s,`categorized_status`=%s where comment_id=%s'''
#        try:
#            cursor.execute(query1,(str(rows[1]),str(rows[2]),str(rows[3])))
#        except Exception , e:
#            print('Error in Updates query execution :', e)
#            return None
#        con.commit()
#    print("Succesful")

######################################################################################################################################

#def twitter(con):
#    cursor = con.cursor()
#    query = 'SELECT id, tweets, categorized_status FROM cingulariti.searchtweets where categorized_status = "Active"'
#    try:
#        cursor.execute(query)
#    except Exception as e:
#        print('Error in query execution ',e)
#
#    df = pd.DataFrame(list(cursor.fetchall()),columns=['id','tweets','categorized_status'])
#    if df.empty:
#        return
#    df = generate_sentiment(df,df['tweets'])
#    df = df [['sentiment_type','categorized_status','id']]
#
#    for rows in df.itertuples():
#        query1 = '''Update searchtweets Set `sentiment_type`=%s,`categorized_status`=%s where id=%s'''
#        try:
#            cursor.execute(query1,(str(rows[1]),str(rows[2]),str(rows[3])))
#        except Exception , e:
#            print('Error in Updates query execution :', e)
#            return None
#        con.commit()
#    print("Succesful")


######################################################################################################################################
######################################################################################################################################

def remove_punctuation(s):
	'''
	Function to remove punctuation from text
	'''
    a = set(string.punctuation)
    a.remove('.')
    statement = ''.join([i for i in s if i not in a])
    return statement

def dictionary_of_improper_words(tempreviewlist):
	path = os.getcwd()
	word_re = re.compile(r'\b[a-zA-Z\'&.]+\b')
	imprcsv_dict = pd.Series.from_csv(path+"\imprp_dict.csv").to_dict()
	def helper(dic, match):
		word = match.group(0)
		return dic.get(word, word)
	func = lambda s: word_re.sub(partial(helper, imprcsv_dict), str(s))
	rev_with_proper_words = tempreviewlist.apply(func)
	rev_with_proper_words = pd.DataFrame(rev_with_proper_words)
	rev_with_proper_words.columns = ['Cleaned_Review']
	rev_with_proper_words['Cleaned_Review'] = rev_with_proper_words['Cleaned_Review'].str.lower().str.split()
	return rev_with_proper_words

######################################################################################################################################

def remove_stopwords(x):
    path=os.getcwd()
    stop = set(stopwords.words('english'))
    stop.remove('not')
    stop.update(("mon","tue","wed","thu","fri","sat","sun","sunday","monday","tuesday","thursday","friday","saturday","sunday","thurs","thur","tues"))
    stop.update(("january","february","march","april","may","june","july","august",
          "september","october","november","december","jan","feb","mar","apr",
          "may","jun","jul","aug","sep","oct","nov","dec","thanking","thanks"))
    stopwords_csv = pd.read_csv(path+"\stopwords.csv")
    stopwords_csv_values = stopwords_csv.values.flatten()
    stop.update(stopwords_csv_values)

    list_of_cities_and_towns = pd.read_csv(path+"\list_of_cities_and_towns.csv")
    state_values = pd.DataFrame(list_of_cities_and_towns, columns = ['State'])
    state_values = state_values['State'].str.lower()
    state_values = state_values.values.flatten()
    stop.update(state_values)

    city_values = pd.DataFrame(list_of_cities_and_towns, columns = ['Name of City'])
    city_values = city_values['Name of City'].str.lower()
    city_values = city_values.values.flatten()
    stop.update(city_values)
    a = [item for item in x if item not in stop]
    return a

def flatten_list(templist):
	return list(chain.from_iterable(templist))

###################################################################################################################################################################

def convert_and_encode(tlist):
    temp_list_trigrams = []
    final_data = []
    for i in range(0,len(tlist)):
        temp_list_trigrams.append(' '.join(tlist[i]))
    for i in temp_list_trigrams:
#        final_data.append(re.sub((r'[^\x00-\x7F]'),'',i.encode('utf-8','ignore')))
        final_data.append(re.sub((r'[^\x00-\x7F]'),'',i.decode('ascii','ignore')))
    return final_data

def remove_hashtags_and_links(listtemp):
    list2 = []
    for i in listtemp:
        list2.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",i).split()))
    return list2

def lemmatize(x):
    a = []
    try:
        for i in x:
            a.append(WordNetLemmatizer().lemmatize(i))
        a.append(i)
        return a
    except:
        return x

def clean_data(x):
    try:
        for i in x:
            a = ' '.join(x)
            return a
    except:
        return ''

######################################################################################################################################

# Function : To define classification model as well as clean and preprocess the traiing data
# Input(s) : Dataframe of new mails with cleaned concatenated column after removing std text
# Output(s): New mails list(dataframe) with proper words concatenated column of Subject+Body
def clean_and_fit_training_data():
    print 'Preprocessing training data...'

    #classifier
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1,2))),
                         ('tfidf', TfidfTransformer()),
                        ('clf',MultinomialNB())])
#                         SGDClassifier(loss='log', penalty='l2',
#						 alpha=1e-5, n_iter=5, random_state=42, learning_rate = 'optimal'))])

    #Preprocess training data
    traindataframe = pd.read_csv('fulldata.csv')
#    traindataframe,testdataframe = train_test_split(traindataframe, test_size=0.3, random_state=0)
#    a = cross_val_score(text_clf,traindataframe['Review'].tolist(),traindataframe['Polarity'].tolist(),cv=5)
#    testdataframe  = traindataframe
    reviewlist = traindataframe['Review'].apply(lambda x : remove_punctuation(str(x)))
    polaritylist = traindataframe['Polarity']
    reviewlist = dictionary_of_improper_words(reviewlist);
    tempolist = reviewlist['Cleaned_Review'].tolist()
    reviewlist['Cleaned_Review'] = reviewlist['Cleaned_Review'].apply(lambda x: remove_stopwords(x))
    x1 = reviewlist['Cleaned_Review'].tolist()

    reviewlistforsearch = traindataframe.loc[traindataframe['Polarity'] == 'negative']
    reviewlistforsearch = reviewlistforsearch['Review'].apply(lambda x : remove_punctuation(str(x)))
    reviewlistforsearch = dictionary_of_improper_words(reviewlistforsearch)
    tempolistforsearch = reviewlistforsearch['Cleaned_Review'].tolist()
#    print len(dftosearch)
    #Cleaning up training data
    train_data = convert_and_encode(x1)
    train_data = remove_hashtags_and_links(train_data)
    train_data_with_stopwords = convert_and_encode(tempolist)
    train_data_with_stopwords = remove_hashtags_and_links(train_data_with_stopwords)

    data_for_search = convert_and_encode(tempolistforsearch)
    data_for_search = remove_hashtags_and_links(data_for_search)
    searchseries = pd.Series(data_for_search)
    #Create dataframes
    df_with_stopwords = pd.DataFrame({'Review' : train_data_with_stopwords, 'Polarity': polaritylist})
    df_without_stopwords = pd.DataFrame({'Review' : train_data, 'Polarity': polaritylist})
    train_x_with_stopwords = df_with_stopwords['Review']
    train_x = df_without_stopwords['Review']
    train_y = df_without_stopwords['Polarity']

    #Fit data into classifir
    print 'Fitting data into model...'
    text_clf.fit(train_x, train_y)

#    tt = train_x.tolist()
#    tokenising_for_lemmatization = [word_tokenize(i) for i in tt]
#    dftest['Lemmatized_training_data'] = pd.Series(tokenising_for_lemmatization)
#    dftest['Lemmatized_training_data'] = dftest['Lemmatized_training_data'].apply(lambda x : lemmatize(x))
#    dftest['Lemmatized_training_data'] = dftest['Lemmatized_training_data'].apply(lambda x : clean_data(x))
#    list_of_lemmatized_reviews_train = dftest['Lemmatized_training_data'][dftest['Lemmatized_training_data'] != ''].tolist()
#    text_clf.fit(list_of_lemmatized_reviews_train,train_y)

    return train_x_with_stopwords,train_x,train_y,train_data_with_stopwords,text_clf,searchseries

######################################################################################################################################

# Function : To define classification model as well as clean and preprocess the traiing data
# Input(s) : Dataframe of new mails with cleaned concatenated column after removing std text
# Output(s): New mails list(dataframe) with proper words concatenated column of Subject+Body
def preprocess_testing_data(test_list,train_list,text_clf):
        print 'Preprocessing Test Data...'
        #tokenizing training data and extracting bigrams and trigrams
        tokenized_sents = [word_tokenize(i) for i in train_list]
        bi_tokens = [list(bigrams(i))  for i in tokenized_sents]
        tri_tokens = [list(trigrams(i))  for i in tokenized_sents]
        temp_list_bigrams = flatten_list(bi_tokens)
        temp_list_trigrams = flatten_list(tri_tokens)
        exam = pd.read_csv('negative_words.csv')
        exam = exam['Words'].values.tolist()
        list_trigrams = []
        list_bigrams = []

        #Construct n-gram dictionary from negative words
        for i in exam:
            extract_tuple = [item for item in temp_list_trigrams if i in item]
            extract_tuple_bigrams = [item for item in temp_list_bigrams if i in item]
            list_trigrams.append(extract_tuple)
            list_bigrams.append(extract_tuple_bigrams)

        #Construct n-gram phrases
        list_trigrams = flatten_list(list_trigrams)
        list_bigrams = flatten_list(list_bigrams)
        trigrams_with_negative_phrases = [' '.join(i) for i in list_trigrams]
        bigrams_with_negative_phrases = [' '.join(i) for i in list_bigrams]

#        Add additional bigrams & trigrams
        new_trigrams = pd.read_csv('D_Vois_NegativeTrigrams.csv')
        add_trigrams = new_trigrams['Phrases'].tolist()
        for i in add_trigrams:
            trigrams_with_negative_phrases.append(i)
        new_bigrams = pd.read_csv('D_Vois_NegativeBigrams.csv')
        add_bigrams = new_bigrams['Phrases'].tolist()
        for i in add_bigrams:
            bigrams_with_negative_phrases.append(i)

        #check if any n-grams are present in the testing data.
        #If present, directly classify them as negative
        def check_for_trigrams(x):
            for word in trigrams_with_negative_phrases:
                if word in x:
                    return 'negative'
            return 'None'

        def check_for_bigrams(x):
            for word in bigrams_with_negative_phrases:
                if word in x:
                    return 'negative'
            return 'None'


        list_of_remaining_reviews = []
        df1 = pd.DataFrame({'Review' : test_list})
        print 'Searching for n-grams...'
        df1['Polarity'] = df1['Review'].apply(lambda x:check_for_trigrams(x))
#        df1['Polarity'] = 'None'
        remaining_reviews_after_trigram_check = df1.loc[df1['Polarity'] == 'None']
        remaining_reviews_after_trigram_check['Polarity'] = remaining_reviews_after_trigram_check['Review'].apply(lambda x:check_for_bigrams(x))
#        bigram_checked = remaining_reviews_after_trigram_check['Polarity']
        df1['Polarity'][df1['Polarity'] == 'None'] = remaining_reviews_after_trigram_check['Polarity']
#        print df1
        a = df1.loc[df1['Polarity'] == 'negative']
        a.to_csv('Dictionaryclassified.csv')
        remaining_reviews = remaining_reviews_after_trigram_check.loc[remaining_reviews_after_trigram_check['Polarity'] == 'None']
#        print len(remaining_reviews)
        temp_remaining_reviews = remaining_reviews['Review'].tolist()
        for i in temp_remaining_reviews:
            list_of_remaining_reviews.append(i.split())

        list_remaining_reviews = pd.DataFrame({'Remaining Reviews' : list_of_remaining_reviews})
        list_remaining_reviews['Remaining Reviews'] = list_remaining_reviews['Remaining Reviews'].apply(lambda x: remove_stopwords(x))
        list_remaining_reviews = list_remaining_reviews['Remaining Reviews'].tolist()
        list_remaining_reviews = convert_and_encode(list_remaining_reviews)


#        tokenising_for_lemmatization = [word_tokenize(i) for i in list_remaining_reviews]
#        df1['Lemmatized_training_data'] = pd.Series(tokenising_for_lemmatization)
#        df1['Lemmatized_training_data'] = df1['Lemmatized_training_data'].apply(lambda x : lemmatize(x))
#        df1['Lemmatized_training_data'] = df1['Lemmatized_training_data'].apply(lambda x : clean_data(x))
#        list_of_lemmatized_reviews = df1['Lemmatized_training_data'][df1['Lemmatized_training_data'] != ''].tolist()
#        classification_remaining_reviews = text_clf.predict(list_of_lemmatized_reviews)

        classification_remaining_reviews = text_clf.predict(list_remaining_reviews)
        remaining_reviews_dataframe = pd.DataFrame({'Reviews with Stopwords' : remaining_reviews['Review'],'Polarity' : classification_remaining_reviews})
#        print len(remaining_reviews_dataframe)
#        b = df1.loc[df1['Polarity'] == 'None']
#        c = df1.loc[df1['Polarity'] == 'negative']
#        print len(b)
#        print len(c)
        df1['Polarity'][df1['Polarity'] == 'None'] = remaining_reviews_dataframe['Polarity']
#        df1['Polarity'] = remaining_reviews_dataframe['Polarity']
#        df1['Polarity'][df1['Polarity'] == np.nan] = bigram_checked
        pred_y = df1['Polarity'].tolist()

        return pred_y,text_clf

######################################################################################################################################

def generate_sentiment(df,testlist):

    train_x_with_stopwords,train_x,train_y,train_data_with_stopwords,text_clf,searchseries = clean_and_fit_training_data()
#    testlist = testdataframe['Review']
    testlist = testlist.apply(lambda x : remove_punctuation(str(x)))
    testreviewlist = dictionary_of_improper_words(testlist);
    tempolist = testreviewlist['Cleaned_Review'].tolist()
    testreviewlist['Cleaned_Review'] = testreviewlist['Cleaned_Review'].apply(lambda x: remove_stopwords(x))
    x1 = testreviewlist['Cleaned_Review'].tolist()
    test_data = convert_and_encode(x1)
    test_data = remove_hashtags_and_links(test_data)

    test_data_with_stopwords = convert_and_encode(tempolist)
    test_data_with_stopwords = remove_hashtags_and_links(test_data_with_stopwords)

    prediction,text_clf = preprocess_testing_data(test_data_with_stopwords,searchseries,text_clf)
    print 'Predicting...'
    df['categorized_status'] = 'Inactive'
    df['sentiment_type'] = prediction
    return df

######################################################################################################################################
######################################################################################################################################

def main():
    df = pd.read_csv('testingdata.csv')
    testdataframe = generate_sentiment(df,df['Review'])
    print testdataframe
#    accuracy = accuracy_score(testdataframe['Polarity'],testdataframe['sentiment_type'])
#    temp = testdataframe.loc[testdataframe['Polarity'] != testdataframe['sentiment_type']]
#    temp.to_csv('Misclassifieddata.csv')
#    a = confusion_matrix(testdataframe['Polarity'],testdataframe['sentiment_type'])
#    print a
#    print accuracy*100
	#con = connect()
    #facebook(con)
    #googleplus(con)
    #linkedin(con)
    #youtube(con)
    #twitter(con)
    print 'Successful'


main()
