import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import tweepy
import re
from wordcloud import WordCloud, STOPWORDS
from transformers  import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from keras_preprocessing.sequence import pad_sequences

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#matplotlib.use('Agg')


STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    
    html_temp = """
	<div style="background-color:LightBlue;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    # Twitter API Connection 
    consumer_key = st.secrets["consumer_key"]
    consumer_secret = st.secrets["consumer_secret"]
    access_token = st.secrets["access_token"]
    access_token_secret = st.secrets["access_token_secret"]



    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    
    
    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    
    # Write a Function to extract tweets:
    def get_tweets(topic, count):
        raw_data = api.search_tweets(q=topic, lang='en', result_type = "recent", count = count) # recent tweets in english
        # clean tweet API data
        raw_data = [i._json for i in raw_data]
        raw_data = [i['text'] for i in raw_data]
        # clean tweets
        data = []
        for i in range(len(raw_data)):
            data.append(preprocess_tweet(raw_data[i]))
        return raw_data, data

    # Function to Clean the Tweet.
    def preprocess_tweet(tweet):
        # remove links
        tweet = re.sub(r'http\S+', '', tweet)
        # remove special characters
        tweet = re.sub(r'@[a-zA-Z0-9\s]+', '', tweet)
        # remove hashtags
        tweet = re.sub(r'#\S+', ' ', tweet)
        # remove RT   
        tweet = re.sub(r'RT', ' ', tweet)
        # remove \n 
        tweet = re.sub(r'\n', ' ', tweet)
        
        # remove leading and trailing spaces
        tweet = tweet.strip()
        return tweet
    
        
    # Funciton to analyze Sentiment of tweet dataset
    def tweet_sentiment(tweets):
        # load Bert model finetuned for binary tweet sentiment classification
        output_dir = "./models/"
        model = BertForSequenceClassification.from_pretrained(output_dir)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        device = torch.device("cpu")
        attention_masks = []
        for i in range(len(tweets)):
            # add CLS for classification and SEP for end of each sentence of each input
            sentences = tweets[i].split(".")
            sentences[0] = "[CLS] " + sentences[0] + " [SEP]"
            for j in range(1,len(sentences)):
                sentences[j] = sentences[j] + " [SEP]"
            tweets[i] = " ".join(sentences)
                
            # tokenize the input, convert to id and pad
            tweets[i] = tokenizer.tokenize(tweets[i])
            tweets[i] = tokenizer.convert_tokens_to_ids(tweets[i])
            tweets[i] = torch.flatten(torch.tensor(pad_sequences([tweets[i]], maxlen=512, dtype="long", truncating="post", padding="post")))
            
            # mask
            mask = [float(i > 0) for i in tweets[i]]
            attention_masks.append(mask)
    
        # convert to tensor
        tweets = torch.stack(tweets)
        attention_masks = torch.tensor(attention_masks)
        # Prediction on tweets
        prediction_dataset = TensorDataset(tweets, attention_masks)
        prediction_sampler = SequentialSampler(prediction_dataset)
        prediction_dataloader = DataLoader(prediction_dataset, sampler=prediction_sampler, batch_size=32)
        # Put model on GPU in evaluation mode
        model.cpu()
        model.eval()

        # Tracking variables 
        predictions = []

        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                
            # Move logits to CPU
            logits = logits['logits'].detach().cpu()
            # Store predictions
            sent = torch.argmax(logits, axis=1).flatten()
            predictions.append(sent)
        # concatenate all predictions batches
        predictions = list(torch.cat(predictions, dim=0))
        return predictions
 
    image = mpimg.imread('index.png')
   # image = Image.open(imgplot)
    st.image(image, caption='Twitter for Analytics',use_column_width=True)
    
    
    # Collect Input from user :
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))     
    
    if len(Topic) > 0 :
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            original_tweets,tweets = get_tweets(Topic , count=200)
        st.success('Tweets have been Extracted')    

        # Call function to get the Sentiments
        sentiments = tweet_sentiment(tweets)
        # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(sentiments)))
        st.write("Total Positive Tweets are : {}".format(sentiments.count(1)))
        st.write("Total Negative Tweets are : {}".format(sentiments.count(0)))
        
        # See the Extracted Data : 
        if st.button("Extracted Data"):
            #st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the first 5 tweets of the Data :")
            st.write(original_tweets[:5])
     
        ## store sentiments and original tweets in a panda dataframe
        df = pd.DataFrame({'Sentiment':sentiments,'Tweet':original_tweets})
        # preprocess tweets, remove topic word from tweets and make letters lowercase
        df["Tweet"] = df["Tweet"].apply(lambda x : preprocess_tweet(x))
        df["Tweet"] = df["Tweet"].apply(lambda x: x.lower())
        df["Tweet" ] = df["Tweet"].apply(lambda x: x.replace(Topic,""))
        
        #Wordcloud for Positive tweets only
        if st.button("WordCloud for Positive Tweets"):
            st.success("WordCloud Positive Tweets")
            # get tweets column for which sentiment is positive
            positive = df[df['Sentiment'] == torch.tensor(1)]["Tweet"].values
            positive = " ".join(positive)
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Negative tweets only       
        if st.button("WordCloud for Negative Tweets"):
            st.success("WordCloud for Negative Tweets")
            negative = df[df['Sentiment'] == torch.tensor(0)]["Tweet"].values
            negative = " ".join(negative)
            stopwords = set(STOPWORDS)
            wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=70).generate(negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()

    st.sidebar.header("About App")
    st.sidebar.info("A Twitter Sentiment analysis Project which will scrap twitter for certain topic and analyze it.")

    if st.button("Exit"):
        st.balloons()


if __name__ == '__main__':
    main()