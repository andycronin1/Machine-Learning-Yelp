import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

#reading in json's
businesses = pd.read_json('yelp_business.json', lines=True)
reviews = pd.read_json('yelp_review.json', lines=True)
user = pd.read_json('yelp_user.json', lines=True)
checkins = pd.read_json('yelp_checkin.json', lines=True)
tips = pd.read_json('yelp_tip.json', lines=True)
photos = pd.read_json('yelp_photo.json', lines=True)

pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500


#merging datasets
df = pd.merge(businesses, reviews, how='left', on='business_id')
df = pd.merge(df, user, how='left', on='business_id')
df = pd.merge(df, checkins, how='left', on='business_id')
df = pd.merge(df, tips, how='left', on='business_id')
df = pd.merge(df, photos, how='left', on='business_id')
#print(len(df))

#list of features to remove and dropping them using df.drop
features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']

df.drop(features_to_remove, axis=1, inplace=True)

#replacing N/A values in columns with 0's
df.fillna({'weekday_checkins':0,
           'weekend_checkins':0,
           'average_tip_length':0,
           'number_tips': 0,
           'average_caption_length': 0,
           'number_pics': 0},
          inplace=True)

#finding if there are any NA values
#df.isna().any()

#printing correlation between coefficients
#print(df.corr())

#Extracting specific variables needed for model 
features = df[['average_review_length','average_review_age']]
ratings = df['stars']

#setting up training and test sets with a test size of 20%
X_train,X_test,y_train,y_test = train_test_split(features, ratings, test_size=0.2, random_state=1)

#generating the linear regression model
model = LinearRegression()
#fitting the model to the training data
model.fit(X_train, y_train)

#print(model.score(X_test,y_test))

#sorting into a list of coefficients from most predicitve to least predictive
sorted_list = sorted(list(zip(['average_review_length','average_review_age'],model.coef_)),key = lambda x: abs(x[1]),reverse=True)
#print(sorted_list)

#generating subsets for further modelling
# subset of only average review sentiment
sentiment = ['average_review_sentiment']

# subset of all features that have a response range [0,1]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']

# subset of all features that vary on a greater range than [0,1]
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']

#subset of all features that vary with floats
# all features
all_features = binary_features + numeric_features

#testing and running predictions 1
#y_predicted = model.predict(X_test)
#plt.scatter(y_predicted,y_test)
#plt.show()

#Generating a function to more easily compare the performance of each model
# take a list of features to model as a parameter
def model_these_features(feature_list):
    
    #retrieving data from dataframe using .loc 
    #retrieving all rows from column 'stars' and storing in ratings
    ratings = df.loc[:,'stars']
    #retrieving all rows from list of feature names and storing in features
    features = df.loc[:,feature_list]
    
    #setting up training and test sets with test size of 20%
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    
    # these lines allow the model to work when we model on just one feature instead of multiple features.
    #finding length of the shape of the data frame (how many elements are in the tuple that represents the dimensionality of the dataframe)
    #dimensionality of the dataframe is the same as the dims of a matrix. (1,2) means 1 column 2 rows for example. 
    if len(X_train.shape) < 2:
        #reshaping X_train
        X_train = np.array(X_train).reshape(-1,1)
        #reshaping X_test
        X_test = np.array(X_test).reshape(-1,1)
    
    #Starting a linear regression model and fiting the X_train, y_train data to it
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    #Calculating the coefficient of determination (R^2) for X_train and X_test to test accuracy of model 
    print('Train Score:', model.score(X_train,y_train))
    print('Test Score:', model.score(X_test,y_test))
    
    # printing the model features and their corresponding coefficients, from most predictive to least predictive
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    
    #predicting y from X_test
    y_predicted = model.predict(X_test)
    
    #plotting results 
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()
    
features = df.loc[:,all_features]
ratings = df.loc[:,'stars']
X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)

#dataframe with mean, minimum, and maximum values for each feature
pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])

#array of feature values for danielles restaurant 
danielles_delicious_delicacies = np.array([0.14,0.35,0.7,0.1,0.1,0.16,35,1,1,2,1182,498,0.5,15,19,43,35,6,105,2006,11,162,0.2,45,52]).reshape(1,-1)

#prediction of yelp ratings for danielle's restaurant 
predicted_yelp_score = model.predict(danielles_delicious_delicacies)
print(predicted_yelp_score)

