from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_excel('UserData.xlsx')
    a = [x for x in request.form.values()]
    df = df.append(pd.Series(a, index=['Name', 'SchoolFees', 'Facilities', 'StaffQuality', 
                                       'GradingSystem',
                                       'Distance', 'PercentagePass', 'GovtRanking', 
                                       'Infrastructures']), ignore_index=True )
    df = df.fillna(' ')
    item_similarity = cosine_similarity(df.drop('Name', axis=1))
    b = int(df.index.values[-1])
    similarity = list(enumerate(item_similarity[b]))
    sorted_school_of_likes = sorted(similarity, key=lambda x:x[1],  reverse=True)
    sorted_school_of_likes = sorted_school_of_likes[1:6]
    lucid_index = [i[0] for i in sorted_school_of_likes]
    
    return render_template('index.html', prediction_text = 'recommended schools are :\n {} \n {} \n {} \n {} \n {} \n'.format(df.Name.iloc[lucid_index[0]],df.Name.iloc[lucid_index[1]],
                                                                                                 df.Name.iloc[lucid_index[2]],df.Name.iloc[lucid_index[3]],df.Name.iloc[lucid_index[4]]))

if __name__ == "__main__":
    app.run(debug=True)
















