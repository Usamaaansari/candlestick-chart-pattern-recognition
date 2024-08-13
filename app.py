from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message
from fetch_patterns import get_pattern_counts, get_stocks_by_pattern
from sqlalchemy import create_engine,text
import requests
import plotly.graph_objs as go
import json
import plotly


app=Flask(__name__)

app.secret_key = 'asd@123'  # Change this to something secure
NEWS_API_KEY = '34439ae4d88c46f7b24726ac8eeac87f'


# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Change this to your SMTP server
app.config['MAIL_PORT'] = 465  #Change this to the port your SMTP server uses
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

app.config['MAIL_USERNAME'] = 'ansariusama688@gmail.com'
app.config['MAIL_PASSWORD'] = 'gfng aqvv ueph utrj'
app.config['MAIL_DEFAULT_SENDER'] = 'ansariusama688@gmail.com'
mail = Mail(app)


#########################Loading and Preprocessing of Data############################################

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def get_indian_business_news():
    url = f'https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles'] if 'articles' in news_data else []


############################## ALL ROUTES #######################################################################
@app.route('/')
def index():
    articles = get_indian_business_news()
    # Get the first three articles
    article1 = articles[0] if len(articles) > 0 else None
    article2 = articles[1] if len(articles) > 1 else None
    article3 = articles[2] if len(articles) > 2 else None
    return render_template('index.html', article1=article1, article2=article2, article3=article3)


@app.route('/signup',methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['txt']
        email = request.form['email']
        sql='mysql+pymysql://root:root@localhost:3306/users'
        engine=create_engine(sql)
        conn=engine.connect()
        # Check if email already exist in database
        df=conn.execute(text(f"select * from myusers where email = '{email}'"))
        user = df.fetchone()
        if user:
            flash('Email already exists!')
        else:
            conn.execute(text(f"INSERT INTO myusers (username, password, email) VALUES ('{username}','{email}')"))
            conn.commit()
            conn.close()
            send_welcome_email(email)
            flash('Sign up successful! Check your email for a welcome message.')
    return render_template("signup.html")


def send_welcome_email(email):
    msg = Message('Welcome to Our App!', recipients=[email])
    msg.body = 'Thank you for signing up to our app!'
    mail.send(msg)

@app.route('/submit_feedback', methods=['POST','GET'])
def submit_feedback():
    name = request.form['name']
    email = request.form['user_email']
    message = request.form['Message']
    sql='mysql+pymysql://root:root@localhost:3306/users'
    engine=create_engine(sql)
    conn=engine.connect()
    conn.execute(text(f"INSERT INTO messages (name, email, message) VALUES ('{name}','{email}','{message}')"))
    conn.commit()
    flash('Message sent successfully!')
    return redirect(url_for('index'))



@app.route('/patterns/<market>')
def patterns(market):
    table_name = 'nifty500_patterns' if market == 'nse' else 'sp500_patterns'
    pattern_counter = get_pattern_counts(table_name)
    return render_template('patterns.html', pattern_counter=pattern_counter,market=market)

@app.route('/patterns/<market>/<pattern>')
def show_stocks(market, pattern):
    table_name = 'nifty500_companies' if market == 'nse' else 'sp500_companies'
    stocks = get_stocks_by_pattern(table_name, pattern)
    return render_template('stocks.html', stocks=stocks, pattern=pattern)

@app.route('/news1')
def news1():
    news_articles = get_indian_business_news()
    
    return render_template('news1.html',news_articles=news_articles)

@app.route('/chart/<symbol>')
def chart(symbol):
    # Fetch historical stock data from yfinance
  # Change to your desired symbol
    data = yf.download(symbol, period="3mo")

    # Create candlestick trace
    trace = go.Candlestick(x=data.index,
                           open=data['Open'],
                           high=data['High'],
                           low=data['Low'],
                           close=data['Close'])

    # Create layout
    layout = go.Layout(title=f'Interactive Candlestick Chart for {symbol}',
                       xaxis=dict(title='Date'),
                       yaxis=dict(title='Price'))

    # Create figure
    fig = go.Figure(data=[trace], layout=layout)

    return render_template('chart.html', plot=json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))


if __name__=='__main__':
    app.run(debug=True)