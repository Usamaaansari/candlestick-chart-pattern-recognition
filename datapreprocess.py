# from playwright.sync_api import sync_playwright
# import keras.utils as image
# import requests
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# from keras.models import load_model
# import os
# from collections import Counter,defaultdict

########################## STEP 1 DOWNLOAD NIFTY500 AND S&P 500 STOCKS ABD PREPROCESS IT BY CHNAGING THE COLUMN NAME AND SYMBOL ###########################
#nifty50_stocks=pd.read_csv('nifty500.csv')
# nifty50_stocks['Symbol'] = nifty50_stocks['Symbol'].astype(str) + '.NS'

# with sync_playwright() as p:
#   browser = p.chromium.launch()
#   context = browser.new_context()
#   page = context.new_page()

#   try:
#     page.goto('https://stockanalysis.com/list/sp-500-stocks/')

#     # Wait for the table to be loaded (adjust selector as needed)
#     page.wait_for_selector('table')

#     # Get the table element
#     table_element = page.query_selector('table')

#     # Extract table data using synchronous methods
#     table_data = table_element.inner_text().split('\n')

#     # Process the table data (remove header row if necessary)
#     processed_data = [row.strip().split('\t') for row in table_data[1:]]  # Assuming tab-delimited data

#     # Create a DataFrame from the processed data
#     df = pd.DataFrame(processed_data)  # Assuming consistent column names in first row

#     # Save DataFrame as CSV
#     df.to_csv('sp500_data.csv', index=False)  # Save without index column

#     print("Data saved to sp500_data.csv")
#   except Exception as e:
#     print(f"Error occurred: {e}")
#   finally:
#     browser.close()

############################## STEP 2  FETCHING HISTORIC DATA AND SAVING AS PNG IMAGE ####################################################################
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# import os

# def fetch_historic_data(stock_symbol, duration='3mo'):
#     try:
#         # Fetch historic data using yfinance
#         data = yf.download(stock_symbol, period=duration)
#         return data
#     except Exception as e:
#         print(f"Failed to fetch data for {stock_symbol}: {e}")
#         return None
    
    
# ################ AFTER FETCHING DATA PLOT LINE PLOT USING 'CLOSE' VALUE OF STOCK AND SAVING IT IN DIRECTORY FOR PREDCITION###     
# def plot_and_save_data(historic_data, stock_name, stock_symbol, directory):
#     # Plot historic data
#     plt.figure(figsize=(10, 6))
#     plt.plot(historic_data['Close'], label='Close')
#     plt.title(f"{stock_name} ({stock_symbol}) - Historic Data")
#     plt.xlabel('Date')
#     plt.ylabel('Price (INR)')
#     plt.legend()
#     plt.grid(False)
#     plt.xticks(rotation=45)

#     # Ensure the directory exists
#     os.makedirs(directory, exist_ok=True)
    
#     # Save plot as PNG image
#     plt.savefig(os.path.join(directory, f"{stock_symbol}_historic_data.png"))
#     plt.close()

# # Directory to save images
# base_img_dir = './csv_files'

# # Iterate over each CSV file in the directory
# for file_name in os.listdir(base_img_dir):
#     if file_name.endswith('.csv'):
#         csv_dir = os.path.splitext(file_name)[0]  # Extract CSV file name without extension
#         file_path = os.path.join(base_img_dir, file_name)
#         df = pd.read_csv(file_path)
        
#         # Create a separate directory for each CSV file
#         img_dir = os.path.join(base_img_dir, csv_dir)
        
#         # Perform fetch_historic_data and plot_and_save_data for each stock in the CSV file
#         for index, row in df.iterrows():
#             stock_symbol = row['Symbol']
#             stock_name = row['Company Name']
            
#             historic_data = fetch_historic_data(stock_symbol)
#             if historic_data is not None:
#                 plot_and_save_data(historic_data, stock_name, stock_symbol, img_dir)


############### STEP 3 READING THE IMAGES AND PREDICTING IT'S CLASS AND STORING ALL INFORMATION IN DATABASE ############################

# import os
# import pandas as pd
# import tensorflow as tf
# from collections import defaultdict, Counter
# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, exists
# from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Define models using declarative base
# Base = declarative_base()
# class PatternCount(Base):
#     __tablename__ = 'nifty500_patterns'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     count = Column(Integer)
#     pattern = Column(String(255))

# class CompanySymbol(Base):
#     __tablename__ = 'nifty500_companies'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     company_name = Column(String(255))
#     symbol = Column(String(255))
#     pattern_id = Column(Integer, ForeignKey('nifty500_patterns.id'))
#     pattern = relationship('PatternCount')

# class SP500PatternCount(Base):
#     __tablename__ = 'sp500_patterns'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     count = Column(Integer)
#     pattern = Column(String(255))

# class SP500CompanySymbol(Base):
#     __tablename__ = 'sp500_companies'
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     company_name = Column(String(255))
#     symbol = Column(String(255))
#     pattern_id = Column(Integer, ForeignKey('sp500_patterns.id'))
#     pattern = relationship('SP500PatternCount')

# # Database setup
# DATABASE_URL = 'mysql+pymysql://root:root@localhost/users'
# engine = create_engine(DATABASE_URL)
# Session = sessionmaker(bind=engine)
# Base.metadata.create_all(engine)

# def load_and_preprocess_image(img_path, target_size):
#     img = load_img(img_path, target_size=target_size)
#     img_array = img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch
#     img_array /= 255.0  # Normalize to [0,1]
#     return img_array

# def predict_image(model, img_array, class_indices):
#     predictions = model.predict(img_array)
#     predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
#     class_labels = {v: k for k, v in class_indices.items()}
#     predicted_class_label = class_labels[predicted_class_index]
#     return predicted_class_label

# def process_and_insert_data(model_path, csv_path, image_dir, db_url,pattern_class, company_class, table_name):
#     # Load the model
#     model = tf.keras.models.load_model(model_path)

#     # Initialize data structures
#     stock_name = defaultdict(list)
#     pattern_counter = Counter()
#     symbol_dict = defaultdict(list)

#     # Load stock data
#     nifty50_stocks = pd.read_csv(csv_path)

#     # Database session
#     engine = create_engine(db_url)
#     Session = sessionmaker(bind=engine)
#     session = Session()

#     # Uncomment the following line if tables are not created

#     class_indices = {
#     'double_bottom': 0,
#     'double_top': 1,
#     'head_and_shoulders': 2,
#     'inverse_head_and_shoulders': 3,
#     'triple_bottom': 4,
#     'triple_top': 5}

#     # Iterate over images
#     for i in os.listdir(image_dir):
#         img_path = os.path.join(image_dir, i)
#         target_size = (480, 480)  # Should match the input size of your model
#         img_array = load_and_preprocess_image(img_path, target_size)

#         stock_sym = i.split('_historic')[0]
#         comp_name = nifty50_stocks[nifty50_stocks['Symbol'] == stock_sym]['Company Name'].values[0]

#         # Predict the class of the image
#         predicted_class_label = predict_image(model, img_array, class_indices)
#         pattern_counter[predicted_class_label] += 1
#         stock_name[predicted_class_label].append(comp_name)
#         symbol_dict[predicted_class_label].append(stock_sym)

#     # Insert patterns and counts into PatternCount table
#     for predicted_class_label, count in pattern_counter.items():
#         # Check if the pattern already exists
#         pattern_count = session.query(pattern_class).filter_by(pattern=predicted_class_label).first()
#         if pattern_count is None:
#             pattern_count = pattern_class(count=count, pattern=predicted_class_label)
#             session.add(pattern_count)
#             session.commit()
#         else:
#             # Update the count if the pattern already exists
#             pattern_count.count += count
#             session.commit()
#         pattern_id = pattern_count.id

#         # Insert company name and symbol into CompanySymbol table
#         for comp_name, symbol in zip(stock_name[predicted_class_label], symbol_dict[predicted_class_label]):
#             # Check if the company symbol already exists
#             company_symbol = session.query(company_class).filter_by(symbol=symbol, pattern_id=pattern_id).first()
#             if company_symbol is None:
#                 company_symbol = company_class(company_name=comp_name, symbol=symbol, pattern_id=pattern_id)
#                 session.add(company_symbol)
#                 session.commit()

#     # Close the session
#     session.close()

#     print(f"Data inserted successfully for {table_name}!")
    
# ################ Defining Parameters #############################################
# csv_path_nifty = './csv_files/nifty500.csv'
# nifty_image_dir = './csv_files/nifty500/'

# model_path='final_project_image_classification.h5'
# db_url= 'mysql+pymysql://root:root@localhost/users'

# csv_path_snp = './csv_files/sp500.csv'
# sp_image_dir = './csv_files/sp500/'

# # Call the function ###  FOR NIFTY ############################
# process_and_insert_data(model_path, csv_path_nifty, nifty_image_dir, db_url,PatternCount,CompanySymbol,'nifty500')

# ##################### FOR S&P ###########################
# process_and_insert_data(model_path, csv_path_snp, sp_image_dir, db_url,SP500PatternCount,SP500CompanySymbol,'S&P')

