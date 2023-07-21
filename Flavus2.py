import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from surprise import Dataset, Reader, SVD, accuracy
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from fbprophet import Prophet
from textblob import TextBlob
import matplotlib.pyplot as plt
import pickle
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sklearn.ensemble import IsolationForest
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from bs4 import BeautifulSoup

# Obtain URLs for datasets
sales_data_url = input("Please enter the URL for your sales data: ")
org_data_url = input("Please enter the URL for your internal organization data: ")
inventory_data_url = input("Please enter the URL for your inventory data: ")
feedback_data_url = input("Please enter the URL for your customer feedback data: ")

# Download data
sales_data = pd.read_csv(sales_data_url)
org_data = pd.read_csv(org_data_url)
inventory_data = pd.read_csv(inventory_data_url)
feedback_data = pd.read_csv(feedback_data_url)

# Data preprocessing: remove outliers and handle missing values
sales_data = sales_data[(np.abs(zscore(sales_data)) < 3).all(axis=1)]
sales_data.fillna(sales_data.mean(), inplace=True)

# Basic data preprocessing
scaler = StandardScaler()

# Encoding categorical features in sales data
sales_data = pd.get_dummies(sales_data, columns=['item'])

# Scaling numerical features
sales_data['rating'] = scaler.fit_transform(sales_data['rating'].values.reshape(-1, 1))
org_data['features'] = scaler.fit_transform(org_data['features'].values.reshape(-1, 1))

# Split datasets for training and testing
sales_train, sales_test = train_test_split(sales_data, test_size=0.2, random_state=42)
org_train, org_test = train_test_split(org_data, test_size=0.2, random_state=42)

# Sentiment Analysis on customer feedback
feedback_data['sentiment'] = feedback_data['feedback'].apply(lambda text: TextBlob(text).sentiment.polarity)
positive_feedback = feedback_data[feedback_data.sentiment > 0]
negative_feedback = feedback_data[feedback_data.sentiment < 0]
neutral_feedback = feedback_data[feedback_data.sentiment == 0]
print(f"Positive feedback count: {len(positive_feedback)}")
print(f"Negative feedback count: {len(negative_feedback)}")
print(f"Neutral feedback count: {len(neutral_feedback)}")

# Prepare data for Surprise library
reader = Reader(rating_scale=(sales_train['rating'].min(), sales_train['rating'].max()))
data = Dataset.load_from_df(sales_train[['user', 'item', 'rating']], reader)

# Boosting Sales: train a recommender system on sales data
trainset = data.build_full_trainset()
algo = SVD()
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

# Use the best model
algo = gs.best_estimator['rmse']
algo.fit(trainset)

# Predict on the test set
testset = list(zip(sales_test['user'].values, sales_test['item'].values, sales_test['rating'].values))
predictions = algo.test(testset)
print("Sales model RMSE: ", accuracy.rmse(predictions))
print("Sales model MAE: ", accuracy.mae(predictions))

# Evaluate clustering for Internal Organization
sil_scores = []
K = range(2,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X_train)
    sil_scores.append(silhouette_score(X_train, km.labels_))

# Plot silhouette scores
plt.figure(figsize=(6,6))
plt.plot(K, sil_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

# Perform regression on org data
lr = LinearRegression().fit(X_train, y_train)
preds = lr.predict(X_test)

# Evaluate the model
rmse = np.sqrt(np.mean((preds - y_test)**2))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print("Organization model RMSE: ", rmse)
print("Organization model MAE: ", mae)
print("Organization model R2 Score: ", r2)

# Create PDF report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size = 15)
pdf.cell(200, 10, txt = f"Sales model RMSE: {accuracy.rmse(predictions)}", ln = True, align = 'C')
pdf.cell(200, 10, txt = f"Sales model MAE: {accuracy.mae(predictions)}", ln = True, align = 'C')
pdf.cell(200, 10, txt = f"Organization model RMSE: {rmse}", ln = True, align = 'C')
pdf.cell(200, 10, txt = f"Organization model MAE: {mae}", ln = True, align = 'C')
pdf.cell(200, 10, txt = f"Organization model R2 Score: {r2}", ln = True, align = 'C')
pdf.output("Report.pdf")

# Inventory Visibility: perform time series forecasting on inventory data
m = Prophet()
m.fit(inventory_data)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast.plot(y='yhat', figsize=(10,6), title='Inventory Forecast')

# Perform anomaly detection on the inventory data
clf = IsolationForest(contamination=0.01)
preds = clf.fit_predict(inventory_data)
anomalies = inventory_data[preds == -1]

# If any anomalies are detected, email alert to the administrator
if len(anomalies) > 0:
    msg = MIMEMultipart()
    msg['From'] = "alert@flavus.com"
    msg['To'] = "admin@flavus.com"
    msg['Subject'] = "Flavus Alert: Detected Anomalies in Inventory Data"
    body = f"Flavus has detected {len(anomalies)} anomalies in your inventory data. Please check the attached report for details."
    msg.attach(MIMEText(body, 'plain'))
    filename = "Anomalies_Report.pdf"
    attachment = open(filename, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= "+filename)
    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("alert@flavus.com", "password")
    text = msg.as_string()
    server.sendmail("alert@flavus.com", "admin@flavus.com", text)
    server.quit()

# Customer Engagement: train the recommender model on full sales data
reader = Reader(rating_scale=(sales_data['rating'].min(), sales_data['rating'].max()))
full_data = Dataset.load_from_df(sales_data[['user', 'item', 'rating']], reader)
full_trainset = full_data.build_full_trainset()
algo.fit(full_trainset)

# Save the trained model
filename = 'trained_market_model.sav'
pickle.dump(algo, open(filename, 'wb'))
print("Customer engagement model trained and saved as 'trained_market_model.sav'")

# Competitor Analysis
def competitor_analysis():
    # Scraping competitor data
    competitor_data = requests.get("https://competitor.com/data")
    if competitor_data.status_code == 200:
        # Parsing competitor data
        soup = BeautifulSoup(competitor_data.content, 'html.parser')
        # Implement analysis logic here
        # For example, extract relevant information from the HTML and perform data analysis
        # This has been left as a placeholder to be edited by the end user due to variations in competitor data. 
        # Example: Extracting product names from competitor website
        product_names = soup.find_all('div', class_='product-name')
        for product in product_names:
            print(product.text)
    else:
        print("Failed to fetch competitor data. Please try again later.")

# Handling user queries
def chatbot_response(query):
    # Implement generic logic to handle user queries
    if "sales" in query.lower():
        return "Our sales have increased by 10% compared to the previous quarter."
    elif "inventory" in query.lower():
        return "The current inventory level is 500 units."
    elif "customer feedback" in query.lower():
        return "We have received positive feedback from our customers regarding the new product."
    else:
        return "I'm sorry, I couldn't understand your query. Can you please rephrase?"

while True:
    user_query = input("You: ")
    if user_query.lower() == 'exit':
        break
    else:
        response = chatbot_response(user_query)
        print("Chatbot: ", response)

    # Perform Competitor Analysis
    competitor_analysis()
