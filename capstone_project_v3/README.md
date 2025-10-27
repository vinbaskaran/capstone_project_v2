# Flask Sentiment-Enhanced Recommendation System

## 🚀 Project Overview
A Flask web application that provides **TOP 5 SENTIMENT-ENHANCED RECOMMENDATIONS** for any user. The system combines collaborative filtering with sentiment analysis to deliver personalized product recommendations.

## ✅ What's Been Created

### 1. Core Application Files
- **`app.py`** - Main Flask application with recommendation logic
- **`templates/index.html`** - Beautiful user interface for entering usernames
- **`templates/results.html`** - Professional results display with sentiment metrics

### 2. Model & Data Files (in `model_data/` folder)
- **`reviews_data.csv`** - Review data with sentiment predictions (2,570 records)
- **`user_mapping.pkl`** - User ID mappings (603 users)
- **`item_mapping.pkl`** - Product ID mappings (97 products)
- **`user_item_matrix.pkl`** - User-item rating matrix (603 x 97)
- **`item_features.pkl`** - Product feature vectors
- **`knn_model.pkl`** - K-Nearest Neighbors model for collaborative filtering
- **`tfidf_vectorizer.pkl`** & **`product_features_tfidf.pkl`** - Text feature components

### 3. Heroku Deployment Files
- **`requirements.txt`** - Python dependencies
- **`Procfile`** - Heroku process configuration
- **`runtime.txt`** - Python version specification
- **`.gitignore`** - Git ignore rules

## 🌟 Key Features

### 1. **Sentiment-Enhanced Recommendations**
- Combines collaborative filtering with sentiment analysis
- Prioritizes products with high positive sentiment scores
- Shows sentiment metrics for each recommendation

### 2. **Professional Web Interface**
- Modern, responsive design with Bootstrap
- Interactive user selection from sample users
- Detailed recommendation cards with multiple metrics

### 3. **Multiple Access Methods**
- Web interface at `/`
- REST API at `/api/recommend/<username>`
- Health check endpoint at `/health`

### 4. **Rich Recommendation Data**
Each recommendation includes:
- **Predicted Rating** (your likely rating)
- **Community Rating** (average user rating)
- **Sentiment Score** (0-100 based on review sentiment)
- **Positive Review Percentage**
- **Total Review Count**
- **Original Collaborative Filtering Rank**

## 🔧 Local Testing
Your app is running successfully at `http://localhost:5000`

## 🚀 Heroku Deployment Steps

### 1. Prepare for Deployment
```bash
# Initialize git repository (if not already)
git init
git add .
git commit -m "Initial commit - Flask recommendation app"
```

### 2. Create Heroku App
```bash
# Install Heroku CLI first (https://devcenter.heroku.com/articles/heroku-cli)
heroku login
heroku create your-app-name-here
```

### 3. Deploy to Heroku
```bash
git push heroku main
```

### 4. Verify Deployment
```bash
heroku open
heroku logs --tail
```

## 📊 Technical Specifications

### System Architecture
- **Backend**: Flask 2.3.3 with Python 3.11
- **ML Libraries**: scikit-learn, pandas, numpy
- **Models**: Item-based Collaborative Filtering + Sentiment Analysis
- **Frontend**: Bootstrap 5.1.3 with responsive design

### Performance Metrics
- **Dataset**: 2,570 reviews across 97 products
- **Users**: 603 unique users
- **Model Accuracy**: 94.20% sentiment classification
- **Recommendation Engine**: Item-based CF with 1.0729 RMSE

### API Endpoints
- `GET /` - Main web interface
- `POST /recommend` - Get recommendations via form
- `GET /api/recommend/<username>` - JSON API endpoint
- `GET /health` - Health check endpoint

## 🎯 How It Works

1. **User Input**: Enter any username from the system
2. **Collaborative Filtering**: Find similar users and predict ratings
3. **Sentiment Analysis**: Analyze review sentiment for each recommended product
4. **Smart Ranking**: Combine predicted ratings with sentiment scores
5. **Top 5 Selection**: Return the best products based on sentiment enhancement

## 🔍 Sample Users to Try
The system includes 603 users. Some examples you can test with:
- rebecca, john, sarah, michael (and many more available in the interface)

## 📈 Sentiment Scoring Algorithm
```
Sentiment Score = (Positive Review % × 0.7) + (Average Rating × 20 × 0.3)
```
This ensures products with both high ratings AND positive sentiment get prioritized.

## 🛠️ Future Enhancements
- Add user authentication
- Implement real-time model updates
- Add more recommendation algorithms
- Include product images and descriptions
- Add user feedback collection

## 📞 Support
Your sentiment-enhanced recommendation system is ready for production deployment on Heroku!

---
**Total Development Time**: Complete end-to-end system built and tested ✅
**Status**: Ready for Heroku deployment 🚀