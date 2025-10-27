from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

from pathlib import Path
BASE = Path(__file__).resolve().parent

class ProductRecommendationSystem:
    """Simplified version of the recommendation system for Flask deployment"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.item_features = None
        self.user_mapping = None
        self.item_mapping = None
        self.reviews_df = None
        self.knn_model = None
        self.tfidf_vectorizer = None
        self.product_features_tfidf = None



    def load_models_and_data(self):
        """Load all necessary models and data with clear errors on Heroku."""
        try:
            # Helper to assert files exist
            def must_exist(path: Path):
                if not path.exists():
                    raise FileNotFoundError(f"Required file not found: {path}")
                return path

            # Build all paths relative to this file
            data_dir = BASE / "model_data"

            reviews_csv              = must_exist(data_dir / "reviews_data.csv")
            user_mapping_pkl         = must_exist(data_dir / "user_mapping.pkl")
            item_mapping_pkl         = must_exist(data_dir / "item_mapping.pkl")
            user_item_matrix_pkl     = must_exist(data_dir / "user_item_matrix.pkl")
            item_features_pkl        = must_exist(data_dir / "item_features.pkl")
            knn_model_pkl            = must_exist(data_dir / "knn_model.pkl")
            tfidf_vectorizer_pkl     = must_exist(data_dir / "tfidf_vectorizer.pkl")
            product_features_tfidf_p = must_exist(data_dir / "product_features_tfidf.pkl")

            # Load core data
            self.reviews_df = pd.read_csv(reviews_csv)

            # Load mappings
            with open(user_mapping_pkl, "rb") as f:
                self.user_mapping = pickle.load(f)
            with open(item_mapping_pkl, "rb") as f:
                self.item_mapping = pickle.load(f)

            # Load matrices / models
            self.user_item_matrix        = joblib.load(user_item_matrix_pkl)
            self.item_features           = joblib.load(item_features_pkl)
            self.knn_model               = joblib.load(knn_model_pkl)
            self.tfidf_vectorizer        = joblib.load(tfidf_vectorizer_pkl)
            self.product_features_tfidf  = joblib.load(product_features_tfidf_p)

            logger.info("✅ All models and data loaded successfully.")
            return True

        except Exception as e:
            # Log full traceback so you can see the exact reason in Heroku logs
            logger.error("❌ Error loading models/data: %s\n%s", str(e), traceback.format_exc())
            return False

    
    def get_user_recommendations(self, username, n_recommendations=20):
        """Get recommendations for a user using item-based collaborative filtering"""
        try:
            if username not in self.user_mapping:
                return None, f"User '{username}' not found in the system"
            
            user_id = self.user_mapping[username]
            
            # Get user's ratings
            user_ratings = self.user_item_matrix.iloc[user_id].copy()
            
            # Find items the user hasn't rated
            unrated_items = user_ratings[user_ratings == 0].index.tolist()
            
            if len(unrated_items) == 0:
                return None, "User has rated all available items"
            
            # Calculate predicted ratings for unrated items
            predictions = []
            
            for item_idx in unrated_items:
                # Find similar items to this item
                item_features_vector = self.item_features[item_idx].reshape(1, -1)
                distances, indices = self.knn_model.kneighbors(item_features_vector, n_neighbors=min(20, len(self.item_features)))
                
                # Calculate weighted rating prediction
                similar_items = indices[0][1:]  # Exclude the item itself
                similarities = 1 / (1 + distances[0][1:])  # Convert distances to similarities
                
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_item_idx, similarity in zip(similar_items, similarities):
                    user_rating = user_ratings.iloc[similar_item_idx]
                    if user_rating > 0:  # User has rated this similar item
                        weighted_sum += similarity * user_rating
                        similarity_sum += similarity
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                else:
                    predicted_rating = user_ratings[user_ratings > 0].mean()  # Fallback to user's average
                
                # Get item name
                reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
                item_name = reverse_item_mapping.get(item_idx, f"Item_{item_idx}")
                
                predictions.append({
                    'item_idx': item_idx,
                    'product_name': item_name,
                    'predicted_rating': round(predicted_rating, 2)
                })
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # Get top N recommendations
            top_recommendations = predictions[:n_recommendations]
            
            # Add ranking
            for i, rec in enumerate(top_recommendations, 1):
                rec['rank'] = i
            
            return top_recommendations, None
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return None, f"Error generating recommendations: {str(e)}"

class SentimentBasedRecommendationSystem:
    """Sentiment-enhanced recommendation system"""
    
    def __init__(self, rec_system, reviews_df):
        self.rec_system = rec_system
        self.reviews_df = reviews_df
    
    def analyze_product_sentiment(self, product_name):
        """Analyze sentiment for a specific product"""
        try:
            # Filter reviews for this product
            product_reviews = self.reviews_df[self.reviews_df['name'] == product_name]
            
            if len(product_reviews) == 0:
                return {
                    'product_name': product_name,
                    'total_reviews': 0,
                    'positive_reviews': 0,
                    'negative_reviews': 0,
                    'positive_percentage': 0.0,
                    'average_rating': 0.0,
                    'sentiment_score': 0.0
                }
            
            # Count sentiment distribution
            sentiment_counts = product_reviews['user_sentiment'].value_counts()
            positive_count = sentiment_counts.get('Positive', 0)
            negative_count = sentiment_counts.get('Negative', 0)
            total_count = len(product_reviews)
            
            # Calculate positive percentage
            positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0
            
            # Calculate average rating
            avg_rating = product_reviews['reviews_rating'].mean()
            
            # Create a sentiment score (combines positive percentage and average rating)
            sentiment_score = (positive_percentage * 0.7) + (avg_rating * 20 * 0.3)
            
            return {
                'product_name': product_name,
                'total_reviews': total_count,
                'positive_reviews': positive_count,
                'negative_reviews': negative_count,
                'positive_percentage': round(positive_percentage, 2),
                'average_rating': round(avg_rating, 2),
                'sentiment_score': round(sentiment_score, 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {product_name}: {str(e)}")
            return {
                'product_name': product_name,
                'total_reviews': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'positive_percentage': 0.0,
                'average_rating': 0.0,
                'sentiment_score': 0.0
            }
    
    def get_sentiment_enhanced_recommendations(self, username, n_recommendations=20, top_n=5):
        """Get top N recommendations based on sentiment analysis"""
        try:
            # Get initial recommendations
            initial_recs, error = self.rec_system.get_user_recommendations(username, n_recommendations)
            
            if error:
                return None, error
            
            # Analyze sentiment for each recommendation
            sentiment_results = []
            
            for rec in initial_recs:
                product_name = rec['product_name']
                sentiment_analysis = self.analyze_product_sentiment(product_name)
                
                # Combine recommendation data with sentiment analysis
                enhanced_rec = {**rec, **sentiment_analysis}
                sentiment_results.append(enhanced_rec)
            
            # Sort by sentiment score (highest first)
            sentiment_results.sort(key=lambda x: x['sentiment_score'], reverse=True)
            
            # Get top N based on sentiment
            top_sentiment_recs = sentiment_results[:top_n]
            
            # Calculate summary statistics
            avg_positive_percentage = np.mean([r['positive_percentage'] for r in sentiment_results]) if sentiment_results else 0
            avg_sentiment_score = np.mean([r['sentiment_score'] for r in sentiment_results]) if sentiment_results else 0
            
            return {
                'username': username,
                'top_sentiment_recommendations': top_sentiment_recs,
                'all_analyzed_products': sentiment_results,
                'analysis_summary': {
                    'total_analyzed': len(sentiment_results),
                    'top_selected': len(top_sentiment_recs),
                    'avg_positive_percentage': round(avg_positive_percentage, 2),
                    'avg_sentiment_score': round(avg_sentiment_score, 2)
                }
            }, None
            
        except Exception as e:
            logger.error(f"Error getting sentiment recommendations: {str(e)}")
            return None, f"Error generating sentiment recommendations: {str(e)}"

# Initialize global variables
rec_system = None
sentiment_rec_system = None

def initialize_systems():
    """Initialize the recommendation systems"""
    global rec_system, sentiment_rec_system
    
    try:
        # Initialize recommendation system
        rec_system = ProductRecommendationSystem()
        
        # Load models and data
        if not rec_system.load_models_and_data():
            logger.error("Failed to load models and data")
            return False
        
        # Initialize sentiment system
        sentiment_rec_system = SentimentBasedRecommendationSystem(rec_system, rec_system.reviews_df)
        
        logger.info("Recommendation systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing systems: {str(e)}")
        return False

# --- Heroku/Gunicorn: initialize on import ---
try:
    initialized = initialize_systems()
    if not initialized:
        app.logger.error("initialize_systems() returned False during import")
    else:
        app.logger.info("initialize_systems() succeeded during import")
except Exception:
    app.logger.exception("initialize_systems() raised during import")



@app.route('/')
def index():
    """Home page"""
    # Get some sample usernames for display
    sample_users = []
    if rec_system and rec_system.user_mapping:
        import random
        all_users = list(rec_system.user_mapping.keys())
        sample_users = random.sample(all_users, min(10, len(all_users)))
    
    return render_template('index.html', sample_users=sample_users)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get sentiment-enhanced recommendations for a user"""
    try:
        username = (request.form.get('username') or '').strip().lower()
        if not username:
            app.logger.warning("recommend: empty username")
            # If the request is expecting JSON (e.g., via fetch/axios), return JSON
            if request.headers.get("Accept", "").startswith("application/json"):
                return jsonify({"ready": True, "error": "username required"}), 400
            flash('Please enter a username')
            return redirect(url_for('index'))

        # Fail fast if systems aren't ready
        if rec_system is None or sentiment_rec_system is None:
            app.logger.error("recommend: systems not initialized")
            if request.headers.get("Accept", "").startswith("application/json"):
                return jsonify({
                    "ready": False,
                    "error": "Recommendation system not initialized on server"
                }), 503
            flash('Recommendation system not initialized')
            return redirect(url_for('index'))

        # Validate user
        if username not in rec_system.user_mapping:
            app.logger.warning("recommend: unknown username '%s'", username)
            if request.headers.get("Accept", "").startswith("application/json"):
                return jsonify({"ready": True, "error": f"Username '{username}' not found"}), 404
            flash(f"Username '{username}' not found in our system")
            return redirect(url_for('index'))

        # Existing recommendation logic
        results, error = sentiment_rec_system.get_sentiment_enhanced_recommendations(
            username, n_recommendations=20, top_n=5
        )
        if error:
            app.logger.error("recommend: pipeline error: %s", error)
            if request.headers.get("Accept", "").startswith("application/json"):
                return jsonify({"ready": True, "error": str(error)}), 500
            flash(f"Error: {error}")
            return redirect(url_for('index'))

        return render_template(
            'results.html',
            username=username,
            recommendations=results['top_sentiment_recommendations'],
            summary=results['analysis_summary']
        )

    except Exception as e:
        app.logger.exception("recommend: unhandled exception")
        if request.headers.get("Accept", "").startswith("application/json"):
            return jsonify({"ready": True, "error": str(e)}), 500
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('index'))


@app.route('/api/recommend/<username>')
def api_recommendations(username):
    """API endpoint for getting recommendations"""
    try:
        username = username.strip().lower()
        
        if not rec_system or not sentiment_rec_system:
            return jsonify({'error': 'Recommendation system not initialized'}), 500
        
        if username not in rec_system.user_mapping:
            return jsonify({'error': f"Username '{username}' not found"}), 404
        
        # Get sentiment-enhanced recommendations
        results, error = sentiment_rec_system.get_sentiment_enhanced_recommendations(
            username, n_recommendations=20, top_n=5
        )
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in API endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'rec_system_loaded': rec_system is not None,
            'sentiment_system_loaded': sentiment_rec_system is not None
        }
        
        if rec_system:
            status['total_users'] = len(rec_system.user_mapping) if rec_system.user_mapping else 0
            status['total_products'] = len(rec_system.item_mapping) if rec_system.item_mapping else 0
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route("/api/ready")
def api_ready():
    ok = (rec_system is not None) and (sentiment_rec_system is not None)
    return jsonify({"ready": ok}), (200 if ok else 500)
    


if __name__ == '__main__':
    # Initialize systems before starting the app
    if initialize_systems():
        logger.info("Starting Flask application...")
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    else:

        logger.error("Failed to initialize recommendation systems. Exiting.")





