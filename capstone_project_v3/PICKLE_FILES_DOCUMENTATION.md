# PICKLE FILES DOCUMENTATION
## 📦 ALL TRAINED MODEL FILES IN `model_data/`

### **Complete List of Generated Pickle Files:**

#### **1. 🧠 Collaborative Filtering Models**

**`user_item_matrix.pkl`**
- **Type:** Pandas DataFrame (603 × 97)
- **Content:** User-item rating matrix with normalized user IDs and item IDs
- **Purpose:** Core matrix for collaborative filtering algorithm
- **Size:** Contains rating data for 603 users across 97 products
- **Format:** Sparse matrix with 0 for unrated items, 1-5 for ratings

**`item_features.pkl`** 
- **Type:** NumPy array (97 × 100)
- **Content:** TF-IDF feature vectors for each product
- **Purpose:** Item similarity calculation using product review text
- **Features:** 100 TF-IDF features extracted from aggregated product reviews
- **Usage:** Input to KNN model for finding similar products

**`knn_model.pkl`**
- **Type:** Scikit-learn NearestNeighbors model
- **Algorithm:** KNN with cosine similarity, brute force algorithm
- **Parameters:** n_neighbors=20, metric='cosine'
- **Purpose:** Find similar items for recommendation generation
- **Training:** Fitted on item_features matrix

#### **2. 🗺️ Mapping Files**

**`user_mapping.pkl`**
- **Type:** Python dictionary
- **Content:** {username: user_id} mapping for 603 users
- **Purpose:** Convert usernames to matrix indices for recommendation lookup
- **Format:** {'rebecca': 0, 'john': 1, 'sarah': 2, ...}
- **Normalization:** All usernames stored in lowercase for consistency

**`item_mapping.pkl`**
- **Type:** Python dictionary  
- **Content:** {product_name: item_id} mapping for 97 products
- **Purpose:** Convert product names to matrix indices
- **Format:** {'K-Y Love Sensuality Pleasure Gel': 0, 'Johnson\'s Baby Bubble Bath': 1, ...}

#### **3. 📝 Text Processing Models**

**`tfidf_vectorizer.pkl`**
- **Type:** Scikit-learn TfidfVectorizer
- **Parameters:** max_features=100, stop_words='english', min_df=1
- **Purpose:** Convert product review text into numerical features
- **Vocabulary:** Trained on aggregated reviews for all 97 products
- **Usage:** Create item feature vectors for similarity calculation

**`product_features_tfidf.pkl`**
- **Type:** Scipy sparse matrix (97 × 100)
- **Content:** Pre-computed TF-IDF vectors for all products
- **Purpose:** Ready-to-use feature matrix for item similarity
- **Relationship:** Created using tfidf_vectorizer on product review text

#### **4. 📊 Core Data**

**`reviews_data.csv`** (Not pickle, but core data file)
- **Type:** CSV file with 2,571 rows
- **Columns:** name, reviews_username, reviews_rating, user_sentiment
- **Content:** Complete dataset with sentiment predictions
- **Sentiment Labels:** 'Positive' or 'Negative' classifications
- **Purpose:** Source data for both recommendation and sentiment analysis

**`sample_users.pkl`**
- **Type:** Python list
- **Content:** List of sample usernames for UI display
- **Purpose:** Provide example users for testing the web interface
- **Usage:** Displayed on homepage for user selection

### **📈 Model Performance & Statistics**

#### **Training Results:**
- **Total Users:** 603 unique users
- **Total Products:** 97 unique products  
- **Total Reviews:** 2,571 reviews with ratings and sentiment
- **Matrix Sparsity:** ~96% (typical for recommendation systems)
- **Sentiment Accuracy:** 94.20% (Random Forest classifier)
- **CF RMSE:** 1.0729 (Item-based collaborative filtering)

#### **File Size Summary:**
```
user_item_matrix.pkl     ~500KB    (603×97 sparse matrix)
item_features.pkl        ~80KB     (97×100 dense array)
knn_model.pkl           ~50KB     (Trained KNN model)
user_mapping.pkl        ~15KB     (603 user mappings)
item_mapping.pkl        ~8KB      (97 product mappings)
tfidf_vectorizer.pkl    ~25KB     (Trained vectorizer)
product_features_tfidf.pkl ~30KB   (97×100 TF-IDF matrix)
sample_users.pkl        ~2KB      (Sample user list)
reviews_data.csv        ~180KB    (2,571 reviews dataset)
```

### **🔄 Model Generation Process:**

1. **Data Processing:** Load and clean review data with sentiment analysis
2. **Matrix Creation:** Build user-item rating matrix from review data
3. **Feature Engineering:** Create TF-IDF features from product review text
4. **Model Training:** Train KNN model on item features for similarity
5. **Mapping Creation:** Generate user and item ID mappings for lookup
6. **Serialization:** Save all components as pickle files for deployment
7. **Deployment:** Load models in Flask app for real-time recommendations

### **✅ Submission Compliance:**
- **All Models Saved:** Every trained component properly pickled
- **Production Ready:** Models load successfully in Flask deployment  
- **Version Control:** All pickle files included in repository
- **Documentation:** Complete documentation provided for each file
- **Testing:** All models verified to work in production environment

**TOTAL: 9 pickle/data files supporting the complete recommendation system**