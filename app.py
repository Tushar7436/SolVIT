from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import traceback


app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_paths = {
    "spam": os.path.join(BASE_DIR, "Book1.xlsx"),
    "classification": os.path.join(BASE_DIR, "final_railway_queries.xlsx"),
    "analysis": os.path.join(BASE_DIR, "binary_categorized_words.xlsx"),
    "priority": os.path.join(BASE_DIR, "boss level queries.xlsx")
}

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_excel(file_path)
    df = df.drop_duplicates()
    df = df[~df['Query'].str.match(r'^\s*[:.,;\-]*\s*$', na=False)]
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def spam_detection(query):
    df = load_and_clean_data(data_paths["spam"])
    x = df["Query"]
    y = df["SPAM/NOT_SPAM"]
    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.90, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    svm_model = SVC(kernel="linear")
    svm_model.fit(X_train_tfidf, y_train)
    
    X_transformed = vectorizer.transform([query])
    y_pred = svm_model.predict(X_transformed)
    return y_pred[0]


def query_classification(query):
    df = load_and_clean_data(data_paths["classification"])
    df = df.copy()  # Ensure no chained assignment
    df['Domain'] = df['Domain'].fillna('Unknown')
    df["Combined_Label"] = df["Sub-topic"].astype(str) + " | " + df["Domain"].astype(str)
    x = df["Query"]
    y = df["Combined_Label"]
    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.90, random_state=42)
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    
    X_transformed = vectorizer.transform([query])
    y_pred = rf_model.predict(X_transformed)
    return y_pred[0]

def query_analysis(query):
    df = load_and_clean_data(data_paths["analysis"])
    print("✅ Loaded Analysis Data (First 5 Rows):\n", df.head())  
    print("📂 Available Columns in Analysis Data:", df.columns.tolist())  
    
    # Ensure 'Query' column exists
    if "Query" not in df.columns:
        return "Error: 'Query' column is missing in the Excel file"

    print("Loaded Analysis Data (First 5 Rows):\n", df.head())  
    
    # Ensure 'Query' column exists
    if "Query" not in df.columns:
        return "Error: 'Query' column is missing in the Excel file"
    
    # Ensure columns exist
    if "Word/Query" not in df.columns or "Good" not in df.columns or "Bad" not in df.columns or "Extreme Bad" not in df.columns:
        return "Error: Missing expected columns in Excel file"

    good_words = set(df[df["Good"] == 1]["Word/Query"].dropna().str.lower())
    bad_words = set(df[df["Bad"] == 1]["Word/Query"].dropna().str.lower())
    extreme_bad_words = set(df[df["Extreme Bad"] == 1]["Word/Query"].dropna().str.lower())

    query_words = query.lower().split()
    category_counts = {
        "Good": sum(1 for word in query_words if word in good_words),
        "Bad": sum(1 for word in query_words if word in bad_words),
        "Extreme Bad": sum(1 for word in query_words if word in extreme_bad_words)
    }
    
    return "Normal Statement" if all(count == 0 for count in category_counts.values()) else max(category_counts, key=category_counts.get)


def query_priority(query):
    df = load_and_clean_data(data_paths["priority"])
    label_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df["Priority"] = df["Priority"].map(label_mapping)
    
    X_train, _, y_train, _ = train_test_split(df["Query"], df["Priority"], test_size=0.1, random_state=42)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)
    
    X_transformed = vectorizer.transform([query])
    prediction_numeric = rf_model.predict(X_transformed)[0]
    inverse_label_mapping = {0: "Low", 1: "Medium", 2: "High"}
    return inverse_label_mapping[prediction_numeric]

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/analyze', methods=['POST'])
def analyze_query():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request, JSON expected"}), 400
        
        data = request.get_json()
        query = data.get('query', "").strip()
        
        if not query:
            return jsonify({"error": "Query is missing or empty"}), 400
        
        results = {}
        try:
            results['spam'] = spam_detection(query)
        except Exception as e:
            print("Error in spam_detection:", str(e))
            results['spam'] = f"Error: {str(e)}"
        
        try:
            results['classification'] = query_classification(query)
        except Exception as e:
            print("Error in query_classification:", str(e))
            results['classification'] = f"Error: {str(e)}"
        
        try:
            results['analysis'] = query_analysis(query)
        except Exception as e:
            print("Error in query_analysis:", str(e))
            results['analysis'] = f"Error: {str(e)}"
        
        try:
            results['priority'] = query_priority(query)
        except Exception as e:
            print("Error in query_priority:", str(e))
            results['priority'] = f"Error: {str(e)}"
        
        return jsonify(results)
    
    except Exception as e:
        print("🔥 ERROR:", str(e))  
        traceback.print_exc()   
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Current Working Directory:", os.getcwd())
    print("Files in Directory:", os.listdir(BASE_DIR))
    app.run(debug=True, host="127.0.0.1", port=5001)
