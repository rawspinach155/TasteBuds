import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import creds

# Get API key using Google's Secret Storage
SEARCH_URL = "https://api.yelp.com/v3/businesses/search"
HEADERS = {
    "Authorization": f"Bearer {creds.API_KEY}",
    "Accept": "application/json"
}

# Mapping meal types to Yelp categories
MEAL_TYPE_CATEGORIES = {
    "breakfast": "breakfast, brunch, cafe, coffee",
    "brunch": "breakfast, brunch, cafe, coffee",
    "lunch": "cafe, deli, sandwiches lunch",
    "dinner": "dinner, restaurant, steakhouses, bar",
    "dessert": "dessert, bakery, ice cream, boba, sweets",
    "snack": "snack, fast food, finger food",
    "hangover food": "greasy, burgers, comfort food, heavy, hot dogs, fried",
    "cheat meal": "greasy, burgers, comfort food, heavy, hot dogs, fried",
    "healthy": "salad, bowls, healthy, vegetarian, vegan, vegetable",
    "vegan & vegetarian": "salad, bowls, healthy, vegetarian, vegan, vegetable"
}

def fetch_restaurants(location, cuisine, meal, limit=20):
    """Fetch restaurants from Yelp API based on location, cuisine, and meal type."""
    meal_category = MEAL_TYPE_CATEGORIES.get(meal.lower(), "restaurants")  # Default to "restaurants"

    params = {
        "term": cuisine,
        "location": location,
        "categories": f"{meal_category}, {cuisine}",
        "sort_by": "best_match",
        "limit": limit
    }

    try:
        response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
        response.raise_for_status()  # Raises HTTP errors

        # Check for 'error' key in response JSON
        data = response.json()
        if "error" in data and data["error"]["code"] == "LOCATION_NOT_FOUND":
            return "LOCATION_NOT_FOUND"

        return data.get("businesses", [])

    except requests.exceptions.RequestException:
        return "LOCATION_NOT_FOUND"  # Ensures invalid locations do not crash the app


def parse_data(businesses):
    """Extract relevant data from API response."""
    data = []
    for b in businesses:
        data.append({
            "name": b["name"],
            "rating": b["rating"],
            "review_count": b["review_count"],
            "price": b.get("price", "N/A"),
            "latitude": b["coordinates"]["latitude"],
            "longitude": b["coordinates"]["longitude"],
            "categories": ", ".join([cat["title"] for cat in b["categories"]]),
            "address": ", ".join(b["location"]["display_address"])
        })
    return pd.DataFrame(data)

# Streamlit UI elements
st.image("header.png", use_container_width=True)
st.write("""
    Get personalized restaurant recommendations based on your group preferences!
""")

# Hardcoded valid cuisiines
VALID_CUISINES = [
    "american", "chinese", "japanese", "italian", "mexican", "thai", "indian",
    "french", "greek", "korean", "vietnamese", "spanish", "mediterranean",
    "brazilian", "caribbean", "ethiopian", "turkish", "middle eastern",
    "argentine", "algerian", "american BBQ", "angolan", "armenian", "asian fusion",
    "australian", "belgian", "bengali", "brazilian steakhouse", "british",
    "cambodian", "cajun", "caribbean", "chilean", "colombian", "congolese",
    "cuban", "cypriot", "czech", "danish", "dominican", "filipino", "georgian",
    "german", "haitian", "hawaiian", "hungarian", "icelandic", "irish", "israeli",
    "jamaican", "kenyan", "kosher", "laotian", "latin", "libyan", "lithuanian",
    "malaysian", "mongolian", "moroccan", "nepalese", "nigerian", "norwegian",
    "pakistani", "peruvian", "portuguese", "romanian", "russian", "south african",
    "sri lankan", "swedish", "swiss", "tanzanian", "thai street food", "tibetan",
    "venezuelan", "welsh", "zimbabwean"
]

# Step 1: Select meal type
meal = st.selectbox("🍛 What meal do you want to eat?", list(MEAL_TYPE_CATEGORIES.keys()))

# Step 2: Select number of people
num_people = st.number_input("👥 How many people are eating?", min_value=1, max_value=10, step=1, value=2)

# Step 3: Collect each person's cuisine preference using dropdown
cuisine_preferences = []

for i in range(num_people):
    preference = st.multiselect(f"🍣 Person {i+1}'s preferred cuisine", VALID_CUISINES)
    if preference:
        cuisine_preferences.extend(preference)

# from google.colab import drive
# drive.mount('/content/drive')

# # to access dataset in drive
# file_path = "/content/drive/My Drive/DSU Curriculum Project/uscities.csv"

# Load the 'uscities.csv' file
cities = pd.read_csv('uscities.csv')

# # Load the 'uscities.csv' file
# cities = pd.read_csv(file_path)

# Extract unique city and state combinations (city_ascii + state_id)
cities["city_state"] = cities["city_ascii"] + ", " + cities["state_id"]

# Drop duplicates to get unique city-state combinations and store it as a list
VALID_CITIES = cities["city_state"].drop_duplicates().tolist()

# Streamlit UI: City dropdown using VALID_CITIES list
city = st.selectbox("📍 Select your city", VALID_CITIES)

# **SUBMIT BUTTON**
if st.button("Submit"):
    # Check for invalid cuisines
    if not cuisine_preferences:
        st.error("⚠️ Please select at least one cuisine.")
    # Check for invalid location
    if not city:
        st.error("⚠️ Please enter a valid city and state.")
    else:
        st.success("✅ Valid cuisines detected. Fetching recommendations...")
        # Continue with the restaurant recommendation process...

        # Step 1: Convert cuisine preferences into numerical feature vectors using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(cuisine_preferences)

        # Step 2: Compute cosine similarity between preferences
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Step 3: Aggregate cuisine importance across all people
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
        cuisine_ranking = dict(zip(vectorizer.get_feature_names_out(), tfidf_scores))

        # Step 4: Rank cuisines based on their total weight
        sorted_cuisines = sorted(cuisine_ranking.items(), key=lambda x: x[1], reverse=True)
        df_ranked = pd.DataFrame(sorted_cuisines, columns=["Cuisine", "Weight"])

        # Extract top cuisines based on weight
        df_ranked["Recommendation Ratio"] = df_ranked["Weight"] / df_ranked["Weight"].sum()
        top_cuisines = df_ranked.sort_values("Recommendation Ratio", ascending=False).head(3).copy()

        # Determine recommendation allocation
        top_cuisine_name = top_cuisines.iloc[0]["Cuisine"]
        top_cuisine_ratio = top_cuisines.iloc[0]["Recommendation Ratio"]
        unique_weights = top_cuisines["Recommendation Ratio"].nunique()

        if top_cuisine_ratio > 0.67:
            cuisine_allocation = [top_cuisine_name] * 3
        elif top_cuisine_ratio > 0.34:
            second_cuisine_name = top_cuisines.iloc[1]["Cuisine"]
            cuisine_allocation = [top_cuisine_name] * 2 + [second_cuisine_name]
        elif unique_weights == 1:
            cuisine_allocation = list(top_cuisines["Cuisine"])[:3]
        else:
            cuisine_allocation = list(top_cuisines["Cuisine"])[:3]

        # **Function to get the best restaurant recommendations**
        def get_top_restaurant_recommendations(location, cuisine_allocation, meal, limit=20):
            all_restaurants = []
            for cuisine in cuisine_allocation:
                restaurants = fetch_restaurants(location, cuisine, meal, limit)
                df = parse_data(restaurants)
                if not df.empty:
                    all_restaurants.append(df)

            if all_restaurants:
                df_restaurants = pd.concat(all_restaurants, ignore_index=True)
            else:
                return []

            df_sorted = df_restaurants.sort_values(by=["rating", "review_count"], ascending=[False, False])
            selected_restaurants = set()
            recommendations = []

            for cuisine in cuisine_allocation:
                top_cuisine_restaurants = df_sorted[df_sorted["categories"].str.contains(cuisine, case=False, na=False)]
                for _, restaurant in top_cuisine_restaurants.iterrows():
                    if restaurant["name"] not in selected_restaurants:
                        recommendations.append(restaurant)
                        selected_restaurants.add(restaurant["name"])
                        break
                if len(recommendations) == 3:
                    break

            return recommendations

        # Get restaurant recommendations
        top_recommendations = get_top_restaurant_recommendations(city, cuisine_allocation, meal)

        # **Display recommendations**
        st.subheader("🏆 Top 3 Restaurant Recommendations:")
        if top_recommendations:
            for r in top_recommendations:
                st.write(f"**{r['name']}** ({r['categories']}) - ⭐ {r['rating']} ({r['review_count']} reviews) - 📍 {r['address']}")
        else:
            st.write("❌ No restaurants found for the selected preferences. Try different options!")