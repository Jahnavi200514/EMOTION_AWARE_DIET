import streamlit as st
from transformers import pipeline
import random
import requests

st.set_page_config(page_title="Emotion-Aware Diet Recommendation", layout="centered")
st.title("🍽 Emotion-Aware Diet Recommendation System")


# ---------------------- MODELS ----------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

emotion_classifier = load_emotion_model()


def detect_emotion(user_text):
    result = emotion_classifier(user_text)[0]
    emotion = result['label']
    score = result['score']
    return emotion, score


@st.cache_resource
def load_llm_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_model()


# ---------------------- FALLBACK TEMPLATES ----------------------
templates = {
    "sadness": ["Light and comforting, this meal can help lift your mood."],
    "joy": ["Bright and cheerful, matches your positive energy."],
    "anger": ["Calming and balanced, helps soothe your mind."],
    "fear": ["Grounding and satisfying, helps reduce anxiety."],
    "surprise": ["Exciting and flavorful, perfect for unexpected moments."],
    "disgust": ["Fresh and clean flavors, easy to enjoy."],
    "neutral": ["A balanced and satisfying meal for steady energy."]
}


# ---------------------- THEMEALDB API ----------------------
def fetch_meals_from_api(cuisine, num_meals=3):
    """Fetch meals dynamically from TheMealDB API by cuisine (area)."""
    url = f"https://www.themealdb.com/api/json/v1/1/filter.php?a={cuisine}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        meals = data.get("meals", [])
        if meals:
            return random.sample(meals, min(num_meals, len(meals)))
    return []


# ---------------------- HYBRID MEAL SUGGESTION ----------------------
@st.cache_data
def suggest_meals_hybrid(emotion, cuisine, num_meals=3):
    meals = fetch_meals_from_api(cuisine, num_meals)
    meals_with_descriptions = []

    for meal in meals:
        meal_name = meal["strMeal"]
        meal_img = meal["strMealThumb"]

        prompt = f"""
Meal: {meal_name}
Write **one short, engaging sentence** explaining why this meal is good for someone feeling {emotion}.
Focus on emotional benefit and nutritional value. Avoid repetition.
"""
        try:
            description = llm_pipeline(
                prompt,
                max_length=50,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                eos_token_id=llm_pipeline.tokenizer.eos_token_id
            )[0]['generated_text'].strip()

            if len(description) < 10 or any(word in description.lower() for word in ["meal", "focus on", "avoid repetition"]):
                description = random.choice(templates.get(emotion.lower(), ["Delicious and healthy meal."]))
        except:
            description = random.choice(templates.get(emotion.lower(), ["Delicious and healthy meal."]))

        meals_with_descriptions.append((meal_name, meal_img, description))

    return meals_with_descriptions


# ---------------------- STREAMLIT UI ----------------------
user_input = st.text_area("How are you feeling today?", height=100)
cuisine = st.selectbox("Preferred cuisine:", ["Indian", "Italian", "Chinese", "Mexican"])

if st.button("Get Meal Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter how you are feeling to get recommendations.")
    else:
        with st.spinner("Detecting emotion and generating meals..."):
            emotion, score = detect_emotion(user_input)
            meal_suggestions = suggest_meals_hybrid(emotion, cuisine, num_meals=3)

        st.success(f"**Detected Emotion:** {emotion} ({score:.2f})")
        st.subheader("🍲 Recommended Meals:")

        for idx, (meal, img, description) in enumerate(meal_suggestions, 1):
            st.markdown(f"**{idx}. {meal}**")
            st.image(img, width=250)
            st.markdown(f"👉 {description}\n")
