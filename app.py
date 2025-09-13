# app.py

import streamlit as st
from transformers import pipeline
import random

st.set_page_config(page_title="Emotion-Aware Diet Recommendation", layout="centered")
st.title("🍽 Emotion-Aware Diet Recommendation System")

# -------------------------------
# 1️⃣ Load Emotion Detection Model
# -------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

emotion_classifier = load_emotion_model()

def detect_emotion(user_text):
    result = emotion_classifier(user_text)[0]
    emotion = result['label']
    score = result['score']
    return emotion, score

# -------------------------------
# 2️⃣ Load LLM for generating descriptions
# -------------------------------
@st.cache_resource
def load_llm_model():
    # Using Flan-T5 Base for better instruction-following
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_model()

# -------------------------------
# 3️⃣ Pre-defined meal database
# -------------------------------
meals_db = {
    "Indian": {
        "Weight Loss": ["Steamed Vegetable Dhokla", "Moong Dal Salad", "Grilled Paneer Skewers"],
        "Muscle Gain": ["Chickpea Curry with Brown Rice", "Paneer Tikka Wrap", "Lentil Soup"],
        "Balanced Diet": ["Vegetable Khichdi", "Mixed Dal Soup", "Quinoa Pulao"]
    },
    "Chinese": {
        "Weight Loss": ["Steamed Veg Dumplings", "Tofu Stir Fry", "Cabbage Soup"],
        "Muscle Gain": ["Chicken & Broccoli Stir Fry", "Egg Fried Rice with Veggies", "Tofu & Vegetable Noodles"],
        "Balanced Diet": ["Vegetable Fried Rice", "Hot & Sour Soup", "Vegetable Spring Rolls"]
    },
    "Italian": {
        "Weight Loss": ["Grilled Vegetable Salad", "Zucchini Noodles with Tomato Sauce", "Minestrone Soup"],
        "Muscle Gain": ["Whole Wheat Pasta with Chicken", "Caprese Salad with Eggs", "Vegetable Risotto"],
        "Balanced Diet": ["Pasta Primavera", "Margherita Pizza with Whole Wheat Base", "Minestrone Soup"]
    },
    "Mexican": {
        "Weight Loss": ["Grilled Veg Tacos", "Black Bean Salad", "Chicken Lettuce Wraps"],
        "Muscle Gain": ["Chicken Burrito Bowl", "Quinoa & Bean Salad", "Grilled Fish Tacos"],
        "Balanced Diet": ["Vegetable Quesadilla", "Bean & Rice Bowl", "Chicken Tacos"]
    }
}

# -------------------------------
# 4️⃣ Pre-defined fallback templates per emotion
# -------------------------------
templates = {
    "sadness": [
        "Light and comforting, this meal can help lift your mood.",
        "Nutritious and easy to digest, perfect for feeling low.",
        "Warm and soothing, ideal to brighten a sad day."
    ],
    "joy": [
        "Bright and cheerful, matches your positive energy.",
        "Full of flavor and nutrients to keep your spirits high.",
        "A fun and vibrant meal to celebrate your happiness."
    ],
    "anger": [
        "Calming and balanced, helps soothe your mind.",
        "Rich in nutrients that can help reduce stress.",
        "Comforting and filling, perfect to regain emotional balance."
    ],
    "fear": [
        "Grounding and satisfying, helps reduce anxiety.",
        "Nutritious and steadying, great for feeling tense.",
        "Light yet nourishing, to calm nerves and boost energy."
    ],
    "surprise": [
        "Exciting and flavorful, perfect for unexpected moments.",
        "A lively meal to match your energetic mood.",
        "Bright and tasty, keeping the surprise fun and enjoyable."
    ],
    "disgust": [
        "Fresh and clean flavors, easy to enjoy.",
        "Light and palatable, to refresh your senses.",
        "Nutritious and mild, ideal for recovering appetite."
    ],
    "neutral": [
        "A balanced and satisfying meal for steady energy.",
        "Simple and healthy, keeping your mood calm and stable.",
        "Nutritious and easy to enjoy at any time."
    ]
}

# -------------------------------
# 5️⃣ Generate meals + emotion-aware descriptions
# -------------------------------
@st.cache_data
def suggest_meals_hybrid(emotion, diet_goal, cuisine, num_meals=3):
    options = meals_db.get(cuisine, {}).get(diet_goal, ["Mixed Salad"])
    selected_meals = random.sample(options, min(num_meals, len(options)))
    
    meals_with_descriptions = []
    for meal in selected_meals:
        prompt = f"""
Meal: {meal}
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
            
            # Fallback if the output is too short, empty, or repeats instructions
            if len(description) < 10 or any(word in description.lower() for word in ["meal", "focus on", "avoid repetition"]):
                description = random.choice(templates.get(emotion.lower(), ["Delicious and healthy meal."]))
        except:
            description = random.choice(templates.get(emotion.lower(), ["Delicious and healthy meal."]))
        
        meals_with_descriptions.append((meal, description))
        
    return meals_with_descriptions

# -------------------------------
# 6️⃣ Streamlit UI
# -------------------------------
user_input = st.text_area("How are you feeling today?", height=100)
diet_goal = st.selectbox("Select your dietary goal:", ["Weight Loss", "Muscle Gain", "Balanced Diet"])
cuisine = st.selectbox("Preferred cuisine:", ["Indian", "Italian", "Chinese", "Mexican"])

if st.button("Get Meal Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter how you are feeling to get recommendations.")
    else:
        with st.spinner("Detecting emotion and generating meals..."):
            emotion, score = detect_emotion(user_input)
            meal_suggestions = suggest_meals_hybrid(emotion, diet_goal, cuisine, num_meals=3)

        st.success(f"**Detected Emotion:** {emotion} ({score:.2f})")
        st.subheader("🍲 Recommended Meals:")
        for idx, (meal, description) in enumerate(meal_suggestions, 1):
            st.markdown(f"**{idx}. Meal:** {meal}")
            st.markdown(f"**Description:** {description}\n")
