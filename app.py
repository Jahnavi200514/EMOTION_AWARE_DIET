import streamlit as st
import requests
import random
from transformers import pipeline

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Emotion-Aware Diet Chatbot", layout="centered")
st.title("🍽 Emotion-Aware Diet Chatbot")

SPOONACULAR_API_KEY = "525ebaaa46034202a3426620b2526323"  # 🔑 Replace with your key

# ---------------------- EMOTION DETECTION ----------------------
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

emotion_classifier = load_emotion_model()

def detect_emotion(user_text):
    result = emotion_classifier(user_text)[0]
    return result['label'], result['score']

# ---------------------- LLM MODEL ----------------------
@st.cache_resource
def load_llm_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_model()

# ---------------------- FALLBACK DESCRIPTIONS ----------------------
templates = {
    "sadness": ["Comforting and warm, this dish can lift your spirits."],
    "joy": ["Bright and cheerful, just like your mood!"],
    "anger": ["Calming and soothing, helps relax your mind."],
    "fear": ["Grounding and nourishing, keeps you steady."],
    "surprise": ["Exciting and flavorful, a joyful twist!"],
    "disgust": ["Fresh and clean flavors, easy to enjoy."],
    "neutral": ["Balanced and wholesome, great for any day."]
}

# ---------------------- SPOONACULAR API ----------------------
def fetch_meals_from_spoonacular(cuisine, num_meals=3):
    url = f"https://api.spoonacular.com/recipes/complexSearch"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "cuisine": cuisine,
        "number": num_meals,
        "addRecipeNutrition": True
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        return res.json().get("results", [])
    return []

def fetch_instructions(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/analyzedInstructions"
    res = requests.get(url, params={"apiKey": SPOONACULAR_API_KEY})
    if res.status_code == 200 and res.json():
        return [step["step"] for step in res.json()[0]["steps"]]
    return ["Instructions not available."]

# ---------------------- HYBRID MEAL SUGGESTION ----------------------
@st.cache_data
def suggest_meals_hybrid(emotion, cuisine, num_meals=3):
    meals = fetch_meals_from_spoonacular(cuisine, num_meals)
    meals_with_descriptions = []

    for meal in meals:
        meal_name = meal["title"]
        try:
            prompt = f"Meal: {meal_name}\nOne short sentence why this is good for someone feeling {emotion}."
            description = llm_pipeline(prompt, max_length=50, do_sample=True, temperature=0.8)[0]['generated_text']
        except:
            description = random.choice(templates.get(emotion.lower(), ["A delicious and healthy choice!"]))

        meal["description"] = description
        meals_with_descriptions.append(meal)

    return meals_with_descriptions

# ---------------------- SESSION STATE ----------------------
if "meal_suggestions" not in st.session_state:
    st.session_state.meal_suggestions = []
if "selected_meal" not in st.session_state:
    st.session_state.selected_meal = None

# ---------------------- FLOW ----------------------
if not st.session_state.meal_suggestions:
    # Step 1: Emotion Input
    user_input = st.text_area("💬 How are you feeling today?", height=100)
    cuisine = st.selectbox("🌍 Preferred cuisine:", ["Indian", "Italian", "Chinese", "Mexican", "American"])

    if st.button("Get Meal Recommendations"):
        if user_input.strip():
            with st.spinner("Detecting emotion and generating meals..."):
                emotion, score = detect_emotion(user_input)
                st.session_state.meal_suggestions = suggest_meals_hybrid(emotion, cuisine, 3)
                st.session_state.emotion = emotion
                st.session_state.score = score
        else:
            st.warning("Please enter how you are feeling.")

else:
    if st.session_state.selected_meal is None:
        # Step 2: Show Meal List
        st.success(f"**Detected Emotion:** {st.session_state.emotion} ({st.session_state.score:.2f})")
        st.subheader("🍲 Recommended Meals for You:")

        for meal in st.session_state.meal_suggestions:
            with st.container():
                st.image(meal["image"], width=200)
                st.markdown(f"### {meal['title']}")
                st.write(f"👉 {meal['description']}")
                if st.button(f"View Details of {meal['title']}", key=meal["id"]):
                    st.session_state.selected_meal = meal
                    st.rerun()

    else:
        # Step 3: Show Dish Details
        meal = st.session_state.selected_meal
        st.header(f"🍴 {meal['title']}")
        st.image(meal["image"], width=300)

        # Expandable Sections
        with st.expander("📝 Ingredients"):
            if "nutrition" in meal and "ingredients" in meal["nutrition"]:
                for ing in meal["nutrition"]["ingredients"]:
                    st.write(f"- {ing['amount']} {ing['unit']} {ing['name']}")

        with st.expander("📖 Instructions"):
            steps = fetch_instructions(meal["id"])
            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")

        with st.expander("🔥 Calories & Nutrition"):
            if "nutrition" in meal and "nutrients" in meal["nutrition"]:
                for n in meal["nutrition"]["nutrients"][:8]:
                    st.write(f"- {n['name']}: {n['amount']} {n['unit']}")

        # Back button
        if st.button("⬅️ Back to Recommendations"):
            st.session_state.selected_meal = None
            st.rerun()
