import os
from typing import List
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found! Set it in a .env file or environment variable.")
    st.stop()

# Define the LLM with API Key
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# Define the Itinerary Prompt
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
    ("human", "Create an itinerary for my day trip.")
])

# Function to generate itinerary
def create_itinerary(city: str, interests: str) -> str:
    response = llm.invoke(itinerary_prompt.format_messages(city=city, interests=interests))
    return response.content

# Streamlit UI
st.title("🌍 AI Travel Itinerary Planner")
st.write("Enter a city and your interests to generate a personalized day trip itinerary.")

# User input fields
city = st.text_input("Enter the city for your day trip")
interests = st.text_area("Enter your interests (comma-separated)")

# Generate itinerary button
if st.button("Generate Itinerary"):
    if city and interests:
        itinerary = create_itinerary(city, interests)
        st.subheader("Generated Itinerary:")
        st.write(itinerary)
    else:
        st.warning("Please enter both city and interests.")