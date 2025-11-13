import pandas as pd
from generators.litellm import LiteLLMGenerator
import json 

# Define all the constants and system prompt
PROFILE_SYSTEM_PROMPT = """
You are a stateful, expert AI agent. Your job is to build a clean, structured JSON profile for a user.
You will be given the user's `current_profile` (as JSON) and a `batch_of_new_messages` (as a JSON list).
Your task is to read the `current_profile`, then process *every* message in the batch *in chronological order* to create the new, updated profile.

**YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT AND NOTHING ELSE.**

**PROFILE SCHEMA:**
{
  "user_name": "string",
  "user_id": "string",
  "contact_info": {
    "phone": "string",
    "email": "string",
    "address": "string",
    "passport": "string"
  },
  "preferences": {
    "seat": "string (e.g., 'aisle', 'window')",
    "hotel_amenities": ["string"],
    "allergies": ["string"],
    "other": ["string"]
  },
  "trips_and_events": [
    {
      "item": "string (e.g., 'Trip to Paris', 'U2 Concert')",
      "status": "string (e.g., 'Requested', 'Completed', 'Canceled', 'Changed')",
      "request_date": "string (YYYY-MM-DD)",
      "event_date": "string (YYYY-MM-DD, if known)",
      "user_aliases": ["string (e.g., 'concert package')"],
      "feedback": "string (e.g., 'Positive', 'Negative')"
    }
  ]
}

**YOUR REASONING RULES:**
1.  **JSON ONLY:** Your entire response must be *only* the new JSON profile. Do NOT add "Here is the updated profile..."
2.  **FOLLOW SCHEMA:** Your output *must* follow the schema. If a section is empty, use an empty object or list (e.g., `"contact_info": {}`).
3.  **CHRONOLOGICAL:** Process the `batch_of_new_messages` in the order they are given.
4.  **CONTRADICTIONS:** The newest message *always* wins. If `current_profile` says `"seat": "window"` but a new message says "I prefer aisle", the new profile *must* have `"seat": "aisle"`.
5.  **RESOLVE DATES:** Resolve relative dates ('next month', 'this Friday') using the message's timestamp as the "current date".
6.  **CONNECT & REASON:** This is the most important rule. You must connect feedback and aliases.
    - **Example 1:** If a "Pending" event is "Son's Graduation" and a new message says "the concert package was great", you must reason they are the same. Update the event to `status: "Completed"`, `feedback: "Positive"`, and add `"user_aliases": ["concert package"]`.
    - **Example 2:** If a "Pending" event is "Ballet Tickets" and a new message says "thank you for handling that ticket issue", update the event `status` to "Completed".
"""

BATCH_SIZE = 100
path = "C:\\MY FILES\\Peeyush-Personal\\Coding\\Aurora-Technical-Assessment-NLP-QA-"
mistral = LiteLLMGenerator(model_name="ollama/mistral")
deepseek = LiteLLMGenerator(model_name="ollama/deepSeek-r1")

# Load data and sort according to timestamp 
json_data = json.load(open(f"{path}\\data\\response_1762800357568.json"))
data = pd.DataFrame(json_data["items"])
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values(by="timestamp", ascending=True)

test_profile = "Vikram Desai"
vikram_data = data[data["user_name"] == test_profile][["timestamp", "message"]]
vikram_data = vikram_data["timestamp"].astype(str) + ": " + vikram_data["message"]
vikram_data = list(vikram_data)

current_profile_str = "{}"  # Start with empty profile for simplicity
for i in range(0, len(vikram_data), BATCH_SIZE):
    batch = vikram_data[i:min(i+BATCH_SIZE, len(vikram_data))]
    batch = "\n".join(batch)
    prompt = f"""
    Name of the User: {test_profile}

    Current Profile:
    {current_profile_str}
    
    Batch of New Messages:
    {batch}
    
    Based on the above, update the profile according to the schema and rules provided.
    """
    
    response = deepseek.generate(
        PROFILE_SYSTEM_PROMPT + prompt,
    )
    
    # Update current profile for next batch
    current_profile_str = response.strip()
    print(f"Updated Profile after processing batch {i//BATCH_SIZE + 1}:\n{current_profile_str}\n")

    # Save intermediate profiles
    with open(f"{path}\\profiles\\vikram_deepseek_v{i//BATCH_SIZE + 1}.txt", "w") as f:
        f.write(current_profile_str)
        current_profile_str
    