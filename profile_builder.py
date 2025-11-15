import pandas as pd
from generators.litellm import LiteLLMGenerator
import json 
from tqdm.auto import tqdm

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
7.  **DISTINGUISH PREFERENCES from REQUESTS:** You *must* understand the difference between a permanent preference and a one-time request.
    - A **PERMANENT PREFERENCE** is a global rule.
      - *Examples:* "I am allergic to shellfish.", "I *always* prefer aisle seats.", "Ensure *all* my hotel rooms have hypoallergenic products."
      - *Action:* Add these to the `preferences` object.   
    - A **ONE-TIME REQUEST** is part of a specific trip.
      - *Examples:* "I need a piano for my Vienna room.", "Get me a kitchenette for my Paris trip.", "I need a private guide for Rodeo Drive."
      - *Action:* Do **NOT** add these to the `preferences` object. Instead, find the matching event in `trips_and_events` and add this to its `"details": [...]` list.
"""

BATCH_SIZE = 50
mistral = LiteLLMGenerator(model_name="ollama/mistral")

# Load data and sort according to timestamp 
json_data = json.load(open(f"data/response_1762800357568.json"))
data = pd.DataFrame(json_data["items"])
data["timestamp"] = pd.to_datetime(data["timestamp"])
data = data.sort_values(by="timestamp", ascending=True)

test_profile = "Vikram Desai"
vikram_data = data[data["user_name"] == test_profile]
data = vikram_data.copy()
vikram_data = vikram_data["timestamp"].astype(str) + ": " + vikram_data["message"]
vikram_data = list(vikram_data)

for user in tqdm(data["user_name"].unique()):
    user_data = data[data["user_name"] == user][["timestamp", "message"]]
    user_data = user_data["timestamp"].astype(str) + ": " + user_data["message"]
    user_data = list(user_data)

    current_profile_str = "{}"  # Start with empty profile for simplicity
    for i in range(0, len(user_data), BATCH_SIZE):
        batch = user_data[i:min(i+BATCH_SIZE, len(user_data))]
        batch = "\n".join(batch)
        prompt = f"""
        Name of the User: {test_profile}

        Current Profile:
        {current_profile_str}
        
        Batch of New Messages:
        {batch}
        
        Based on the above, update the profile according to the schema and rules provided.
        """
        
        response = mistral.generate(
            PROFILE_SYSTEM_PROMPT + prompt,
        )
        
        # Update current profile for next batch
        current_profile_str = response.strip()
        # Save intermediate profiles
        with open(f"profiles/{user}_mistral_b{BATCH_SIZE}_v{i//BATCH_SIZE + 1}.txt", "w") as f:
            f.write(current_profile_str)

    # You can also save the final profile after all batches
    with open(f"profiles/{user}_mistral_b{BATCH_SIZE}_latest.txt", "w") as f:
        f.write(current_profile_str)