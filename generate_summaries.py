"""
Step 2: Generate experiment summaries from each paper.
Uses one strong model (Claude Sonnet 4.5) to generate summaries.
These summaries are used as input for Task 1.
"""

import os
import json
import time
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Use a strong model for summary generation
SUMMARY_MODEL = "anthropic/claude-sonnet-4.5"

SYSTEM_PROMPT = (
    "You are a helpful assistant for summarizing key details of experiments "
    "and methodologies from scientific papers."
)

USER_INSTRUCTION = (
    "Summarize the following research paper, focusing ONLY on this question: "
    "Carefully analyze ONLY the experiments performed or methods used. "
    "Do NOT include results, abstract, introduction, or discussion. "
    "Output MUST be valid JSON of the form: "
    '{"title": "<paper title>", "experiments_summary": "<concise summary>"} '
    "Do NOT wrap the JSON in markdown code fences. "
    "Paper text: "
)

def call_openrouter(model, system_prompt, user_message, max_tokens=2000, temperature=0.7):
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    for attempt in range(5):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 60)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error {resp.status_code}: {resp.text[:200]}")
                time.sleep(5)
        except Exception as e:
            print(f"    Exception: {e}")
            time.sleep(5)
    
    return None

def truncate_text(text, max_chars=100000):
    """Truncate text to fit within context limits."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text

def main():
    # Load paper metadata
    with open("data/papers_metadata.json") as f:
        papers = json.load(f)
    
    os.makedirs("data/summaries", exist_ok=True)
    
    # Check which summaries already exist
    existing = set()
    for fname in os.listdir("data/summaries"):
        if fname.endswith(".json"):
            existing.add(fname.replace(".json", ""))
    
    for i, paper in enumerate(papers):
        paper_id = paper["paper_id"]
        
        if paper_id in existing:
            print(f"[{i+1}/50] {paper_id} - already done")
            continue
        
        print(f"[{i+1}/50] Generating summary for {paper_id}...")
        
        # Load paper text
        with open(paper["text_path"]) as f:
            text = f.read()
        
        text = truncate_text(text)
        user_message = USER_INSTRUCTION + text
        
        response = call_openrouter(
            SUMMARY_MODEL, SYSTEM_PROMPT, user_message,
            max_tokens=2000, temperature=0.3
        )
        
        if response:
            # Save raw response
            summary_data = {
                "paper_id": paper_id,
                "model": SUMMARY_MODEL,
                "response": response,
            }
            
            with open(f"data/summaries/{paper_id}.json", 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"  Done ({len(response)} chars)")
        else:
            print(f"  FAILED")
        
        time.sleep(1)  # Rate limiting
    
    # Verify all summaries
    count = len([f for f in os.listdir("data/summaries") if f.endswith(".json")])
    print(f"\nTotal summaries: {count}/50")

if __name__ == "__main__":
    main()
