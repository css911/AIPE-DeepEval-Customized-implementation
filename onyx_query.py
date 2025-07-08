import requests
import json
import os
import sys
from dotenv import load_dotenv

# we will eventually need the ssl certificate from the assistant website
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
API_BASE_URL = os.getenv('API_BASE_URL', 'https://assistant.ai.it.ufl.edu')
API_KEY = os.getenv('API_KEY', 'on_UL9S2jsUjnc6RQOxZJbavVytlHhbE-M7kiLbWrXxcwq9heIOufI-NknljTgqzXyk5EqrpAWZhpnLlY1i-aR0myN_3Nu4Y4AezM3OZ7GuAFivDrsgwj7bUR55N1xqsvAKwpgZjPLRQR8lhnX8HJM1fJ8e0y9kl_H53N_64Gxfg_8yQ3SdlfNB3_h9jkSD7dUiri4KmHQFWlM_yTE_2wqIBIN6e9uRaR4HvF10sSdCXaxYuCni0T7H38QfjWlShLaE')

if not API_KEY:
    print("Error: API_KEY environment variable is required")
    sys.exit(1)

def send_message(persona_id, prompt_id, message, chat_session_id=None, parent_message_id=None):
    url = f"{API_BASE_URL}/api/chat/send-message"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "alternate_assistant_id": persona_id,
        "prompt_id": prompt_id,
        "message": message,
        "chat_session_id": chat_session_id or "create_new_session_id_or_use_existing",
        "parent_message_id": parent_message_id,
        "file_descriptors": [],
        "search_doc_ids": None,
        "regenerate": False,
        "retrieval_options": {
            "run_search": "auto",
            "real_time": True,
            "filters": {
                "source_type": None,
                "document_set": None,
                "time_cutoff": None,
                "tags": []
            }
        },
        "llm_override": {
            "model_provider": "LiteLLM",
            "model_version": "llama-3.1-70b-instruct"
        }
    }
    
    with requests.post(url, headers=headers, json=payload, stream=True, verify=False) as response:
        response.raise_for_status()
        
        answer_pieces = []
        final_context_docs = []
        all_content = []
        
        for line in response.iter_lines():
            if line:
                if line.startswith(b'data: '):
                    line = line[6:]
                
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    if "answer_piece" in data:
                        answer_pieces.append(data["answer_piece"])
                    
                    if "message" in data:
                        complete_message = data["message"]
                    
                    if "final_context_docs" in data:
                        final_context_docs = data["final_context_docs"]
                        for doc in final_context_docs:
                            if "content" in doc and doc["content"].strip():
                                all_content.append(doc["content"])
                    
                    if "tool_result" in data:
                        for doc in data.get("tool_result", []):
                            if "content" in doc and doc["content"].strip():
                                all_content.append(doc["content"])
                                
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {line}")
                    continue
    
    return {
        "answer": "".join(answer_pieces),
        "context": all_content 
    }

def create_chat_session():
    """Create a new chat session and return the ID"""
    url = f"{API_BASE_URL}/api/chat/create-chat-session"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Origin": f"{API_BASE_URL}"
    }
    payload = {
        "title": "API Chat Session"
    }
    
    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.json().get("chat_session_id")

if __name__ == "__main__":
    chat_session_id = create_chat_session()
    print(f"Created chat session: {chat_session_id}")
    
    result = send_message(55, 61, "What are the office hours for the EHS division?", chat_session_id)
    
    print("Complete answer built from pieces:", result["answer"])
    print("Complete context: ", result["context"])