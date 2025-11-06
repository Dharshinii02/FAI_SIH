# FAI_SIH
### Intelligent Enterprise Assistant 
An AI-powered virtual assistant designed to help the public sector work smarter. It automates tasks like document search, policy answering, leave requests, and helpdesk support — all through natural conversation.
### Overview
The Intelligent Enterprise Assistant acts as a **smart FAQ chatbot** for organizational use.  
It uses predefined knowledge to respond to queries and can be easily extended to support more workflows.

This prototype demonstrates how AI-powered chat support can:
1. Answer repeated employee/citizen queries instantly
2. Reduce helpdesk workload
3. Improve access to organizational knowledge
4. Serve as a foundation for integrating automation and data lookup
### Problem Statement
Public sector organizations handle large volumes of repetitive queries related to rules, HR policies, procedures, and internal services.  
Traditional systems require manual lookup, causing delays and inefficiency.
### Features
 • Conversational Chat Interface
 • Predefined knowledge responses
 • Web-based UI 
 • Accessible using any browser 
 • Easy to Customize
 • Responds to common policy and helpdesk queries
 • Users can ask questions naturally 
 ### Project Structure
 ```
intelligent-enterprise-assistant/
│
├── app.py
│
├── templates/
│ └── index.html 
│
└── static/
└── (CSS / JS files)

 ```
### Implementation Code:
#### app.py
```
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Your Knowledge Base (The same as before)
responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help?",
    "how are you": "I'm working perfectly! 😊",
    "bye": "Goodbye! Have a great day!",
    "what is an intelligent enterprise": "An Intelligent Enterprise uses AI, automation, and data to improve efficiency and decision making.",
    "what is your purpose": "My purpose is to assist employees and citizens by providing quick and accurate information.",
    "how do you help employees": "I help employees by answering questions instantly and reducing manual work.",
    "how to apply for leave": "You can apply leave using the HR Portal under Employee Self Service.",
    "what are office working hours": "Office hours are Monday to Friday, 9 AM to 5 PM.",
    "how to register a complaint": "You can register a complaint through the Citizen Grievance Portal.",
    "how to reset my password": "You can reset your password through the IT Helpdesk Support System.",
    "where to get service forms": "Service forms are available on the official e-Governance portal.",
    "what documents are needed for id card": "You need Aadhaar Card, Employee ID, and a passport-size photograph."
}

# 2. Setup the "AI" (TF-IDF Vectorizer)
# We get all the questions (keys) from our responses
known_questions = list(responses.keys())

# Create the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# "Train" the vectorizer on our known questions
# This creates a mathematical representation of our questions
question_vectors = vectorizer.fit_transform(known_questions)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_msg = request.form["msg"].lower()
    
    # 3. Find the Best Match
    # Convert the user's message into a vector
    user_vector = vectorizer.transform([user_msg])
    
    # Calculate the similarity between the user's message and all known questions
    similarities = cosine_similarity(user_vector, question_vectors)
    
    # Find the index of the most similar question
    closest_match_index = similarities.argmax()
    
    # Get the similarity score (a value between 0 and 1)
    max_similarity = similarities[0, closest_match_index]
    
    # 4. Get the Reply
    # We set a "confidence threshold" to avoid bad matches.
    # If the best match is less than 30% similar, we say we don't understand.
    if max_similarity > 0.3:
        # Get the best matching question
        best_question = known_questions[closest_match_index]
        # Get the answer for that question
        reply = responses[best_question]
    else:
        reply = "Sorry, I'm not sure how to help with that. Can you try rephrasing?"
        
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)

```
#### index.html
```
<!DOCTYPE html>
<html>
<head>
    <title>Simple Chatbot</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>

<h2>💬 Simple AI Chatbot</h2>

<div id="chatbox"></div>

<div class="input-area">
    <input type="text" id="userInput" placeholder="Type your message...">
    <button onclick="sendMessage()">Send</button>
</div>

<script src="/static/script.js"></script>
</body>
</html>

```
#### style.css
```
body {
  background: #f2f2f2;
  text-align: center;
  font-family: Arial, sans-serif;
}

h2 {
  color: #333;
}

#chatbox {
  width: 60%;
  height: 350px;
  background: white;
  border-radius: 6px;
  padding: 10px;
  margin: auto;
  margin-bottom: 10px;
  overflow-y: auto;
  border: 1px solid #ccc;
}

.input-area {
  display: flex;
  justify-content: center;
  gap: 10px;
}

#userInput {
  width: 50%;
  padding: 10px;
  border-radius: 4px;
  border: 1px solid #555;
}

button {
  padding: 10px 20px;
  border: none;
  background: #007bff;
  color: white;
  cursor: pointer;
  border-radius: 4px;
}

button:hover {
  background: #0056b3;
}
```
#### script.js
```
function sendMessage() {
  let msg = document.getElementById("userInput").value;
  if (msg.trim() === "") return;

  document.getElementById("chatbox").innerHTML += `<p><b>You:</b> ${msg}</p>`;

  fetch("/get", {
    method: "POST",
    headers: {"Content-Type": "application/x-www-form-urlencoded"},
    body: `msg=${msg}`
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById("chatbox").innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
  });

  document.getElementById("userInput").value = "";
}
```
### How to Run Everything
```
 Step 1: Open Terminal in your project folder
 Step 2: Install Dependencies
 pip install flask
 Step 3: Start the Flask Server
 python app.py
 You will see
 Running on http://127.0.0.1:5000/
 Step 4: Open in Browser
 http://127.0.0.1:5000/
 Step 5: Open frontend

 Open frontend/index.html in your browser.
 Ask something like:
 Step 4: Open frontend
  what is an intelligent enterprise?

✅ It’ll reply using your sample policy data!
```
### Future Enhancements

• Add AI-based contextual search using embeddings
• Voice-based interaction (speech-to-text)
• Hindi / Telugu / Tamil language support
• Integration with actual HRMS system
• Admin dashboard for analytics
### OUTPUT:
<img width="1920" height="1080" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/1f9d9c40-b798-4317-8c55-d7577a94e683" />


### RESULT:
Thus the Chatbot is executed Successfully.

