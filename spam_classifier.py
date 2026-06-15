# ============================================
#   SPAM EMAIL CLASSIFIER
#   Techniques: Naive Bayes + TF-IDF
#   Libraries: scikit-learn, pandas
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ── STEP 1: Sample Dataset ──────────────────
# (In real project, load a CSV file instead)
data = {
    "email": [
        "Congratulations! You won a free iPhone. Click here to claim now!",
        "URGENT: Your account has been suspended. Verify immediately.",
        "Get rich quick! Earn $5000 a day working from home.",
        "Free Viagra! Limited time offer. Buy now!",
        "Win a lottery prize of $1,000,000. You have been selected!",
        "Dear friend, I need your help transferring $10 million.",
        "Click here for FREE weight loss pills. No exercise needed!",
        "You are a WINNER! Claim your prize before it expires.",

        "Hey, are we still on for lunch tomorrow?",
        "Please find the attached project report for your review.",
        "Meeting rescheduled to 3 PM. Let me know if that works.",
        "Your order has been shipped. Track it here.",
        "Reminder: Team standup at 10 AM tomorrow.",
        "Can you send me the notes from today's class?",
        "Happy birthday! Hope you have a wonderful day.",
        "The assignment submission deadline is Friday 11:59 PM.",
        "Let's catch up this weekend. It's been a while!",
        "Your library book is due for return by Monday.",
    ],
    "label": [
        "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam",
        "ham",  "ham",  "ham",  "ham",
        "ham",  "ham",  "ham",  "ham",
        "ham",  "ham",
    ]
}

df = pd.DataFrame(data)
print("── Dataset ──────────────────────────────")
print(f"Total emails : {len(df)}")
print(f"Spam count   : {(df['label'] == 'spam').sum()}")
print(f"Ham count    : {(df['label'] == 'ham').sum()}")
print()

# ── STEP 2: Split into Train & Test ────────
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"],
    test_size=0.25,
    random_state=42
)

# ── STEP 3: TF-IDF Vectorization ───────────
# Converts text emails into numbers the model can understand
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── STEP 4: Train Naive Bayes Model ────────
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ── STEP 5: Evaluate ───────────────────────
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("── Model Results ────────────────────────")
print(f"Accuracy : {accuracy * 100:.1f}%")
print()
print(classification_report(y_test, y_pred))

# ── STEP 6: Test on New Emails ─────────────
print("── Live Prediction ──────────────────────")
test_emails = [
    "You have won a free trip to Maldives! Claim now!",
    "Can you please review the pull request I submitted?",
    "LIMITED OFFER: Buy 1 get 10 FREE! Click immediately!",
    "Reminder: your dentist appointment is at 4 PM today.",
]

for email in test_emails:
    vec   = vectorizer.transform([email])
    pred  = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf  = max(proba) * 100
    tag   = "🚨 SPAM" if pred == "spam" else "✅ HAM "
    print(f"{tag}  ({conf:.0f}% confident)")
    print(f"      \"{email[:60]}...\"" if len(email) > 60 else f"      \"{email}\"")
    print()
