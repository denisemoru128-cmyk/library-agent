"""Training data and fixed answers for the chatbot.

INTENT_EXAMPLES is the labelled dataset the chatbot LEARNS from: example
sentences for each thing a student might want ("intent"). A machine-learning
classifier (TF-IDF + logistic regression, see chatbot.py) is trained on it, so
the bot recognises new phrasings it has never seen, not only these exact ones.

FAQ holds the library rules. Each entry has several ways of asking and one
answer built from config.py, so the answers can never disagree with the real rules.

None of the sentences used to TEST the chatbot (see evaluation.py) appear here.
"""
from .config import FINE_PER_DAY, LOAN_DAYS, MAX_ACTIVE_LOANS, MAX_RENEWALS, REMINDER_DAYS_BEFORE_DUE

INTENT_EXAMPLES = {
    "availability": [
        "is this book available", "do you have this title", "check if the book is in stock",
        "can i borrow this book", "how many copies are available",
        "does the library have this book", "is there a copy of this title",
        "check availability for this book", "can i check out this title",
        "is it in the library right now", "are there any copies left", "is this title on the shelf",
        "can i get this book today", "is there a free copy of this book", "do you still have this one",
        "is it available", "is that book available", "can i borrow it",
        "do you stock this book", "is there any copy of it left", "can i take this book out today",
    ],
    "due_soon": [
        "what books are due this week", "which loans are due soon", "what do i need to return",
        "show books due in the next seven days", "what is due this week",
        "list my upcoming due dates", "which books should i return this week",
        "show loans due soon", "what is due next week", "when should i return my books",
        "what is due in the next few days", "do i have anything due soon",
        "which of my books are due tomorrow", "remind me what is due", "what should i bring back soon",
        "any loans expiring this week",
        "which books are due before the weekend", "show me what expires soon", "loans ending in the next few days",
    ],
    "overdue": [
        "what books are overdue", "show overdue loans", "which books have not been returned on time",
        "list overdue books", "do i have any overdue books", "show me late returns",
        "which loans are past their due date", "find overdue items", "my overdue loans",
        "who has overdue books", "which books are late", "show late books", "any books past due",
        "list books that are past their return date", "show me books that have been kept too long",
        "am i late with any books",
        "who has books past due", "show books not returned by their due date", "list late loans",
    ],
    "recommendations": [
        "recommend a book for me", "what should i read next", "suggest books based on my history",
        "find something i might like", "recommend books in my interests",
        "what books are popular with readers like me", "give me reading suggestions",
        "show my personalized recommendations", "books similar to what i borrowed",
        "what would you recommend", "suggest something to read", "any book recommendations for me",
        "i want something new to read", "what other books would i like", "recommend me a good book",
        "what do you suggest i borrow",
        "give me ideas on what to borrow", "what should i pick up next", "suggest some reading",
    ],
    "search": [
        "search for books about machine learning", "find a book on history",
        "something about machine learning for beginners", "look up titles about biology",
        "find books matching this topic", "search the catalog for algorithms",
        "i need a book about economics", "show me books on artificial intelligence",
        "find titles related to this subject", "look for a book about networking",
        "i am looking for a book on statistics", "do you have anything on law",
        "books about african history", "find me books about python programming",
        "i want to read about climate", "anything on marketing",
        "find me something on accounting", "books covering geography", "show books on criminal law",
    ],
    "author_search": [
        "which books are available from this author", "show me books by this author",
        "what titles has this author written", "find available books by an author",
        "which titles can i borrow from this writer", "list books written by an author",
        "does the library have books by this author", "show available titles from a writer",
        "what books can i borrow by this author", "find all titles by an author",
        "show books written by this author", "any books from this writer", "find works by an author",
    ],
    "forecast": [
        "forecast book demand", "predict which books will be borrowed next month",
        "which titles need more copies", "show the borrowing demand forecast",
        "predict demand by genre", "what books will be popular next week",
        "estimate future library borrowing", "show projected loans",
        "which category will have the most demand", "forecast borrowing for the next month",
        "which books should the library buy", "which books are in high demand",
        "books we should purchase more copies of", "show demand forecast for exam period",
        "what will students borrow most soon",
        "which books will run short", "predict borrowing during exams", "what titles need more copies soon",
    ],
    "policy": [
        "how long can i keep a book", "what is the loan period", "how many days do i have to return a book",
        "how many books can i borrow at once", "what is the borrowing limit",
        "how much is the fine for late books", "what happens if i return a book late",
        "what is the fine per day", "how do i renew my loan", "how many times can i renew",
        "what is the renewal policy", "do you send reminders", "will i be notified before my book is due",
        "what are the library rules", "tell me the library policy", "what are the borrowing rules",
        "how much do overdue books cost", "are renewals allowed",
        "is there a limit on how many books i can take", "will the library warn me before a book is due", "what is the penalty for keeping a book too long",
    ],
    "my_loans": [
        "what books do i have", "show my loans", "which books have i borrowed",
        "list the books i currently have out", "my borrowed books", "what am i currently borrowing",
        "how much do i owe in fines", "what is my fine", "do i have any fines", "show my account",
        "how many books do i have checked out", "what do i owe the library", "my current loans",
        "books i have on loan", "how many more books can i borrow", "what is my borrowing status",
        "how many books do i have right now", "what do i owe", "list what i am holding",
    ],
    "renew": [
        "renew my book", "i want to renew a loan", "extend my loan", "can you renew clean code for me",
        "please renew this book", "renew it", "extend the due date", "i need more time with this book",
        "renew my loan on this title", "give me more time to return this book",
        "extend the loan period for this book", "renew the book i borrowed", "i would like to renew",
        "renew that one",
        "i need more days with this book", "can i keep this book longer", "extend it please",
    ],
    "similar": [
        "books similar to this one", "what else is like this book", "find books like clean code",
        "something similar to this title", "other books like it", "more books like this",
        "what is similar to this", "show me similar books", "books related to this book",
        "if i liked this what else would i like", "titles like atomic habits", "anything else like it",
        "what is like this book", "find more like it", "what other books resemble this one",
    ],
    "smalltalk": [
        "hello", "hi", "hey there", "good morning", "good afternoon", "hi assistant", "hello library",
        "hey", "greetings", "thanks", "thank you", "thank you so much", "bye", "goodbye", "that is all",
        "see you later", "great thanks", "cheers",
        "good evening to you", "thanks again", "see you tomorrow",
        "have a nice day",
    ],
    "help": [
        "help", "what can you do", "how do i use this assistant",
        "show available commands", "what can i ask you", "tell me how this works",
        "what features are available", "can you explain your services",
        "i need help", "what are you able to do", "how does this work", "commands",
        "what questions can i ask", "can you assist me", "how can i use this chat",
    ],
    "unknown": [
        "tell me a joke", "what is the weather outside", "write an essay for me",
        "translate this sentence", "what is the capital of france", "solve my math homework",
        "play a song", "tell me a story", "who won the game", "what time is it",
        "order me a pizza", "book me a flight to mombasa", "what is twenty five times four",
        "who is the richest person", "tell me a fun fact", "can you do my assignment",
        "sing me a song", "how do i cook rice",
        "who is the best footballer", "how high is mount everest", "what is the tallest building",
        "sing a song for me", "what is two plus two", "what is the meaning of life",
        "how far is the moon", "what is the best phone to buy",
    ],
}

# Fixed answers about library rules. Several phrasings per topic: the chatbot
# finds the closest one with TF-IDF similarity and replies with the answer.
FAQ = [
    {
        "topic": "Loan period",
        "questions": ["how long can i keep a book", "what is the loan period", "how many days can i borrow a book for",
                      "when do i have to return a book", "how long is the borrowing period"],
        "answer": f"Books are lent for {LOAN_DAYS} days.",
    },
    {
        "topic": "Borrowing limit",
        "questions": ["how many books can i borrow", "what is the borrowing limit",
                      "how many books can i have at the same time", "maximum number of books per student"],
        "answer": f"You can have up to {MAX_ACTIVE_LOANS} books on loan at the same time.",
    },
    {
        "topic": "Fines",
        "questions": ["how much is the fine for a late book", "what is the fine per day",
                      "what happens if i return a book late", "how much do overdue books cost",
                      "do you charge for late returns"],
        "answer": (f"The fine is KES {FINE_PER_DAY} per overdue day, per book. While you have an overdue "
                   "book you can not borrow more books or renew."),
    },
    {
        "topic": "Renewals",
        "questions": ["how do i renew a book", "can i renew my loan", "how many times can i renew",
                      "what is the renewal policy", "are renewals allowed", "can i extend my loan"],
        "answer": (f"Each loan can be renewed {MAX_RENEWALS} time, for another {LOAN_DAYS} days counted from "
                   "the day you renew. Overdue books can not be renewed. Just ask me: “renew Clean Code”."),
    },
    {
        "topic": "Reminders",
        "questions": ["do you send reminders", "will i be notified before my book is due",
                      "do i get a notification when a book is overdue", "how will i know when to return a book"],
        "answer": (f"The library agent sends a reminder {REMINDER_DAYS_BEFORE_DUE} days before a book is due, "
                   "and an overdue notice every day a book is late."),
    },
]


# Words a student may use for a genre. Used to turn "recommend law books" into a genre filter.
# The longest matching phrase wins, so "computer science" beats the bare word "science".
GENRE_WORDS = {
    "Computer Science": ["computer science", "computing", "programming", "software", "cs"],
    "Business & Economics": ["business", "economics", "finance", "entrepreneurship"],
    "African Literature": ["african literature", "african fiction", "african novels"],
    "World Fiction": ["world fiction", "fiction", "novels", "novel"],
    "Natural Sciences": ["natural sciences", "natural science", "science", "physics", "chemistry", "biology"],
    "History & Politics": ["history", "politics", "political"],
    "Self-Development": ["self development", "self-development", "self help", "self-help", "personal development"],
    "Mathematics & Statistics": ["mathematics", "maths", "math", "statistics", "stats"],
    "Engineering": ["engineering"],
    "Health & Medicine": ["health", "medicine", "medical"],
    "Law & Governance": ["law", "legal", "governance", "constitution"],
}