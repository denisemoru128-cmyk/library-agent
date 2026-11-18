import streamlit as st

st.set_page_config(page_title="Library Agent", page_icon="📚")

st.title("📚 Library Agent")
st.write("Welcome! Ask me about books.")

# Simple book library
books = [
    {"title": "Things Fall Apart", "author": "Chinua Achebe", "category": "Fiction"},
    {"title": "The River Between", "author": "Ngugi wa Thiong'o", "category": "Fiction"},
    {"title": "Long Walk to Freedom", "author": "Nelson Mandela", "category": "Biography"},
    {"title": "Python Crash Course", "author": "Eric Matthes", "category": "Programming"},
    {"title": "Clean Code", "author": "Robert Martin", "category": "Programming"},
]

search = st.text_input("Search for a book or author:")

st.subheader("Available Books")

for book in books:
    if search.lower() in book["title"].lower() or search.lower() in book["author"].lower() or search == "":
        st.write(f"**{book['title']}** - {book['author']} ({book['category']})")

st.divider()
st.write("Ask a question:")
question = st.text_input("e.g. Do you have a book by Ngugi?")

if question:
    st.success(f"Library Agent: Let me check for '{question}'...")
    found = [b for b in books if b["author"].lower() in question.lower() or b["title"].lower() in question.lower()]
    if found:
        for b in found:
            st.write(f"✅ Yes! We have **{b['title']}** by {b['author']}")
    else:
        st.write("Sorry, I didn't find that book. Try searching above.")import streamlit as st

st.set_page_config(page_title="Library Agent", page_icon="📚")

st.title("📚 Library Agent")
st.write("Welcome! Ask me about books.")

# Simple book library
books = [
    {"title": "Things Fall Apart", "author": "Chinua Achebe", "category": "Fiction"},
    {"title": "The River Between", "author": "Ngugi wa Thiong'o", "category": "Fiction"},
    {"title": "Long Walk to Freedom", "author": "Nelson Mandela", "category": "Biography"},
    {"title": "Python Crash Course", "author": "Eric Matthes", "category": "Programming"},
    {"title": "Clean Code", "author": "Robert Martin", "category": "Programming"},
]

search = st.text_input("Search for a book or author:")

st.subheader("Available Books")

for book in books:
    if search.lower() in book["title"].lower() or search.lower() in book["author"].lower() or search == "":
        st.write(f"**{book['title']}** - {book['author']} ({book['category']})")

st.divider()
st.write("Ask a question:")
question = st.text_input("e.g. Do you have a book by Ngugi?")

if question:
    st.success(f"Library Agent: Let me check for '{question}'...")
    found = [b for b in books if b["author"].lower() in question.lower() or b["title"].lower() in question.lower()]
    if found:
        for b in found:
            st.write(f"✅ Yes! We have **{b['title']}** by {b['author']}")
    else:
        st.write("Sorry, I didn't find that book. Try searching above.")
