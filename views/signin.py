"""The sign-in screen.

This is a DEMO sign-in: a registration number is enough, there is no password. It exists so that the app
opens on a clear "who are you?" step and so that a student and the librarian can be shown side by side, in
two browser tabs (each tab is its own session). Real authentication would replace this screen.
"""
import streamlit as st

from library_agent import circulation

from . import theme

SPINE_SIZES = [(300, 50), (262, 42), (322, 38), (280, 44), (308, 36), (340, 56), (290, 46), (314, 40), (270, 42),
               (326, 50), (286, 38)]


def demo_students(conn) -> list[dict]:
    """Two students that show the two situations worth demonstrating, chosen from the live data:
    one in good standing with several books out, and one with the longest-overdue book who also holds
    another loan (so the 'renewing is blocked' rule can be shown)."""
    loans = circulation.open_loans(conn)
    by_student = loans.groupby("student_id").agg(
        books=("loan_id", "size"), late=("days_left", lambda days: int((days < 0).sum())), worst=("days_left", "min"))
    names = loans.drop_duplicates("student_id").set_index("student_id")[["student_name", "reg_no"]]
    accounts = []
    good = by_student[(by_student["late"] == 0) & (by_student["books"] >= 3)]
    if not good.empty:
        sid = int(good.index[0])
        accounts.append({"student_id": sid, "text": f"{names.at[sid, 'student_name']} ({names.at[sid, 'reg_no']}): "
                                                     f"{int(good.at[sid, 'books'])} books on loan, nothing overdue"})
    late = by_student[(by_student["late"] >= 1) & (by_student["books"] >= 2)].sort_values("worst")
    if not late.empty:
        sid = int(late.index[0])
        accounts.append({"student_id": sid, "text": f"{names.at[sid, 'student_name']} ({names.at[sid, 'reg_no']}): "
                                                     f"a book {-int(late.at[sid, 'worst'])} days overdue"})
    return accounts


def find_student(students, text: str):
    """Match 'REG-0035', 'reg0035', ' 35 ' to a student row, or None."""
    typed = text.strip().upper().replace(" ", "")
    if not typed:
        return None
    digits = typed.replace("REG", "").replace("-", "")
    if digits.isdigit():
        typed = f"REG-{int(digits):04d}"
    match = students[students["reg_no"].str.upper() == typed]
    return None if match.empty else match.iloc[0]


def _sign_in(user: dict) -> None:
    st.session_state["user"] = user
    st.session_state["section"] = None           # the shell picks the first screen for this role
    st.rerun()


def render(conn, students, books) -> None:
    art, form = st.columns([5, 4], gap="large")
    with art:
        spines = "".join(theme.spine(row.title, row.genre, *SPINE_SIZES[i % len(SPINE_SIZES)])
                         for i, row in enumerate(books.groupby("genre").head(1).head(11).itertuples()))
        st.html('<div class="si-art"><h2>Find it. Borrow it. Get a reminder before it is due.</h2>'
                f'<div class="shelf">{spines}</div></div>')
    with form:
        st.html('<div class="si-form">'
                f'<img class="logo" src="{theme.data_uri("karatina_logo.png")}" alt="Karatina University">'
                '<h1>Library Agent</h1><p>Sign in with your registration number.</p></div>')
        with st.form("sign_in"):
            typed = st.text_input("Registration number", placeholder="for example REG-0035")
            submitted = st.form_submit_button("Continue", type="primary")
        if submitted:
            student = find_student(students, typed)
            if student is None:
                st.error("No student has that registration number. Check it, or use a demo account below.")
            else:
                _sign_in({"role": "student", "student_id": int(student["student_id"])})
        st.html('<div class="si-or">Demo accounts, no password</div>')
        with st.container(key="demo_accounts"):
            for account in demo_students(conn):
                if st.button(account["text"], key=f"demo_{account['student_id']}"):
                    _sign_in({"role": "student", "student_id": account["student_id"]})
            if st.button("Librarian: circulation desk, agent plan, buying advice", key="demo_librarian"):
                _sign_in({"role": "librarian"})
        st.html('<div class="fine-print">All people, books and loans are simulated. '
                'A real sign-in would replace this screen.</div>')