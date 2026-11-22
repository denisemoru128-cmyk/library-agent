"""Look and feel: the colour palette, the fonts and the small HTML pieces the screens share.

Palette (taken from the Karatina University logo): forest green for the sidebar and buttons, gold for the
active item, red ONLY for overdue books and fines. Headings use Source Serif and text uses Source Sans,
both of which Streamlit already ships, so nothing is downloaded.

The streamlit theme file (.streamlit/config.toml) sets the base colours; the CSS below does the rest, so
the look does not depend on which Streamlit version is installed.
"""
import base64
import functools
import html
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# One colour per genre for the generated book covers.
GENRE_COLORS = {
    "Computer Science": "#22405F", "Business & Economics": "#6B4F1D", "African Literature": "#7A2E2E",
    "World Fiction": "#47346B", "Natural Sciences": "#1E5A5A", "History & Politics": "#5A3A24",
    "Self-Development": "#3C6B4F", "Mathematics & Statistics": "#2F5C8F", "Engineering": "#3F4A4F",
    "Health & Medicine": "#7C2D4F", "Law & Governance": "#1E2433",
}
DEFAULT_COLOR = "#3F4A4F"


@functools.lru_cache(maxsize=None)
def data_uri(name: str) -> str:
    """An image from assets/ as a data: URL, so it can sit inside our own HTML."""
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


def cover(title: str, genre: str, author: str = "", width: int = 76) -> str:
    """A generated book cover: genre colour, title in serif, gold rule. No image files needed.

    The font size shrinks for long words so a title such as "Epidemiology" never gets cut off.
    """
    short = title.split(":")[0]
    longest = max((len(word) for word in short.split()), default=1)
    font = min(11.0, 73 / (0.66 * longest))
    colour = GENRE_COLORS.get(genre, DEFAULT_COLOR)
    first_author = author.split(",")[0]
    return (f'<div class="cover" style="--c:{colour};--fs:{font:.1f}cqw;width:{width}px" aria-hidden="true">'
            f'<div class="in"><span class="t">{html.escape(short)}</span>'
            f'<span class="a">{html.escape(first_author)}</span></div></div>')


def spine(title: str, genre: str, height: int, width: int) -> str:
    colour = GENRE_COLORS.get(genre, DEFAULT_COLOR)
    return (f'<div class="spine" style="--c:{colour};height:{height}px;width:{width}px">'
            f'<span>{html.escape(title.split(":")[0])}</span></div>')


def apply(hide_sidebar: bool = False) -> None:
    """Add the stylesheet to the page. ``hide_sidebar`` is used on the sign-in screen."""
    extra = ('[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],'
             '[data-testid="collapsedControl"]{display:none!important}.stApp{background:#fff}' if hide_sidebar else "")
    st.markdown(f"<style>{CSS}{extra}</style>", unsafe_allow_html=True)


CSS = """
:root{--forest:#12372A;--gold:#B8924A;--gold-l:#D9B45F;--ink:#18271F;--muted:#5C6B63;--rule:#DAE2DC;
  --late:#B3001B;--late-bg:#FCEFF0;--ok:#0A6B37;--serif:"Source Serif","Source Serif 4",Georgia,serif}
.stApp{background:#F2F5F2;color:var(--ink)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"],.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:none}
h1,h2,h3,h4{font-family:var(--serif)!important;font-weight:600!important;letter-spacing:-.01em}
[data-testid="stMetricValue"]{font-family:var(--serif)}

/* sidebar: text-only navigation */
section[data-testid="stSidebar"]{background:var(--forest);border-right:0;min-width:208px!important;max-width:208px!important;width:208px!important}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding:0!important}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{min-height:2.2rem;padding:.4rem .8rem}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:.2rem 0 1rem}
section[data-testid="stSidebar"] [data-testid="stElementContainer"],section[data-testid="stSidebar"] .stButton{width:100%!important}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:.15rem}
.brand{display:flex;align-items:center;gap:10px;padding:0 18px 20px}
.brand img{height:40px;width:auto}
.brand b{font-family:var(--serif);font-size:17px;display:block;line-height:1.15;color:#fff}
.brand small{font-size:12px;color:#B9CCC0}
section[data-testid="stSidebar"] .stButton button{width:100%!important;justify-content:flex-start;background:transparent;border:0;
  border-left:3px solid transparent;border-radius:0;color:#C9D8CF;padding:.4rem 1rem .4rem .9rem;box-shadow:none;min-height:0}
section[data-testid="stSidebar"] .stButton button p{color:inherit;font-size:15.5px;text-align:left}
section[data-testid="stSidebar"] .stButton button>div{width:100%;justify-content:flex-start}
section[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"]{width:100%}
section[data-testid="stSidebar"] .stButton button:hover{color:#fff;background:rgba(255,255,255,.05)}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]{
  background:rgba(255,255,255,.08);color:#fff;border-left-color:var(--gold-l)}
section[data-testid="stSidebar"] .stButton button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] p{font-weight:600}
.clock{margin:1.2rem 18px 0;padding-top:14px;border-top:1px solid rgba(255,255,255,.14);font-size:12.5px;color:#B9CCC0}
.clock b{display:block;font-size:14.5px;font-weight:600;color:#fff;margin:1px 0}
.side-note{margin:.6rem 18px 0;font-size:12px;color:#B9CCC0}

/* top row */
[data-testid="stPopover"] button{border:1px solid var(--rule);background:#fff;border-radius:6px;min-height:2.4rem}
.top-title{font-family:var(--serif);font-size:15px;color:var(--muted);padding-top:.5rem}

/* sign in */
.si-art{background:var(--forest);padding:56px 52px 0;display:flex;flex-direction:column;justify-content:space-between;
  min-height:calc(100vh - 3rem);border-radius:6px}
.si-art h2{font-size:38px;line-height:1.14;color:#fff;max-width:13em;margin:0;padding:0}
.shelf{display:flex;align-items:flex-end;gap:4px;border-bottom:9px solid #0A2219;box-shadow:0 1px 0 rgba(217,180,95,.55);
  overflow:hidden;padding-top:20px}
.spine{writing-mode:vertical-rl;display:flex;justify-content:center;flex:none;
  background:linear-gradient(90deg,rgba(0,0,0,.28),rgba(255,255,255,.1) 20%,rgba(0,0,0,.14)),var(--c);
  border-top:4px solid var(--gold-l);border-radius:2px 2px 0 0;color:#fff;font-family:var(--serif);font-weight:600;
  font-size:13.5px;overflow:hidden}
.spine span{display:block;max-height:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-top:12px}
.si-form .logo{width:270px;max-width:100%;height:auto;margin:1.5rem 0 .6rem -6px}
.si-form h1{font-size:28px;margin:0;padding:0}
.si-form p{color:var(--muted);margin:.2rem 0 .4rem}
.si-or{display:flex;align-items:center;gap:10px;margin:1.2rem 0 .3rem;font-size:12.5px;color:var(--muted)}
.si-or::before,.si-or::after{content:"";flex:1;height:1px;background:var(--rule)}
[class*="st-key-demo"] [data-testid="stElementContainer"],[class*="st-key-demo"] .stButton{width:100%!important}
[class*="st-key-demo"] button{width:100%;justify-content:flex-start;text-align:left;background:#fff;border:1px solid var(--rule)}
[class*="st-key-demo"] button>div{width:100%;justify-content:flex-start}
[class*="st-key-demo"] button:hover{border-color:var(--gold)}
.fine-print{font-size:12.5px;color:var(--muted);margin-top:.8rem}

/* generated book cover */
.cover{aspect-ratio:3/4;container-type:inline-size;flex:none;
  background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(0,0,0,.2)),var(--c);border-radius:2px 5px 5px 2px;
  box-shadow:inset 7px 0 0 rgba(0,0,0,.22),0 1px 0 rgba(0,0,0,.25),0 9px 12px -9px rgba(18,55,42,.7);overflow:hidden;color:#fff}
.cover .in{height:100%;display:flex;flex-direction:column;padding:11cqw 9cqw 9cqw 18cqw}
.cover .in::before{content:"";width:22cqw;height:2px;background:var(--gold-l);margin-bottom:6cqw;flex:none}
.cover .t{font-family:var(--serif);font-weight:600;font-size:var(--fs,11cqw);line-height:1.12;display:-webkit-box;
  -webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:break-word}
.cover .a{margin-top:auto;font-size:7.5cqw;opacity:.82;line-height:1.2}
"""