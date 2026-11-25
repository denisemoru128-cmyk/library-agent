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


def apply(hide_sidebar: bool = False, side_panel: bool = False) -> None:
    """Add the stylesheet to the page. ``hide_sidebar`` is used on the sign-in screen, ``side_panel`` makes
    room for the docked panel on the right (the assistant, or the agent plan on the librarian Dashboard)."""
    extra = ('[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],'
             '[data-testid="collapsedControl"]{display:none!important}.stApp{background:#fff}' if hide_sidebar else "")
    if side_panel:
        extra += SIDE_PANEL_CSS
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
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primary"]:focus:not(:active){background:#1C4D3A;border-color:#1C4D3A;color:#fff}
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

/* page headings */
[data-testid="stMarkdownContainer"] h1{font-size:30px!important;line-height:1.15!important;padding:0 0 .1rem}
[data-testid="stMarkdownContainer"] h3{font-size:19px!important;padding:.2rem 0 .3rem}
.sub{margin:0 0 1rem;color:var(--muted);font-size:15.5px}

/* stat strip */
.strip{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(0,1fr);border-top:1px solid var(--rule);
  border-bottom:1px solid var(--rule);margin:.2rem 0 .5rem}
.cell{padding:11px 16px;border-left:1px solid var(--rule)}
.cell:first-child{padding-left:0;border-left:0}
.cell small{display:block;font-size:13px;color:var(--muted)}
.cell b{display:block;font-family:var(--serif);font-size:26px;line-height:1.2;font-variant-numeric:tabular-nums}
.cell b i{font-family:"Source Sans","Source Sans 3",system-ui,sans-serif;font-style:normal;font-weight:400;font-size:15px;color:var(--muted)}
.cell span{display:block;font-size:13px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cell.late b{color:var(--late)}

/* a book: cover + words */
.bk{display:flex;gap:12px;align-items:flex-start}
.bi{min-width:0}
.bt{font-family:var(--serif);font-weight:600;font-size:15.5px;line-height:1.25}
.ba{font-size:13.5px;color:var(--muted);margin:1px 0 5px}
.bl{font-size:13.5px;margin-top:2px;color:var(--muted)}
.bl.ink{color:var(--ink);font-size:13px}
.bl.ok{color:var(--ok);font-weight:600}
.bl.soon{color:#8A6410;font-weight:600}
.bl.late{color:var(--late);font-weight:600}
.hint{font-size:12.5px;color:var(--muted);margin-top:-.25rem}
.due{padding-top:.35rem}
.due b{display:block;font-size:14.5px}
.due small{font-size:13px;color:var(--muted)}
.due small.late{color:var(--late);font-weight:600}
.due small.soon{color:#8A6410;font-weight:600}

/* cards (Home): the button sits under the words, not under the cover */
[class*="st-key-card_"] .hint{margin-left:88px}
[class*="st-key-card_"] .stButton{margin-left:88px;width:calc(100% - 88px)!important}
[class*="st-key-card_"] [data-testid="stVerticalBlock"]{gap:.45rem}
[class*="st-key-link_"]{display:flex;justify-content:flex-end;width:100%!important}
[class*="st-key-link_"] button,[class*="st-key-link_"] button>div{justify-content:flex-end;text-align:right}
[class*="st-key-link_"] button{color:var(--forest);text-decoration:underline;font-size:13.5px;padding:0}

/* rows (Search, My loans, For you) */
[class*="st-key-row_"],[class*="st-key-lrow_"]{border-bottom:1px solid var(--rule);border-left:3px solid transparent;padding:.7rem .3rem .7rem .8rem}
[class*="st-key-lrow_"]{background:var(--late-bg);border-left-color:var(--late)}
[class*="st-key-row_search_"],[class*="st-key-row_reco_"]{border-left:0;padding-left:.3rem}
.banner{display:flex;flex-direction:column;padding:12px 16px;border-left:4px solid var(--late);background:var(--late-bg);margin:.5rem 0 1rem}
.banner b{font-size:15.5px;color:#8F0016}
.banner span{font-size:14.5px}
.empty{padding:1rem 0}
.empty b{display:block;font-size:15px}
.empty span{display:block;font-size:14px;color:var(--muted);margin-top:2px}
.stButton button:disabled{opacity:.5}
/* small table (Books to buy first) */
.tbl{width:100%;border-collapse:collapse;font-size:14.5px}
.tbl th{text-align:left;font-weight:600;font-size:13px;color:var(--muted);padding:6px 10px 8px 0;border-bottom:1px solid var(--ink)}
.tbl td{padding:11px 10px 11px 0;border-bottom:1px solid var(--rule)}
.tbl .n{text-align:right;font-variant-numeric:tabular-nums}
.tbl th:last-child,.tbl td.buy{padding-left:28px}
.tbl .buy{color:#8A6410;font-weight:600}
.tbl .late{color:var(--late);font-weight:600;font-variant-numeric:tabular-nums}
.st-key-agent_clock{gap:.35rem!important}
.st-key-agent_clock button{color:var(--forest);text-decoration:underline;font-size:14px;padding:0;min-height:0;border:0;background:transparent;justify-content:flex-start}
.st-key-quality_doc h1{display:none}
.st-key-quality_doc h2{font-size:1.45rem;margin-top:1.6rem}
.st-key-quality_doc table{font-size:14.5px}
@media (max-width:700px){.strip{grid-auto-flow:row;grid-template-columns:1fr 1fr}.cell{border-left:0;padding-left:0}}
"""


# The docked panel on the right. Only added where there is one, because it reserves space.
SIDE_PANEL_CSS = """
[data-testid="stMainBlockContainer"]{padding-left:2rem!important;padding-right:calc(340px + 1.6rem)!important}
.st-key-chat_panel,.st-key-plan_panel{position:fixed;top:0;right:0;bottom:0;width:340px;z-index:1000000;background:#fff;
  border-left:1px solid var(--rule);display:flex;flex-direction:column;gap:0!important}
.st-key-chat_head,.st-key-plan_head{flex:none;padding:12px 12px 8px 18px;border-bottom:1px solid var(--rule);gap:0!important}
.chat-h b{display:block;font-family:var(--serif);font-size:19px;line-height:1.2}
.chat-h small{font-size:12.5px;color:var(--muted)}
.st-key-chat_head [class*="st-key-chat_clear"]{display:flex;justify-content:flex-end;width:100%!important}
.st-key-chat_head [class*="st-key-chat_clear"] button{color:var(--muted);font-size:12.5px;padding:0;text-decoration:underline}
/* column-reverse keeps the newest message in view: the log starts scrolled to the bottom */
.st-key-chat_panel>[data-testid="stLayoutWrapper"]:has(.st-key-chat_log){flex:1 1 0;min-height:0}
.st-key-chat_log>[data-testid="stLayoutWrapper"],.st-key-chat_msgs{flex:none!important}
.st-key-chat_log{flex:1 1 0;min-height:0;overflow-y:auto;display:flex;flex-direction:column-reverse;justify-content:flex-start!important;padding:14px 14px 0}
.st-key-chat_chips{flex:none;padding:8px 14px 0;gap:.35rem!important}
.st-key-chat_chips [data-testid="stHorizontalBlock"]{gap:.4rem}
.st-key-chat_chips button{border-radius:999px;min-height:0;padding:.15rem .7rem;font-size:13px;background:#fff;
  border:1px solid var(--rule);color:var(--forest);width:100%}
.st-key-chat_chips button:hover{border-color:var(--gold)}
.st-key-chat_entry{flex:none;padding:8px 14px 14px}
.st-key-chat_entry [data-testid="stChatInput"]{border-radius:8px;background:#fff}

.msg{display:flex;flex-direction:column;max-width:90%;margin:0 0 12px}
.msg.me{align-items:flex-end;margin-left:auto}
.msg.bot{align-items:flex-start}
.bub{padding:9px 13px;border-radius:14px;font-size:14.5px;line-height:1.4;overflow-wrap:anywhere}
.bub p{margin:0}.bub p+p,.bub ul{margin-top:.35rem}.bub ul{padding-left:1.1rem;margin-bottom:0}
.me .bub{background:var(--forest);color:#fff;border-bottom-right-radius:3px}
.bot .bub{background:#EDF2EE;border-bottom-left-radius:3px}
.meta{display:flex;gap:6px;align-items:center;font-size:11.5px;color:var(--muted);margin-top:3px}
.tick{position:relative;display:inline-block;width:17px;height:10px}
.tick::before,.tick::after{content:"";position:absolute;bottom:2px;width:4px;height:8px;box-sizing:border-box;
  border:solid #6E7F75;border-width:0 1.8px 1.8px 0;transform:rotate(45deg)}
.tick::before{left:2px}.tick::after{left:7px;display:none}
.tick.ok::after{display:block}.tick.ok::before,.tick.ok::after{border-color:#0A8A45}
.und{font-size:11.5px;color:var(--muted);margin-top:4px}
.minis{align-self:stretch;margin-top:6px}
.mini{display:flex;gap:10px;align-items:center;padding:7px 0;border-top:1px solid var(--rule)}
.mini b{font-family:var(--serif);font-size:13.5px;display:block;line-height:1.2;font-weight:600}
.mini span{font-size:12.5px;color:var(--muted);display:block}
.mini .ok{color:var(--ok);font-weight:600}.mini .soon{color:#8A6410;font-weight:600}.mini .late{color:var(--late);font-weight:600}
.more{font-size:12.5px;color:var(--muted);padding-top:6px}

/* the agent plan (librarian Dashboard) */
.st-key-plan_panel>[data-testid="stLayoutWrapper"]:has(.st-key-plan_body){flex:1 1 0;min-height:0}
.st-key-plan_body{flex:1 1 0;min-height:0;overflow-y:auto;padding:14px 18px 6px;gap:.2rem!important}
.st-key-plan_foot{flex:none;padding:10px 14px 14px;border-top:1px solid var(--rule);background:#F7F9F7;gap:.3rem!important}
.st-key-plan_foot button{width:100%}
.st-key-plan_foot [class*="st-key-plan_advance"] button,.st-key-plan_foot [class*="st-key-plan_today"] button{
  color:var(--forest);text-decoration:underline;font-size:13.5px;padding:0;min-height:0;border:0;background:transparent}
.st-key-plan_body [class*="st-key-plan_more"] button{color:var(--forest);text-decoration:underline;font-size:13px;padding:0;min-height:0;justify-content:flex-start}
.plan-note{background:#FBF3E2;border-left:3px solid var(--gold);padding:9px 12px;font-size:13.5px;line-height:1.35}
.plan-note.done{background:#E6F2EA;border-left-color:var(--ok)}
.plan-note b{display:block}
.pl-h{display:flex;align-items:center;gap:8px;margin-top:12px}
.pl-h b{font-family:var(--serif);font-size:17px}
.pl-h .count{background:var(--forest);color:#fff;border-radius:999px;font-size:12px;font-weight:600;padding:1px 9px}
.pl-sub{font-size:12.5px;color:var(--muted);margin:2px 0 4px}
.pl-item{padding:8px 0;border-top:1px solid var(--rule)}
.pl-item b{display:block;font-size:14px}
.pl-item span{display:block;font-size:13.5px}
.pl-item small{font-size:12.5px;color:var(--muted)}
.pl-item small.late{color:var(--late);font-weight:600}
.pl-none{font-size:13.5px;color:var(--muted);padding:6px 0}

@media (max-width:1000px){
  [data-testid="stMainBlockContainer"]{padding-right:1rem!important}
  .st-key-chat_panel,.st-key-plan_panel{position:static;flex:none!important;width:auto;height:560px;margin-top:1.5rem;border:1px solid var(--rule)}
}
"""