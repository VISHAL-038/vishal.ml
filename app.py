"""
Vishal | ML Engineer portfolio  -  built with Streamlit.
Edit the CONTENT section below; everything else is layout + styling.
Run locally:  streamlit run app.py
"""
import base64
import math
from pathlib import Path

import altair as alt
import numpy as np
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Vishal | ML Engineer", page_icon="✦", layout="wide",
                   initial_sidebar_state="collapsed")


NAME = "Vishal"
SUBTITLE = "M.Tech · ML · Data Science"
INTRO = "An M.Tech student who likes taking messy data all the way to a model that runs."
BADGE_TEXT = "DATA ✦ MODELS ✦ SYSTEMS ✦ "

EMAIL = "vishaal03.it@gmail.com"
LINKEDIN = "https://www.linkedin.com/in/vishal-datascience/"      
GITHUB = "https://github.com/VISHAL-038"    
RESUME = "https://drive.google.com/file/d/1AAlzpRmzk8WAFxs3KED0tVqQzjhHIHny/view?usp=sharing"        
LINKS = "https://vishallinks-03.streamlit.app/"                        

MARQUEE = ["Machine Learning", "Deep Learning", "Computer Vision", "NLP",
           "Feature Engineering", "Django REST", "Model Evaluation", "Data Science"]

PROJECTS = [
    {
        "id": "p-mediai", "image": "mediai", "short": "MediAI", "title": "MediAI", "tagline": "Healthcare with a prediction layer.",
        "kind": "Full-stack ML", "accent": "#6BE3A0",
        "desc": "An AI-based healthcare platform for disease prediction, appointment scheduling and "
                "prescription management, with secure REST APIs behind it.",
        "points": ["Random Forest, Naïve Bayes and SVM compared on medical datasets",
                   "Django REST APIs for patients, appointments and prescriptions",
                   "SQLite storage for profiles, history and prediction results"],
        "tags": ["Scikit-learn", "Django", "DRF", "Pandas"],
        "stat": "3", "stat_label": "models compared", "link": "https://github.com/VISHAL-038/HealthCare",
    },
    {
        "id": "p-fraud", "image": "fraud", "short": "Fraud detection", "title": "Credit Card Fraud Detection", "tagline": "Finding 0.17% in a haystack.",
        "kind": "Machine learning", "accent": "#FF9B6B",
        "desc": "A pipeline for catching fraudulent transactions in heavily imbalanced data, "
                "judged on precision, recall and F1 rather than raw accuracy.",
        "points": ["Resampling and feature scaling to handle class imbalance",
                   "EDA on the 0.17% fraud rate to find patterns",
                   "Logistic Regression, Decision Tree and Random Forest evaluated"],
        "tags": ["Imbalanced data", "Scikit-learn", "EDA"],
        "stat": "0.17%", "stat_label": "of transactions are fraud", "link": "https://github.com/VISHAL-038/Credit-Card-Fraud-Detection",
    },
    {
        "id": "p-qa", "image": "qa", "short": "QA bot", "title": "QA Bot", "tagline": "Ask your PDFs anything.",
        "kind": "NLP", "accent": "#7AA8FF",
        "desc": "A question-answering app for PDFs that finds the relevant passages with vector "
                "search and answers from them.",
        "points": ["Cohere API for context-aware, document-based answers",
                   "Pinecone vector search for real-time queries",
                   "Streamlit interface"],
        "tags": ["NLP", "Cohere", "Pinecone", "Streamlit"],
        "stat": "PDF", "stat_label": "in, answers out", "link": "https://github.com/VISHAL-038/Q-A-Bot",
    },
    {
        "id": "p-yolo", "image": "yolo", "short": "Object detection", "title": "Real-Time Object Detection", "tagline": "Seeing, frame by frame.",
        "kind": "Computer vision", "accent": "#C58BFF",
        "desc": "A computer vision system that detects multiple objects in live video and draws "
                "labelled bounding boxes as it goes.",
        "points": ["YOLOv3 with OpenCV for fast video inference",
                   "Real-time bounding boxes and class labels"],
        "tags": ["YOLOv3", "OpenCV", "Python"],
        "stat": "Live", "stat_label": "video inference", "link": "https://github.com/VISHAL-038/Real_Time_Object_Detection",
    },
    {
        "id": "p-defacement", "image": "defacement", "short": "Defacement monitor",
        "title": "Defacement Monitoring System", "tagline": "Catching the change as it happens.",
        "kind": "Internship · CyberHawkz", "accent": "#5CE1E6",
        "desc": "An asynchronous system that watches websites and flags defacement in real time, built during "
                "my developer internship (Jan to Jul 2025).",
        "points": ["Async checks with aiohttp and asyncpg for real-time threat detection",
                   "Scraping and scheduling automated with Celery, Redis and APScheduler",
                   "Security hardening (XSS prevention, input validation) and performance tuning"],
        "tags": ["Django", "Celery", "Redis", "PostgreSQL"],
        "stat": "Async", "stat_label": "real-time threat detection", "link": "https://github.com/VISHAL-038/defacement",
    },
]

STAGES = [
    {"name": "Data", "line": "Pull it in and look hard at it.",
     "did": "EDA on a 0.17% fraud rate. Scraping and scheduling with Celery and APScheduler. SQL, SQLite and MongoDB.",
     "used": ["p-fraud", "p-defacement"]},
    {"name": "Clean", "line": "Make it safe to learn from.",
     "did": "Resampling for class imbalance. Input validation and XSS prevention on incoming data.",
     "used": ["p-fraud", "p-defacement"]},
    {"name": "Features", "line": "Give the model something to see.",
     "did": "Feature scaling and engineering for tabular data. Vector representations for searching PDFs.",
     "used": ["p-fraud", "p-qa"]},
    {"name": "Model", "line": "Train it, then try to break it.",
     "did": "Random Forest, Naive Bayes, SVM, Logistic Regression, Decision Tree and YOLOv3, judged on precision, recall and F1.",
     "used": ["p-fraud", "p-mediai", "p-yolo"]},
    {"name": "API", "line": "Let other software use it.",
     "did": "Django REST APIs, async services with aiohttp and asyncpg, role-based access.",
     "used": ["p-mediai", "p-defacement"]},
    {"name": "Deploy", "line": "Keep it running.",
     "did": "Linux hosts, Celery workers with Redis, caching and query tuning, Streamlit front ends.",
     "used": ["p-defacement", "p-qa"]},
]

ABOUT_STATEMENT = ("I'm Vishal. I like machine learning that survives contact with real data: "
                   "the messy kind, the imbalanced kind, the kind that has to sit behind an API.")
ABOUT_P1 = ("I'm pursuing my M.Tech in Computer Science and Engineering, and most of my work sits "
            "between modelling and engineering. I build the model, then build what it has to live in: "
            "REST APIs, databases, schedulers and caches.")
ABOUT_P2 = ("During my developer internship I worked on real-time threat detection and backend "
            "systems in Django, which taught me to care about performance and security as much as "
            "accuracy.")

EDUCATION = [
    ("M.Tech, Computer Science and Engineering", "Chandigarh Group of Colleges, Landran", "2025 — 2027", "8.50 SGPA"),
    ("B.Tech, Computer Science and Engineering", "Rayat Bahra University, Mohali", "2021 — 2025", "8.10 CGPA"),
]
CERTS = ["Data Science Professional Certificate (IBM)",
         "Machine Learning and Deep Learning: Fundamentals and Applications (NPTEL)",
         "Kali Linux (Board Infinity)"]
EXPERIENCE = {
    "role": "Developer Intern", "org": "CyberHawkz Intelligence Services", "when": "Jan 2025 — Jul 2025",
    "points": ["Built an asynchronous defacement monitoring system (Django, aiohttp, asyncpg) for real-time threat detection",
               "Automated scraping and scheduling with Celery, Redis and APScheduler",
               "Built a quiz management app with role-based access, PostgreSQL, REST APIs and bulk mailing",
               "Hardened security (XSS prevention, input validation) and tuned performance (caching, query optimisation)"],
}
TOOLS = ["Python", "Scikit-learn", "Pandas", "NumPy", "OpenCV", "Matplotlib", "Seaborn", "Django",
         "DRF", "SQL", "SQLite", "MongoDB", "Linux", "HTML", "Bootstrap"]

# ════════════════════════════════════════════════════════════════════
# STYLE
# ════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;500;600;800;900&family=Instrument+Serif:ital@0;1&display=swap');
:root{ --bg:#0E0E0E; --ink:#EAE8E1; --mute:#8D8B86; --line:rgba(255,255,255,.09);
       --card:#17171A; --lime:#D9FF5A; --sans:'Hanken Grotesk',system-ui,sans-serif;
       --serif:'Instrument Serif',Georgia,serif; }
html{scroll-behavior:smooth}
#MainMenu,header[data-testid="stHeader"],footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.stApp{background:radial-gradient(rgba(255,255,255,.07) 1px,transparent 1px) 0 0/28px 28px,var(--bg);
       font-family:var(--sans);color:var(--ink)}
.block-container{max-width:none!important;padding:0 4vw 0!important}
.stApp, .stApp p, .stApp div, .stApp span, .stApp label{font-family:var(--sans)}
a{color:inherit;text-decoration:none}
section[id]{scroll-margin-top:90px}

/* nav */
.nav{position:fixed;top:14px;left:2vw;right:2vw;z-index:999;display:flex;align-items:center;justify-content:space-between;
     padding:12px 28px;border:1px solid var(--line);border-radius:22px;background:rgba(18,18,20,.72);backdrop-filter:blur(14px)}
.nav .logo{font-weight:900;font-size:1.25rem;letter-spacing:-.04em}
.nav .logo i{font-family:var(--serif);font-weight:400}
.nav .links{display:flex;align-items:center;gap:30px;font-weight:500;font-size:.95rem}
.nav .links a:hover{color:var(--lime)}
.pill-btn{border:1px solid rgba(255,255,255,.35);border-radius:999px;padding:9px 20px}
.pill-btn:hover{background:var(--lime);color:#0E0E0E!important;border-color:var(--lime)}

/* hero */
.hero{min-height:100vh;display:flex;flex-direction:column;justify-content:space-between;padding:130px 0 40px;position:relative}
.hero h1{margin:0;font-weight:900;letter-spacing:-.06em;line-height:.82;font-size:clamp(4.2rem,17vw,16rem);color:var(--ink)}
.hero h1 .serif{display:block;font-family:var(--serif);font-weight:400;font-style:italic;letter-spacing:-.03em;
                font-size:.44em;margin-left:.9em;margin-top:.12em}
.hero-bottom{display:flex;justify-content:space-between;align-items:flex-end;gap:30px}
.hero p.intro{max-width:520px;font-size:clamp(1.1rem,1.7vw,1.5rem);font-weight:500;line-height:1.4;margin:0;color:#d6d4cd}
.hero .scroll{font-size:.7rem;letter-spacing:.14em;color:var(--mute);font-weight:600}
.badge{position:absolute;right:16vw;top:46%;width:150px;height:150px}
.badge .ring{position:absolute;inset:0;animation:spin 18s linear infinite}
.badge .ring span{position:absolute;left:50%;top:50%;font-size:.72rem;font-weight:700;letter-spacing:.05em;color:var(--ink);
                  height:75px;transform-origin:0 0}
.badge .arrow{position:absolute;inset:0;display:grid;place-items:center;color:var(--lime);font-size:1.9rem;font-weight:800}
@keyframes spin{to{transform:rotate(360deg)}}

/* marquee */
.marquee{overflow:hidden;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:0 -4vw;padding:28px 0;white-space:nowrap}
.marquee .track{display:inline-flex;gap:34px;align-items:center;animation:slide 38s linear infinite}
.marquee .t{font-size:clamp(2rem,4.4vw,4rem);font-weight:800;letter-spacing:-.04em}
.marquee .t.s{font-family:var(--serif);font-style:italic;font-weight:400;color:#bdbbb4;letter-spacing:-.01em}
.marquee .star{color:var(--lime);font-size:1.6rem}
@keyframes slide{to{transform:translateX(-50%)}}

/* section heads */
.sec-head{display:flex;align-items:flex-end;gap:22px;padding:130px 0 20px}
.sec-head .no{color:var(--lime);font-size:.85rem;font-weight:700;margin-bottom:1.2rem;font-feature-settings:"tnum"}
.sec-head h2{margin:0;font-weight:900;letter-spacing:-.055em;line-height:.9;font-size:clamp(3rem,9vw,8rem)}
.sec-head h2 i{font-family:var(--serif);font-weight:400;letter-spacing:-.02em}
.sec-sub{color:var(--mute);max-width:430px;font-size:1.05rem;line-height:1.5;margin:0 0 60px 62px}

/* project cards (sticky stack) */
.stack{position:relative;padding-bottom:30px}
.card{position:sticky;top:90px;display:grid;grid-template-columns:1fr 1.25fr;gap:0;min-height:560px;margin-bottom:34px;
      background:var(--card);border:1px solid var(--line);border-radius:34px;padding:22px;overflow:hidden}
.card .l{padding:22px 30px 22px 22px;display:flex;flex-direction:column}
.card .meta{display:flex;justify-content:space-between;color:var(--mute);font-size:.78rem;font-weight:600;
            padding-bottom:16px;border-bottom:1px solid var(--line)}
.card .meta b{color:var(--c)}
.card h3{margin:auto 0 8px;font-size:clamp(2.3rem,4.4vw,4.2rem);font-weight:800;letter-spacing:-.05em;line-height:.95;padding-top:40px}
.card .tag{font-family:var(--serif);font-style:italic;color:var(--c);font-size:clamp(1.5rem,2.4vw,2.2rem);margin:0 0 18px}
.card p.d{color:#a9a7a1;line-height:1.55;font-size:1rem;margin:0 0 14px;max-width:520px;font-weight:300}
.card ul{margin:0 0 18px;padding-left:18px;color:#a9a7a1;font-size:.93rem;line-height:1.55;font-weight:300}
.card ul li::marker{color:var(--c)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:22px}
.chip{border:1px solid var(--line);border-radius:999px;padding:7px 15px;font-size:.78rem;font-weight:600;color:#cfcdc6}
.go{display:inline-flex;align-items:center;gap:12px;font-weight:700}
.go span{width:46px;height:46px;border-radius:50%;background:var(--c);color:#0E0E0E;display:grid;place-items:center;transition:transform .25s}
.go:hover span{transform:rotate(-45deg)}
.card .r{border-radius:26px;position:relative;overflow:hidden;display:flex;flex-direction:column;justify-content:center;align-items:center;
         background:radial-gradient(circle at 30% 20%,color-mix(in srgb,var(--c) 38%,#101215),#101215 70%);border:1px solid var(--line)}
.card .r::before{content:"";position:absolute;inset:0;background:radial-gradient(rgba(255,255,255,.1) 1px,transparent 1px) 0 0/22px 22px;opacity:.5}
.card .big{position:relative;font-family:var(--serif);font-style:italic;color:var(--c);font-size:clamp(5rem,13vw,12rem);line-height:1;letter-spacing:-.03em}
.card .cap{position:relative;font-weight:600;color:#cfcdc6;margin-top:6px}

/* about */
.about{display:grid;grid-template-columns:.8fr 1.2fr;gap:5vw;align-items:start;margin-top:20px}
.photo{aspect-ratio:4/5;border-radius:30px;overflow:hidden;background:linear-gradient(160deg,#2a2a2e,#141416);display:grid;place-items:center;border:1px solid var(--line)}
.photo img{width:100%;height:100%;object-fit:cover;filter:grayscale(1)}
.photo .ph{font-family:var(--serif);font-style:italic;font-size:9rem;color:var(--lime)}
.about h3{font-size:clamp(1.7rem,3vw,2.8rem);font-weight:800;letter-spacing:-.04em;line-height:1.08;margin:0 0 34px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:34px;color:#a9a7a1;font-weight:300;line-height:1.6}
.stats{display:flex;gap:6vw;border-top:1px solid var(--line);margin-top:44px;padding-top:34px}
.stats .n{font-family:var(--serif);font-style:italic;color:var(--lime);font-size:clamp(3.4rem,6vw,5.6rem);line-height:1}
.stats .k{color:var(--mute);font-size:.8rem;font-weight:600;margin-top:10px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:5vw;margin-top:90px}
.lab{color:var(--mute);font-size:.8rem;font-weight:700;letter-spacing:.06em;margin-bottom:16px}
.row{display:flex;justify-content:space-between;gap:20px;padding:22px 0;border-top:1px solid var(--line)}
.row:last-child{border-bottom:1px solid var(--line)}
.row h4{margin:0 0 4px;font-size:1.25rem;font-weight:700;letter-spacing:-.02em}
.row small{color:var(--mute);font-size:.95rem}
.row .y{color:var(--lime);font-weight:700;font-size:.82rem;text-align:right;white-space:nowrap}
.row .y small{display:block;color:var(--mute);margin-top:4px;font-weight:500}
.exp ul{color:#a9a7a1;font-weight:300;line-height:1.6;padding-left:18px;margin:10px 0 0;font-size:.95rem}
.exp ul li::marker{color:var(--lime)}
.tools{display:flex;flex-wrap:wrap;gap:10px}
.tool{border:1px solid var(--line);border-radius:999px;padding:14px 22px;font-weight:600}
.certs{margin-top:34px;color:#a9a7a1;font-weight:300;line-height:1.7;font-size:.95rem}

/* playground (native streamlit widgets) */
div[data-testid="stSlider"] label p{font-weight:600}
.metric{border:1px solid var(--line);border-radius:22px;padding:20px 22px;background:var(--card)}
.metric .n{font-family:var(--serif);font-style:italic;color:var(--lime);font-size:3rem;line-height:1}
.metric .k{color:var(--mute);font-size:.8rem;font-weight:600;margin-top:8px}
.note{color:#a9a7a1;font-weight:300;line-height:1.55}

/* contact */
.contact{text-align:center;padding:150px 0 70px}
.contact .eyebrow{color:var(--mute);font-size:.72rem;font-weight:600;letter-spacing:.14em}
.contact h2{margin:10px 0 34px;font-weight:900;letter-spacing:-.06em;line-height:.9;font-size:clamp(4rem,13vw,11rem)}
.contact h2 i{font-family:var(--serif);font-weight:400;color:var(--lime);letter-spacing:-.02em}
.contact .pills{display:flex;gap:12px;justify-content:center;flex-wrap:wrap}
.contact .pills a{border:1px solid var(--line);border-radius:999px;padding:12px 24px;font-weight:600;transition:.2s}
.contact .pills a:hover{border-color:var(--lime);color:var(--lime)}
.contact .pills a.main{background:var(--lime);color:#0E0E0E;border-color:var(--lime)}
.foot{display:flex;justify-content:space-between;border-top:1px solid var(--line);padding:22px 0;color:var(--mute);font-size:.72rem;font-weight:600;letter-spacing:.08em;gap:20px;flex-wrap:wrap}

/* responsive + motion */
@media(max-width:900px){
  .nav .links a:not(.pill-btn){display:none}
  .badge{display:none}
  .card{grid-template-columns:1fr;position:relative;top:0}
  .card .r{min-height:260px}
  .about,.two,.cols{grid-template-columns:1fr}
  .stats{flex-wrap:wrap;gap:30px}
  .sec-sub{margin-left:0}
  .hero h1 .serif{margin-left:.3em}
}
@media(prefers-reduced-motion:reduce){ .badge .ring,.marquee .track{animation:none} html{scroll-behavior:auto} }
.card .r.img::before{display:none}
.card .r.img img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
/* how i work */
.pipe{position:relative;display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:20px}
.pipe .rail{position:absolute;left:0;right:0;top:29px;height:2px;background:repeating-linear-gradient(90deg,rgba(255,255,255,.14) 0 8px,transparent 8px 14px)}
.pipe .packet{position:absolute;top:24px;left:0;width:12px;height:12px;border-radius:50%;background:var(--lime);box-shadow:0 0 16px var(--lime);animation:run 7s linear infinite}
@keyframes run{from{left:0}to{left:calc(100% - 12px)}}
.stage{position:relative;z-index:1}
.node{height:58px;border:1px solid var(--line);border-radius:999px;background:#121214;display:flex;align-items:center;justify-content:space-between;
      padding:0 8px 0 22px;font-weight:800;letter-spacing:-.03em;font-size:1.1rem}
.node small{font-family:var(--serif);font-style:italic;font-weight:400;font-size:1.5rem;color:var(--lime);padding-right:12px}
.stage .box{margin-top:14px;border:1px solid var(--line);border-radius:26px;background:var(--card);padding:20px;height:calc(100% - 72px)}
.stage .box h4{margin:0 0 10px;font-family:var(--serif);font-style:italic;font-weight:400;font-size:1.5rem;line-height:1.1}
.stage .box p{margin:0 0 16px;color:#a9a7a1;font-weight:300;font-size:.9rem;line-height:1.55}
.used-lab{color:var(--mute);font-size:.7rem;font-weight:700;letter-spacing:.06em;margin-bottom:8px}
.used{display:flex;flex-wrap:wrap;gap:6px}
.used a{border:1px solid var(--line);border-radius:999px;padding:5px 11px;font-size:.72rem;font-weight:600;color:#cfcdc6}
.used a:hover{border-color:var(--lime);color:var(--lime)}

/* detector */
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line)!important;background:var(--card);border-radius:28px!important}
.metric{background:#121214}
.q{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:14px}
.q th{color:var(--mute);font-weight:700;font-size:.72rem;letter-spacing:.06em;text-align:left;padding:10px 12px;border-bottom:1px solid var(--line)}
.q td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.05);color:#cfcdc6;font-feature-settings:"tnum"}
.q .tp{color:var(--lime);font-weight:700}.q .fp{color:#FF7A6B;font-weight:700}
@media(max-width:900px){.pipe{grid-template-columns:1fr 1fr}.pipe .rail,.pipe .packet{display:none}}
@media(max-width:560px){.pipe{grid-template-columns:1fr}}
@media(prefers-reduced-motion:reduce){.pipe .packet{animation:none}}
/* links: no blue, no underline */
.stApp a,.stApp a:visited,.stApp a:hover,.stApp a:active{color:inherit!important;text-decoration:none!important}
.used a{color:#cfcdc6!important}
.nav .links a:hover,.used a:hover,.contact .pills a:hover{color:var(--lime)!important}
.pill-btn:hover,.contact .pills a.main{color:#0E0E0E!important}
a:focus-visible{outline:2px solid var(--lime);outline-offset:3px;border-radius:6px}
</style>
"""


def html(s: str) -> None:
    """Render HTML through st.markdown. Lines are joined so markdown never sees indented/blank lines."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def badge_html(text: str, radius: int = 66) -> str:
    n = len(text)
    spans = "".join(
        f'<span style="transform:rotate({i * 360 / n:.2f}deg) translate(-.4em,-{radius}px)">{c}</span>'
        for i, c in enumerate(text)
    )
    return f'<div class="badge"><div class="ring">{spans}</div><div class="arrow">↓</div></div>'


@st.cache_data(show_spinner=False)
def _to_uri(path: str, mtime: float, mime: str) -> str:
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def find_img(stem: str):
    """Look in assets/ for <stem>.png/.jpg/.jpeg/.webp and return it as an inline data URI (or None)."""
    for ext, mime in (("png", "png"), ("jpg", "jpeg"), ("jpeg", "jpeg"), ("webp", "webp")):
        p = Path(__file__).parent / "assets" / f"{stem}.{ext}"
        if p.exists():
            return _to_uri(str(p), p.stat().st_mtime, mime)
    return None


def photo_html() -> str:
    uri = find_img("photo")       
    if uri:
        return f'<div class="photo"><img src="{uri}" alt="{NAME}"/></div>'
    return f'<div class="photo"><div class="ph">{NAME[0]}</div></div>'


def sec_head(no: str, title: str, italic: str, sub: str = "") -> str:
    s = f'<div class="sec-head"><div class="no">{no}</div><h2>{title} <i>{italic}</i></h2></div>'
    return s + (f'<p class="sec-sub">{sub}</p>' if sub else "")


def pipeline_html() -> str:
    short = {p["id"]: p["short"] for p in PROJECTS}
    out = '<div class="pipe"><div class="rail"></div><div class="packet"></div>'
    for i, st_ in enumerate(STAGES, 1):
        used = "".join(f'<a href="#{u}">{short[u]}</a>' for u in st_["used"])
        out += (f'<div class="stage"><div class="node">{st_["name"]}<small>0{i}</small></div>'
                f'<div class="box"><h4>{st_["line"]}</h4><p>{st_["did"]}</p>'
                f'<div class="used-lab">USED IN</div><div class="used">{used}</div></div></div>')
    return out + "</div>"


# ════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════
st.markdown(CSS, unsafe_allow_html=True)

# nav
html(f"""
<nav class="nav">
  <a class="logo" href="#top">{NAME[0]}<i>{NAME[-1]}</i></a>
  <div class="links">
    <a href="#work">Work</a><a href="#process">How I work</a><a href="#about">About</a><a href="#detector">Detector</a><a href="#contact">Contact</a>
    <a class="pill-btn" href="{RESUME}" target="_blank">Résumé ↗</a>
  </div>
</nav>
""")

# hero
html(f"""
<section id="top" class="hero">
  <div><h1>{NAME}<span class="serif">{SUBTITLE}</span></h1></div>
  {badge_html(BADGE_TEXT)}
  <div class="hero-bottom"><p class="intro">{INTRO}</p><span class="scroll">SCROLL TO EXPLORE</span></div>
</section>
""")

# marquee
items = "".join(
    f'<span class="t{" s" if i % 2 == 0 else ""}">{m}</span><span class="star">✦</span>' for i, m in enumerate(MARQUEE)
)
html(f'<div class="marquee"><div class="track">{items}{items}</div></div>')

# work
html('<section id="work"></section>' + sec_head(
    "01", "Selected", "work", "Five projects across healthcare, finance, security, language and vision, each starting from a real problem."))

cards = ""
for i, p in enumerate(PROJECTS, 1):
    pts = "".join(f"<li>{x}</li>" for x in p["points"])
    chips = "".join(f'<span class="chip">{t}</span>' for t in p["tags"])
    uri = find_img(p["image"]) if p.get("image") else None
    visual = (f'<div class="r img"><img src="{uri}" alt="{p["title"]}"/></div>' if uri else
              f'<div class="r"><div class="big">{p["stat"]}</div><div class="cap">{p["stat_label"]}</div></div>')
    cards += f"""
    <article class="card" id="{p['id']}" style="--c:{p['accent']}">
      <div class="l">
        <div class="meta"><b>0{i}</b><span>{p['kind']}</span></div>
        <h3>{p['title']}</h3>
        <div class="tag">{p['tagline']}</div>
        <p class="d">{p['desc']}</p>
        <ul>{pts}</ul>
        <div class="chips">{chips}</div>
        <a class="go" href="{p['link']}" target="_blank">View on GitHub <span>↗</span></a>
      </div>
      {visual}
    </article>"""
html(f'<div class="stack">{cards}</div>')

# how i work
html('<section id="process"></section>' + sec_head(
    "02", "How I", "work", "Every project moves through the same six stages. Click a project under any stage to jump to it.")
    + pipeline_html())

# about
edu = "".join(
    f'<div class="row"><div><h4>{d}</h4><small>{s}</small></div><div class="y">{y}<small>{g}</small></div></div>'
    for d, s, y, g in EDUCATION)
exp_pts = "".join(f"<li>{x}</li>" for x in EXPERIENCE["points"])
tools = "".join(f'<span class="tool">{t}</span>' for t in TOOLS)
certs = "<br>".join(f"✦ {c}" for c in CERTS)

html('<section id="about"></section>' + sec_head("03", "Me", "who?") + f"""
<div class="about">
  {photo_html()}
  <div>
    <h3>{ABOUT_STATEMENT}</h3>
    <div class="cols"><p>{ABOUT_P1}</p><p>{ABOUT_P2}</p></div>
  </div>
</div>
<div class="two">
  <div>
    <div class="lab">EDUCATION</div>{edu}
    <div class="certs"><div class="lab">CERTIFICATIONS</div>{certs}</div>
  </div>
  <div>
    <div class="lab">EXPERIENCE</div>
    <div class="row exp"><div><h4>{EXPERIENCE['role']}</h4><small>{EXPERIENCE['org']}</small><ul>{exp_pts}</ul></div>
    <div class="y">{EXPERIENCE['when']}</div></div>
    <div class="lab" style="margin-top:44px">TOOLS &amp; LANGUAGES</div>
    <div class="tools">{tools}</div>
  </div>
</div>
""")

# detector: live demo
html('<section id="detector"></section>' + sec_head(
    "04", "Try the", "detector",
    "A fraud model on synthetic data where only about 2% of transactions are fraud. Pick a model, move the "
    "threshold, and see what it catches and what it gets wrong."))


@st.cache_data(show_spinner="Training models…")
def train_models():
    X, y = make_classification(n_samples=30000, n_features=12, n_informative=6, weights=[0.98, 0.02],
                               class_sep=1.0, flip_y=0.005, random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=7)
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=7),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=7),
    }
    return yte, {n: m.fit(Xtr, ytr).predict_proba(Xte)[:, 1] for n, m in models.items()}


def scores(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


y_true, all_proba = train_models()
with st.container(border=True):
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        model = st.radio("Model", list(all_proba))
        thr = st.slider("Decision threshold", 0.05, 0.99, 0.50, 0.01,
                        help="Flag a transaction as fraud when the model's score is at or above this value.")
    proba = all_proba[model]
    prec, rec, f1, tp, fp, fn = scores(y_true, proba, thr)
    with left:
        c1, c2, c3 = st.columns(3)
        for col, val, lab in ((c1, prec, "Precision"), (c2, rec, "Recall"), (c3, f1, "F1")):
            col.markdown(f'<div class="metric"><div class="n">{val:.2f}</div><div class="k">{lab}</div></div>',
                         unsafe_allow_html=True)
        st.markdown(
            f'<p class="note" style="margin-top:18px">At <b>{thr:.2f}</b>, {model} catches <b>{tp}</b> of '
            f'<b>{tp + fn}</b> fraud cases and raises <b>{fp}</b> false alarms. Accuracy would look great here '
            f'even if it caught nothing, which is why I report precision and recall.</p>', unsafe_allow_html=True)
    with right:
        rows = []
        for g in np.linspace(0.05, 0.99, 60):
            pr, rc, *_ = scores(y_true, proba, g)
            rows += [{"threshold": float(g), "score": pr, "metric": "Precision"},
                     {"threshold": float(g), "score": rc, "metric": "Recall"}]
        base = alt.Chart(alt.Data(values=rows)).encode(
            x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0.05, 0.99])),
            y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(domain=["Precision", "Recall"], range=["#D9FF5A", "#7AA8FF"]),
                            legend=alt.Legend(orient="top", title=None)))
        rule = alt.Chart(alt.Data(values=[{"t": thr}])).mark_rule(color="#FF7A6B", strokeDash=[4, 4]).encode(x="t:Q")
        chart = (base.mark_line(strokeWidth=3) + rule).properties(height=300).configure(
            background="transparent").configure_view(strokeWidth=0).configure_axis(
            gridColor="#ffffff14", labelColor="#8D8B86", titleColor="#8D8B86", domainColor="#ffffff22"
        ).configure_legend(labelColor="#EAE8E1")
        st.altair_chart(chart, width="stretch")

    # alert queue: highest-scoring flagged transactions, with the ground truth revealed
    idx = np.where(proba >= thr)[0]
    idx = idx[np.argsort(-proba[idx])][:8]
    if len(idx):
        trs = "".join(
            f'<tr><td>TX-{10000 + int(i)}</td><td>{proba[i]:.3f}</td>'
            f'<td class="{"tp" if y_true[i] == 1 else "fp"}">{"Fraud caught" if y_true[i] == 1 else "False alarm"}</td></tr>'
            for i in idx)
        html(f'<table class="q"><tr><th>ALERT QUEUE (TOP 8)</th><th>SCORE</th><th>OUTCOME</th></tr>{trs}</table>')
    else:
        st.markdown('<p class="note">Nothing flagged at this threshold. Lower it to see alerts.</p>', unsafe_allow_html=True)

# contact + footer
html(f"""
<section id="contact" class="contact">
  <div class="eyebrow">OPEN TO ML INTERNSHIPS AND COLLABORATIONS</div>
  <h2>Say <i>Hi!!</i></h2>
  <div class="pills">
    <a class="main" href="mailto:{EMAIL}">Email</a>
    <a href="{LINKEDIN}" target="_blank">LinkedIn</a>
    <a href="{GITHUB}" target="_blank">GitHub</a>
    <a href="{RESUME}" target="_blank">Résumé ↗</a>
    <a href="{LINKS}" target="_blank">More links ↗</a>
  </div>
</section>
<div class="foot"><span>© 2026 {NAME.upper()}</span><span>BUILT WITH PYTHON AND STREAMLIT</span><a href="#top">BACK TO TOP ↑</a></div>
""")