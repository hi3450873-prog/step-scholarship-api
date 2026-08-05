"""
STEP — AI Scholarship Matcher (MySQL-backed, scikit-learn)
==========================================================
"""
import os
import json
import urllib.parse
import mysql.connector
import numpy as np

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

load_dotenv()


# --------------------------------------------------------------------------
# DB connection
# --------------------------------------------------------------------------

def _parse_mysql_url(url):
    assert url.startswith("mysql://"), "MYSQL_URL must start with mysql://"
    rest = url[len("mysql://"):]
    creds, hostpart = rest.split("@", 1)
    user, _, pwd = creds.partition(":")
    pwd = urllib.parse.unquote(pwd)
    if "/" in hostpart:
        hostport, _, dbname = hostpart.partition("/")
        dbname = dbname.split("?")[0]
    else:
        hostport, dbname = hostpart, None
    host, _, port = hostport.partition(":")
    return {
        "host": host,
        "port": int(port or 3306),
        "user": user,
        "password": pwd,
        "database": dbname,
    }


def _connect():
    """Connect to Railway MySQL using MYSQL_URL from the .env file."""
    url = os.getenv("MYSQL_URL")

    if not url:
        raise RuntimeError(
            "MYSQL_URL not found.\n"
            "Create a .env file containing:\n"
            "MYSQL_URL=mysql://user:password@host:port/database"
        )

    return mysql.connector.connect(**_parse_mysql_url(url))


# --------------------------------------------------------------------------
# Load scholarships from MySQL
# --------------------------------------------------------------------------

def load_scholarships_from_db():
    """Return a list of scholarship dicts."""
    conn = _connect()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT * FROM scholarships
        ORDER BY scholarship_id
    """)
    rows = cur.fetchall()

    scholarships = []
    for r in rows:
        s = {
            "scholarship_id": r["scholarship_id"],
            "scholarship_name": r["scholarship_name"],
            "provider": r.get("provider"),
            "scholarship_type": r.get("scholarship_type"),
            "category": r.get("category"),
            "description": r.get("description"),
            "application_status": r.get("application_status"),
            "application_start": r.get("application_start"),
            "application_end": r.get("application_end"),
            "duration": r.get("duration"),
            "official_website": r.get("official_website"),
            "notes": r.get("notes"),
            "requirements": {
                "min_gwa": float(r["min_gwa"]) if r.get("min_gwa") is not None else None,
                "max_income": float(r["max_income"]) if r.get("max_income") is not None else None,
                "is_4ps_beneficiary": bool(r.get("is_4ps_beneficiary")),
                "pwd": bool(r.get("pwd")),
                "indigenous_people": bool(r.get("indigenous_people")),
                "region": r.get("region"),
                "ofw": bool(r.get("ofw")),
                "cooperative": bool(r.get("cooperative")),
                "merit_based": bool(r.get("merit_based")),
                "financial_based": bool(r.get("financial_based")),
                "university_based": bool(r.get("university_based")),
                "partner_university": bool(r.get("partner_university")),
                "has_priority_courses": bool(r.get("has_priority_courses")),
            },
            "benefits": [],
            "priority_courses": [],
        }
        scholarships.append(s)

    # Benefits
    cur.execute("""
        SELECT scholarship_id, benefit_id, benefit_name, category,
               amount, frequency, remarks
        FROM scholarship_benefits
    """)
    ben_by_sch = {}
    for b in cur.fetchall():
        ben_by_sch.setdefault(b["scholarship_id"], []).append({
            "benefit_id": b["benefit_id"],
            "benefit_name": b["benefit_name"],
            "category": b["category"],
            "amount": b["amount"],
            "frequency": b["frequency"],
            "remarks": b["remarks"],
        })
    for s in scholarships:
        s["benefits"] = ben_by_sch.get(s["scholarship_id"], [])

    # Priority courses
    cur.execute("""
        SELECT scholarship_id, course_name
        FROM scholarship_priority_courses
    """)
    pc_by_sch = {}
    for pc in cur.fetchall():
        pc_by_sch.setdefault(pc["scholarship_id"], []).append(pc["course_name"])
    for s in scholarships:
        s["priority_courses"] = pc_by_sch.get(s["scholarship_id"], [])

    cur.close()
    conn.close()
    return scholarships


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

NUMERIC_FIELDS = [
    "min_gwa", "max_income", "merit_based", "financial_based",
    "university_based", "partner_university", "is_4ps_beneficiary",
    "pwd", "indigenous_people", "ofw", "cooperative", "has_priority_courses",
]


def _to_float(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _to_bool(v):
    return 1.0 if bool(v) else 0.0


def _courses_to_text(scholarship):
    return " ".join(scholarship.get("priority_courses") or []).lower()


def _scholarship_feature_vector(scholarship):
    r = scholarship.get("requirements") or {}
    vec = []
    for field in NUMERIC_FIELDS:
        v = r.get(field)
        if field in ("min_gwa", "max_income"):
            vec.append(_to_float(v))
        else:
            vec.append(_to_bool(v))
    return vec


def _student_feature_vector(student):
    return [
        _to_float(student.get("gwa")),
        _to_float(student.get("family_income")),
        _to_bool(student.get("merit_seeking", True)),
        _to_bool(student.get("need_based", (student.get("family_income") or 0) <= 300000)),
        _to_bool(student.get("university_based", False)),
        _to_bool(student.get("partner_university", False)),
        _to_bool(student.get("is_4ps_beneficiary")),
        _to_bool(student.get("pwd")),
        _to_bool(student.get("indigenous_people")),
        _to_bool(student.get("ofw")),
        _to_bool(student.get("cooperative", False)),
        _to_bool(True if student.get("course") else False),
    ]


# --------------------------------------------------------------------------
# Eligibility (hard rules)
# --------------------------------------------------------------------------

def _is_eligible(scholarship, student):
    r = scholarship.get("requirements") or {}
    
    # -----------------------------
    # GWA Check
    # -----------------------------
    min_gwa = r.get("min_gwa")
    student_gwa = student.get("gwa")

    if min_gwa is not None and min_gwa <= 1:
        min_gwa *= 100

    if student_gwa is not None and student_gwa <= 1:
        student_gwa *= 100

    if min_gwa is not None and student_gwa is not None:
        if student_gwa < min_gwa:
            return False, f"GWA ({student_gwa}%) below requirement ({min_gwa}%)"

    # -----------------------------
    # Income Check
    # -----------------------------
    max_income = r.get("max_income")
    if max_income is not None:
        income = student.get("family_income")
        if income is not None and income > max_income:
            return False, f"Income exceeds ₱{max_income:,.0f}"

    # -----------------------------
    # Course Check
    # -----------------------------
    courses = scholarship.get("priority_courses") or []
    if courses:
        student_course = (student.get("course") or "").strip().lower()
        if student_course:
            found = any(
                student_course in c.lower() or c.lower() in student_course
                for c in courses
            )
            if not found:
                return False, "Course not eligible"

    # -----------------------------
    # Special Category Checks
    # -----------------------------
    if r.get("is_4ps_beneficiary") and not student.get("is_4ps_beneficiary"):
        return False, "For 4Ps beneficiaries only"

    if r.get("pwd") and not student.get("pwd"):
        return False, "PWD required"

    if r.get("indigenous_people") and not student.get("indigenous_people"):
        return False, "Indigenous People required"

    if r.get("ofw") and not student.get("ofw"):
        return False, "OFW dependent required"

    if r.get("cooperative") and not student.get("cooperative"):
        return False, "Cooperative member required"

    # -----------------------------
    # Region Check
    # -----------------------------
    region = r.get("region")
    if region and str(region).strip().lower() != "any":
        student_region = (student.get("region") or "").strip().lower()
        if student_region and student_region != str(region).strip().lower():
            return False, f"Only for {region}"

    return True, "Eligible"


# --------------------------------------------------------------------------
# Matcher
# --------------------------------------------------------------------------

class ScholarshipMatcher:
    """Scikit-learn matcher backed by MySQL."""

    def __init__(self, db_url=None):
        self._db_url = db_url
        self.scholarships = []
        self._course_vectorizer = None
        self._scaler = None
        self._scholarship_matrix = None
        self._fitted = False

    def _load(self):
        if self._db_url:
            os.environ["MYSQL_URL"] = self._db_url
        self.scholarships = load_scholarships_from_db()

    def fit(self):
        self._load()

        dense = np.array(
            [_scholarship_feature_vector(s) for s in self.scholarships],
            dtype=float
        )
        dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)

        course_text = [_courses_to_text(s) for s in self.scholarships]
        self._course_vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[\w&/.\-]+\b",
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = self._course_vectorizer.fit_transform(course_text)

        self._scaler = StandardScaler()
        dense_scaled = self._scaler.fit_transform(dense)

        self._scholarship_matrix = hstack(
            [csr_matrix(dense_scaled), tfidf]
        ).tocsr()

        self._fitted = True
        return self

    def recommend(self, student, top_n=10, min_match=80, include_ineligible=False):
        if not self._fitted:
            raise RuntimeError("Matcher is not fitted. Call .fit() first.")

        eligibility = [
            (i, *_is_eligible(s, student)) for i, s in enumerate(self.scholarships)
        ]

        student_dense = np.array(_student_feature_vector(student), dtype=float)
        student_dense = np.nan_to_num(student_dense, nan=0.0, posinf=0.0, neginf=0.0)
        student_dense_scaled = self._scaler.transform(student_dense.reshape(1, -1))

        student_course_text = (student.get("course") or "").lower()
        student_tfidf = self._course_vectorizer.transform([student_course_text])

        student_matrix = hstack(
            [csr_matrix(student_dense_scaled), student_tfidf]
        ).tocsr()

        sims = cosine_similarity(student_matrix, self._scholarship_matrix)[0]

        rows = []

        for idx, eligible, reason in eligibility:
            s = self.scholarships[idx]

            if not eligible and not include_ineligible:
                continue

            # -----------------------------
            # Weighted Match Score
            # -----------------------------
            score = 40.0
            req = s.get("requirements", {})

            # GWA (20%)
            min_gwa = req.get("min_gwa")
            if min_gwa is not None and min_gwa <= 1:
                min_gwa *= 100

            if min_gwa is None:
                score += 10  # Reduced partial credit for open requirements
            elif student.get("gwa") is not None:
                if student["gwa"] >= min_gwa:
                    score += 20
                else:
                    score += max(0, 20 * (student["gwa"] / min_gwa))

            # Income (15%)
            max_income = req.get("max_income")
            if max_income is None:
                score += 10  # Partial credit if open to all incomes
            elif student.get("family_income") is not None:
                if student["family_income"] <= max_income:
                    score += 15

            # Course (15%)
            courses = s.get("priority_courses") or []
            if not courses:
                score += 10  # Partial credit if open to all courses
            else:
                student_course = (student.get("course") or "").lower()
                if any(student_course in c.lower() for c in courses):
                    score += 15

            # Region (5%)
            region = req.get("region")
            if region is None or str(region).lower() == "any":
                score += 5
            elif student.get("region"):
                if region.lower() == student["region"].lower():
                    score += 5

            # Special qualifications (5%)
            special = 5
            checks = [
                ("is_4ps_beneficiary", "is_4ps_beneficiary"),
                ("pwd", "pwd"),
                ("indigenous_people", "indigenous_people"),
                ("ofw", "ofw"),
                ("cooperative", "cooperative"),
            ]

            for db_field, student_field in checks:
                if req.get(db_field):
                    if not student.get(student_field):
                        special -= 1

            score += max(0, special)

            # Cosine similarity AI bonus (0–5%)
            score += max(0, float(sims[idx])) * 5

            match_pct = round(min(score, 100))

            # Filter out matches below threshold
            if match_pct < min_match:
                continue

            rows.append({
                "scholarship_id": s.get("scholarship_id"),
                "scholarship_name": s.get("scholarship_name"),
                "provider": s.get("provider"),
                "scholarship_type": s.get("scholarship_type"),
                "category": s.get("category"),
                "score": round(match_pct / 100.0, 4),
                "match_percentage": match_pct,
                "eligible": eligible,
                "match_reason": reason,
                "official_website": s.get("official_website"),
            })

        rows.sort(key=lambda x: x["match_percentage"], reverse=True)
        return rows[:top_n]


if __name__ == "__main__":
    print("Connecting to Railway...")
    matcher = ScholarshipMatcher()
    matcher.fit()

    print(" Connected successfully!")
    print(f"Loaded {len(matcher.scholarships)} scholarships")

    for s in matcher.scholarships[:5]:
        print("-", s["scholarship_name"])